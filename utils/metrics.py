from re import T
from time import time
import logging
import torch
import numpy as np
import os
from utils.reranking import re_ranking
from utils.class_aware import apply_class_distance_penalty, class_distance_penalty_matrix
from utils.inference_postprocess import apply_query_expansion
from utils.group_rerank import (
    apply_group_rerank_to_distmat,
    class_names_for_probabilities,
    class_group_mapping,
    group_rerank_mode,
    log_group_rerank,
    rank_change_examples,
    resolve_query_gallery_classes,
)


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(
        self,
        num_query,
        max_rank=50,
        feat_norm=True,
        reranking=False,
        class_penalty=0.0,
        query_expansion=False,
        qe_topk=5,
        qe_alpha=1.0,
        cfg=None,
        dataset_name=None,
    ):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.class_penalty = class_penalty
        self.query_expansion = query_expansion
        self.qe_topk = qe_topk
        self.qe_alpha = qe_alpha
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.last_group_info = None
        self.last_group_examples = []

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.pred_classes = []
        self.pred_class_probs = []
        self.img_paths = []

    def update(self, output):  # called once for each batch
        pred_prob = None
        img_path = None
        if len(output) == 6:
            feat, pid, camid, pred_class, pred_prob, img_path = output
        elif len(output) == 5:
            feat, pid, camid, pred_class, img_path = output
        elif len(output) == 4:
            feat, pid, camid, pred_class = output
        else:
            feat, pid, camid = output
            pred_class = None
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        if pred_class is not None:
            self.pred_classes.extend(np.asarray(pred_class))
        if pred_prob is not None:
            self.pred_class_probs.extend(np.asarray(pred_prob))
        if img_path is not None:
            self.img_paths.extend(list(img_path))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            # print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.query_expansion:
            qf, gf = apply_query_expansion(qf, gf, self.qe_topk, self.qe_alpha)

        q_classes = None
        g_classes = None
        penalty_matrix = None
        if self.pred_classes:
            q_classes = np.asarray(self.pred_classes[:self.num_query])
            g_classes = np.asarray(self.pred_classes[self.num_query:])
            penalty_matrix = class_distance_penalty_matrix(q_classes, g_classes, self.class_penalty)

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            local_distmat = None
            if penalty_matrix is not None:
                local_distmat = np.zeros((feats.shape[0], feats.shape[0]), dtype=np.float32)
                local_distmat[:self.num_query, self.num_query:] = penalty_matrix
                local_distmat[self.num_query:, :self.num_query] = penalty_matrix.T
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3, local_distmat=local_distmat)

        else:
            # print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
            distmat = apply_class_distance_penalty(distmat, q_classes, g_classes, self.class_penalty)
        base_distmat = np.asarray(distmat, dtype=np.float32).copy()
        if self.cfg is not None and group_rerank_mode(self.cfg) != "none":
            resolved_q_classes, resolved_g_classes, class_source = resolve_query_gallery_classes(
                self.cfg,
                np.asarray(self.pred_classes) if self.pred_classes else None,
                self.img_paths,
                self.num_query,
            )
            if resolved_q_classes is not None and resolved_g_classes is not None:
                q_classes = resolved_q_classes
                g_classes = resolved_g_classes

            q_class_probs = None
            if len(self.pred_class_probs) == len(self.pids):
                q_class_probs = np.asarray(self.pred_class_probs[:self.num_query], dtype=np.float32)
            distmat, self.last_group_info = apply_group_rerank_to_distmat(
                distmat,
                q_classes,
                g_classes,
                self.cfg,
                q_class_probs=q_class_probs,
                class_names=class_names_for_probabilities(self.cfg),
            )
            self.last_group_info["resolved_class_source"] = class_source
            self.last_group_examples = rank_change_examples(
                base_distmat,
                distmat,
                self.img_paths,
                self.num_query,
                q_classes,
                g_classes,
                limit=int(getattr(self.cfg.TEST, "LOG_RANK_CHANGES", 5)),
                mapping=class_group_mapping(self.cfg),
            )
            log_group_rerank(
                logging.getLogger("PAT.test"),
                self.last_group_info,
                self.last_group_examples,
            )
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
