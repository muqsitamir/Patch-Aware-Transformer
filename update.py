import os
import csv
import torch
import argparse
import logging

import numpy as np
from config import cfg
from model import make_model
from utils.logger import setup_logger
from utils.re_ranking import re_ranking
from data.build_DG_dataloader import build_reid_test_loader
from processor.ori_vit_processor_with_amp import do_inference as do_inf
from processor.part_attention_vit_processor import do_inference as do_inf_pat
from utils.class_aware import (
    class_distance_penalty_matrix,
    get_class_csv_name,
    get_class_distance_penalty,
    get_batch_class_targets,
    infer_num_semantic_classes,
    model_is_class_aware,
    read_class_csv,
    use_metadata_classes_for_retrieval,
)
from utils.tta import extract_tta_features, log_tta_settings
from utils.inference_postprocess import (
    apply_query_expansion,
    build_class_postprocess_indices,
    class_postprocess_stats,
)


SUBMISSION_TOPK = 100

#from torch.backends import cudnn

def extract_feature(model, dataloaders, num_query, cfg):
    features = []
    pred_classes = []
    count = 0
    img_paths = []
    use_class_aware = model_is_class_aware(model)
    use_metadata_classes = use_metadata_classes_for_retrieval(cfg)
    logger = logging.getLogger("PAT")
    log_tta_settings(logger, cfg)
    model.eval()

    for data in dataloaders:
        img = data['images']
        #obtain values form dict data
        n, c, h, w = img.size()
        count += n
        input_img = img.cuda()
        if use_class_aware and not use_metadata_classes:
            ff, class_logits = extract_tta_features(
                model, input_img, cfg, return_class_logits=True
            )
        else:
            ff, class_logits = extract_tta_features(model, input_img, cfg)
        features.append(ff)
        if use_metadata_classes:
            class_targets = get_batch_class_targets(data)
            if class_targets is not None:
                pred_classes.append(class_targets.cpu())
        elif use_class_aware and class_logits is not None:
            pred_classes.append(class_logits.argmax(1).cpu())
        img_paths.extend(data.get('img_path', []))
    features = torch.cat(features, 0)
    if pred_classes:
        pred_classes = torch.cat(pred_classes, 0).numpy()
    else:
        pred_classes = None

    # query
    qf = features[:num_query]
    # gallery
    gf = features[num_query:]
    q_pred_classes = pred_classes[:num_query] if pred_classes is not None else None
    g_pred_classes = pred_classes[num_query:] if pred_classes is not None else None
    q_img_paths = img_paths[:num_query]
    g_img_paths = img_paths[num_query:]
    return qf, gf, q_pred_classes, g_pred_classes, q_img_paths, g_img_paths


def dataset_root(cfg):
    dataset_mode = str(getattr(cfg.DATASETS, "MODE", "challenge_only")).lower()
    if dataset_mode == "external_only":
        root = getattr(cfg.DATASETS, "EXTERNAL_ROOT", "")
    else:
        root = getattr(cfg.DATASETS, "ROOT_DIR", "")
    if isinstance(root, (tuple, list)):
        root = root[0]
    return str(root)


def classes_for_paths(image_to_class, image_paths):
    if not image_to_class:
        return None

    classes = []
    found = 0
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        class_name = image_to_class.get(image_path) or image_to_class.get(image_name)
        if class_name is None:
            classes.append("")
        else:
            classes.append(class_name)
            found += 1

    if found == 0:
        return None
    return np.asarray(classes, dtype=object)


def load_csv_submission_classes(cfg, q_img_paths, g_img_paths):
    root = dataset_root(cfg)
    query_csv = get_class_csv_name(cfg, "QUERY_CSV", "query_classes.csv")
    test_csv = get_class_csv_name(cfg, "TEST_CSV", "test_classes.csv")
    query_classes = classes_for_paths(read_class_csv(os.path.join(root, query_csv)), q_img_paths)
    gallery_classes = classes_for_paths(read_class_csv(os.path.join(root, test_csv)), g_img_paths)
    if query_classes is None or gallery_classes is None:
        return None, None
    return query_classes, gallery_classes


def select_submission_classes(q_csv_classes, g_csv_classes, q_pred_classes, g_pred_classes):
    if q_csv_classes is not None and g_csv_classes is not None:
        return q_csv_classes, g_csv_classes, "csv"
    if q_pred_classes is not None and g_pred_classes is not None:
        return q_pred_classes, g_pred_classes, "model"
    return None, None, "none"


def class_mismatch_penalty(cfg):
    test_penalty = float(getattr(cfg.TEST, "CLASS_MISMATCH_PENALTY", 0.0))
    if test_penalty > 0:
        return test_penalty
    return get_class_distance_penalty(cfg)


def save_feature_files(cfg, qf, gf):
    if not bool(getattr(cfg.TEST, "SAVE_FEATURES", True)):
        return

    for path, features in ((cfg.TEST.FEAT_Q_PATH, qf), (cfg.TEST.FEAT_G_PATH, gf)):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.save(path, features)


def _format_class_counts(counts):
    if not counts:
        return "none"
    return ", ".join("{}={}".format(key, counts[key]) for key in sorted(counts))


def log_class_postprocess_stats(logger, q_classes, g_classes, class_topk):
    stats = class_postprocess_stats(q_classes, g_classes, class_topk)
    if stats is None:
        logger.info("Class postprocess stats: unavailable (missing usable query/gallery class labels).")
        return

    logger.info(
        "Class postprocess stats: gallery images per class: {}".format(
            _format_class_counts(stats["gallery_counts"])
        )
    )
    logger.info(
        "Class postprocess stats: same-class gallery items per query class: {}".format(
            _format_class_counts(stats["query_class_counts"])
        )
    )
    logger.info(
        "Class postprocess stats: same-class gallery items across queries: "
        "min={} max={} avg={:.2f}".format(
            stats["same_count_min"],
            stats["same_count_max"],
            stats["same_count_avg"],
        )
    )
    logger.info(
        "Class postprocess stats: queries requiring backfill (< {} same-class gallery items): {}/{}".format(
            stats["class_topk"],
            stats["backfill_query_count"],
            stats["num_queries"],
        )
    )
    if stats["missing_query_classes"] or stats["missing_gallery_classes"]:
        logger.info(
            "Class postprocess stats: missing/invalid labels: queries={} gallery={}".format(
                stats["missing_query_classes"],
                stats["missing_gallery_classes"],
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/PAT.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        "--track", default="./config/PAT.yml", help="path to config file", type=str
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0,0,0, num_semantic_class=infer_num_semantic_classes(cfg))
    model.load_param(cfg.TEST.WEIGHT)

    for testname in cfg.DATASETS.TEST:
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        if cfg.MODEL.NAME == 'part_attention_vit':
            do_inf_pat(cfg, model, val_loader, num_query, dataset_name=testname)
        else:
            do_inf(cfg, model, val_loader, num_query, dataset_name=testname)
    with torch.no_grad():
        qf, gf, q_pred_classes, g_pred_classes, q_img_paths, g_img_paths = extract_feature(model, val_loader, num_query, cfg)

    q_csv_classes, g_csv_classes = load_csv_submission_classes(cfg, q_img_paths, g_img_paths)
    q_classes, g_classes, class_source = select_submission_classes(
        q_csv_classes,
        g_csv_classes,
        q_pred_classes,
        g_pred_classes,
    )
    logger.info("Submission class source: {}".format(class_source))

    # Features are L2-normalized by extract_feature. If query expansion is
    # enabled, qf is replaced by the normalized query-expanded representation
    # before ranking and optional feature saving.
    qf = qf.cpu().numpy()
    gf = gf.cpu().numpy()
    if cfg.TEST.QUERY_EXPANSION:
        qf, gf = apply_query_expansion(qf, gf, cfg.TEST.QE_TOPK, cfg.TEST.QE_ALPHA)
    save_feature_files(cfg, qf, gf)

    q_g_dist = np.dot(qf, np.transpose(gf))
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))

    class_postprocess = str(cfg.TEST.CLASS_POSTPROCESS).lower()
    class_score_mode = str(cfg.TEST.CLASS_SCORE_MODE).lower()
    mismatch_penalty = class_mismatch_penalty(cfg)
    penalty_matrix = None
    postprocess_mode = class_postprocess
    if class_postprocess != "off":
        log_class_postprocess_stats(logger, q_classes, g_classes, SUBMISSION_TOPK)
    if class_postprocess == "penalty_additive" and class_score_mode == "additive":
        penalty_matrix = class_distance_penalty_matrix(q_classes, g_classes, mismatch_penalty)
        # The legacy additive mode biases the k-reciprocal re-ranking distance
        # directly. After that, argsort preserves the final re-ranked order.
        postprocess_mode = "off"

    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, q_g_penalty=penalty_matrix)

    indices = build_class_postprocess_indices(
        re_rank_dist,
        q_classes,
        g_classes,
        mode=postprocess_mode,
        output_topk=SUBMISSION_TOPK,
        class_topk=cfg.TEST.CLASS_TOPK,
        mismatch_penalty=mismatch_penalty,
        score_mode=class_score_mode,
        scale=cfg.TEST.CLASS_SCALE,
    )

    m, n = indices.shape
    # # print('m: {}  n: {}'.format(m, n))
    with open(args.track, 'wb') as f_w:
        for i in range(m):
            write_line = indices[i] + 1
            write_line = ' '.join(map(str, write_line.tolist())) + '\n'
            f_w.write(write_line.encode())


    lista_nombres = ["{:06d}.jpg".format(i) for i in range(1, len(indices) + 1)]
    output_path = args.track.split(".txt")[0] + "_submission.csv"

    with open(output_path, 'w', newline='') as archivo_csv:
        csv_writter = csv.writer(archivo_csv)
        csv_writter.writerow(['imageName', 'Corresponding Indexes'])
        for numero, track in zip(lista_nombres, indices):
            track_str = ' '.join(map(str, track + 1))
            csv_writter.writerow([numero, track_str])
