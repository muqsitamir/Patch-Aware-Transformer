#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import cfg
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from utils.class_aware import infer_num_semantic_classes
from utils.inference_postprocess import build_class_postprocess_indices
from utils.logger import setup_logger
from utils.re_ranking import re_ranking


SUBMISSION_TOPK = 100


def parse_layers(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def normalize(x):
    return torch.nn.functional.normalize(x.float(), p=2, dim=1)


def extract_intermediate_features(model, images, layers, include_parts=False):
    tokens = model.base(images)
    chunks = []
    for layer in layers:
        cls = tokens[layer][:, 0]
        chunks.append(normalize(cls))
        if include_parts:
            parts = tokens[layer][:, 1:4].mean(dim=1)
            chunks.append(normalize(parts))
    return normalize(torch.cat(chunks, dim=1))


def extract_features(model, dataloader, num_query, layers, include_parts, log_period=20):
    model.eval()
    features = []
    start = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader, start=1):
            images = data["images"].cuda(non_blocking=True)
            feat = extract_intermediate_features(model, images, layers, include_parts=include_parts)
            features.append(feat.cpu())
            if log_period > 0 and batch_idx % log_period == 0:
                seen = sum(chunk.shape[0] for chunk in features)
                elapsed = max(time.time() - start, 1e-6)
                print(
                    "extracted {}/{} batches, {} images, {:.1f} img/s".format(
                        batch_idx, len(dataloader), seen, seen / elapsed
                    ),
                    flush=True,
                )
    features = torch.cat(features, dim=0).numpy()
    return features[:num_query], features[num_query:]


def save_feature_files(cfg, qf, gf):
    if not bool(getattr(cfg.TEST, "SAVE_FEATURES", True)):
        return
    for path, features in ((cfg.TEST.FEAT_Q_PATH, qf), (cfg.TEST.FEAT_G_PATH, gf)):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.save(path, features)


def write_outputs(track_path, indices):
    parent = os.path.dirname(track_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(track_path, "w") as f_w:
        for row in indices:
            f_w.write(" ".join(map(str, (row + 1).tolist())) + "\n")
    output_path = track_path.split(".txt")[0] + "_submission.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imageName", "Corresponding Indexes"])
        for i, row in enumerate(indices, start=1):
            writer.writerow(["{:06d}.jpg".format(i), " ".join(map(str, (row + 1).tolist()))])
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Export submission using intermediate ViT layer features.")
    parser.add_argument("--config_file", default="./config/UrbanElementsReID_test.yml")
    parser.add_argument("--track", required=True)
    parser.add_argument("--layers", default="-1,-2,-3,-4")
    parser.add_argument("--include-parts", action="store_true")
    parser.add_argument("--log-period", type=int, default=20)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    layers = parse_layers(args.layers)

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("PAT.intermediate_export", output_dir, if_train=False)
    logger.info(args)
    logger.info("Intermediate layers: {}, include_parts={}".format(layers, args.include_parts))
    logger.info("Running with config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0, num_semantic_class=infer_num_semantic_classes(cfg))
    model.load_param(cfg.TEST.WEIGHT)
    model.cuda()

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    qf, gf = extract_features(model, val_loader, num_query, layers, args.include_parts, args.log_period)
    save_feature_files(cfg, qf, gf)

    q_g = np.dot(qf, gf.T)
    q_q = np.dot(qf, qf.T)
    g_g = np.dot(gf, gf.T)
    dist = re_ranking(q_g, q_q, g_g)
    indices = build_class_postprocess_indices(
        dist,
        mode=cfg.TEST.CLASS_POSTPROCESS,
        output_topk=SUBMISSION_TOPK,
        class_topk=cfg.TEST.CLASS_TOPK,
        mismatch_penalty=cfg.TEST.CLASS_MISMATCH_PENALTY,
        score_mode=cfg.TEST.CLASS_SCORE_MODE,
        scale=cfg.TEST.CLASS_SCALE,
    )
    output_path = write_outputs(args.track, indices)
    logger.info("Wrote submission to {}".format(output_path))
    print("Wrote submission to {}".format(output_path), flush=True)


if __name__ == "__main__":
    main()
