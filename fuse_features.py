import argparse
import csv
import os

import numpy as np

from utils.inference_postprocess import normalize_features
from utils.re_ranking import re_ranking


def load_features(path):
    features = np.load(path)
    if features.ndim != 2:
        raise ValueError("{} must contain a 2D feature array, got shape {}".format(path, features.shape))
    return normalize_features(features)


def weighted_feature_average(feat1, feat2, weight1, weight2):
    if feat1.shape != feat2.shape:
        raise ValueError("Feature shapes must match for feature fusion: {} vs {}".format(feat1.shape, feat2.shape))
    return normalize_features(float(weight1) * feat1 + float(weight2) * feat2)


def cosine_distance(qf, gf):
    return 1.0 - np.dot(qf, gf.T)


def rank_from_features(qf, gf, topk, use_rerank=False):
    if use_rerank:
        q_g_dist = np.dot(qf, gf.T)
        q_q_dist = np.dot(qf, qf.T)
        g_g_dist = np.dot(gf, gf.T)
        distmat = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    else:
        distmat = cosine_distance(qf, gf)
    return np.argsort(distmat, axis=1)[:, :topk]


def write_track(indices, path):
    with open(path, "wb") as f_w:
        for row in indices:
            write_line = row + 1
            write_line = " ".join(map(str, write_line.tolist())) + "\n"
            f_w.write(write_line.encode())


def write_submission(indices, path):
    image_names = ["{:06d}.jpg".format(i) for i in range(1, len(indices) + 1)]
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["imageName", "Corresponding Indexes"])
        for image_name, row in zip(image_names, indices):
            writer.writerow([image_name, " ".join(map(str, row + 1))])


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse two ReID feature runs and write a submission ranking.")
    parser.add_argument("--feat-q", required=True, help="Run 1 query features .npy")
    parser.add_argument("--feat-g", required=True, help="Run 1 gallery features .npy")
    parser.add_argument("--feat-q2", required=True, help="Run 2 query features .npy")
    parser.add_argument("--feat-g2", required=True, help="Run 2 gallery features .npy")
    parser.add_argument("--weight1", type=float, default=0.5)
    parser.add_argument("--weight2", type=float, default=0.5)
    parser.add_argument("--mode", choices=("feature", "distance"), default="feature")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--rerank", action="store_true", help="Apply k-reciprocal re-ranking after feature fusion.")
    parser.add_argument("--output-prefix", default="fused")
    parser.add_argument("--track-txt", default=None)
    parser.add_argument("--submission-csv", default=None)
    parser.add_argument("--no-save-features", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    qf1 = load_features(args.feat_q)
    gf1 = load_features(args.feat_g)
    qf2 = load_features(args.feat_q2)
    gf2 = load_features(args.feat_g2)

    if qf1.shape[0] != qf2.shape[0] or gf1.shape[0] != gf2.shape[0]:
        raise ValueError("Both runs must have the same query/gallery counts.")

    track_txt = args.track_txt or args.output_prefix + ".txt"
    submission_csv = args.submission_csv or args.output_prefix + "_submission.csv"
    ensure_parent(track_txt)
    ensure_parent(submission_csv)

    if args.mode == "feature":
        qf = weighted_feature_average(qf1, qf2, args.weight1, args.weight2)
        gf = weighted_feature_average(gf1, gf2, args.weight1, args.weight2)
        if not args.no_save_features:
            q_path = args.output_prefix + "_qf.npy"
            g_path = args.output_prefix + "_gf.npy"
            ensure_parent(q_path)
            ensure_parent(g_path)
            np.save(q_path, qf)
            np.save(g_path, gf)
        indices = rank_from_features(qf, gf, args.topk, use_rerank=args.rerank)
    else:
        if args.rerank:
            raise ValueError("--rerank is only supported with --mode feature.")
        distmat = float(args.weight1) * cosine_distance(qf1, gf1) + float(args.weight2) * cosine_distance(qf2, gf2)
        indices = np.argsort(distmat, axis=1)[:, :args.topk]

    write_track(indices, track_txt)
    write_submission(indices, submission_csv)
    print("Wrote {}".format(track_txt))
    print("Wrote {}".format(submission_csv))


if __name__ == "__main__":
    main()
