#!/usr/bin/env python3
import argparse
import csv
import os

import numpy as np


SUBMISSION_TOPK = 100


def l2_normalize(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


def topk_indices(scores, k, largest=True):
    k = min(k, scores.shape[1])
    if largest:
        part = np.argpartition(-scores, kth=np.arange(k), axis=1)[:, :k]
        order = np.take_along_axis(scores, part, axis=1)
        return np.take_along_axis(part, np.argsort(-order, axis=1), axis=1)
    part = np.argpartition(scores, kth=np.arange(k), axis=1)[:, :k]
    order = np.take_along_axis(scores, part, axis=1)
    return np.take_along_axis(part, np.argsort(order, axis=1), axis=1)


def weighted_neighbor_average(features, sim, neigh, alpha, power):
    if alpha <= 0 or neigh.shape[1] == 0:
        return features
    rows = np.arange(features.shape[0])[:, None]
    weights = np.maximum(sim[rows, neigh], 0.0) ** power
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    neighbor_feat = np.einsum("nk,nkd->nd", weights, features[neigh])
    return l2_normalize((1.0 - alpha) * features + alpha * neighbor_feat)


def apply_dba(qf, gf, dba_k, dba_alpha, dba_power, qe_k, qe_alpha):
    gf = l2_normalize(gf)
    qf = l2_normalize(qf)

    gsim = gf @ gf.T
    g_neigh = topk_indices(gsim, dba_k, largest=True)
    gf_aug = weighted_neighbor_average(gf, gsim, g_neigh, dba_alpha, dba_power)

    if qe_k > 0 and qe_alpha > 0:
        qg = qf @ gf_aug.T
        q_neigh = topk_indices(qg, qe_k, largest=True)
        rows = np.arange(qf.shape[0])[:, None]
        weights = np.maximum(qg[rows, q_neigh], 0.0) ** dba_power
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        q_neighbor_feat = np.einsum("nk,nkd->nd", weights, gf_aug[q_neigh])
        qf = l2_normalize((1.0 - qe_alpha) * qf + qe_alpha * q_neighbor_feat)

    return qf, gf_aug


def apply_gallery_propagation(scores, gf, prop_k, prop_alpha, prop_mode):
    if prop_k <= 1 or prop_alpha <= 0:
        return scores
    gsim = gf @ gf.T
    neigh = topk_indices(gsim, prop_k, largest=True)
    neigh_scores = scores[:, neigh]
    if prop_mode == "mean":
        propagated = neigh_scores.mean(axis=2)
    elif prop_mode == "weighted":
        rows = np.arange(gf.shape[0])[:, None]
        weights = np.maximum(gsim[rows, neigh], 0.0)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        propagated = np.einsum("gk,qgk->qg", weights, neigh_scores)
    else:
        propagated = neigh_scores.max(axis=2)
    return (1.0 - prop_alpha) * scores + prop_alpha * propagated


def read_class_csv(path):
    if not path:
        return None
    classes = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes.append(row.get("Class", "").strip().lower())
    return np.asarray(classes)


def apply_class_boost(scores, q_classes, g_classes, boost):
    if boost == 0 or q_classes is None or g_classes is None:
        return scores
    same = q_classes[:, None] == g_classes[None, :]
    return scores + boost * same.astype(np.float32)


def read_query_names(path, n):
    if not path:
        return ["{:06d}.jpg".format(i) for i in range(1, n + 1)]
    names = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row.get("imageName", "{:06d}.jpg".format(len(names) + 1)))
    if len(names) != n:
        raise ValueError("Expected {} query names, got {}".format(n, len(names)))
    return names


def write_submission(track_path, indices, query_names):
    parent = os.path.dirname(track_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(track_path, "w") as f:
        for row in indices:
            f.write(" ".join(map(str, (row + 1).tolist())) + "\n")

    output_path = track_path.split(".txt")[0] + "_submission.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imageName", "Corresponding Indexes"])
        for name, row in zip(query_names, indices):
            writer.writerow([name, " ".join(map(str, (row + 1).tolist()))])
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Postprocess saved ReID features into a submission.")
    parser.add_argument("--qf", required=True)
    parser.add_argument("--gf", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--query-csv", default="")
    parser.add_argument("--query-class-csv", default="")
    parser.add_argument("--gallery-class-csv", default="")
    parser.add_argument("--dba-k", type=int, default=1)
    parser.add_argument("--dba-alpha", type=float, default=0.0)
    parser.add_argument("--dba-power", type=float, default=2.0)
    parser.add_argument("--qe-k", type=int, default=0)
    parser.add_argument("--qe-alpha", type=float, default=0.0)
    parser.add_argument("--prop-k", type=int, default=1)
    parser.add_argument("--prop-alpha", type=float, default=0.0)
    parser.add_argument("--prop-mode", choices=("max", "mean", "weighted"), default="max")
    parser.add_argument("--class-boost", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    qf = l2_normalize(np.load(args.qf).astype(np.float32))
    gf = l2_normalize(np.load(args.gf).astype(np.float32))

    qf, gf = apply_dba(qf, gf, args.dba_k, args.dba_alpha, args.dba_power, args.qe_k, args.qe_alpha)
    scores = qf @ gf.T
    scores = apply_gallery_propagation(scores, gf, args.prop_k, args.prop_alpha, args.prop_mode)

    q_classes = read_class_csv(args.query_class_csv)
    g_classes = read_class_csv(args.gallery_class_csv)
    scores = apply_class_boost(scores, q_classes, g_classes, args.class_boost)

    indices = topk_indices(scores, SUBMISSION_TOPK, largest=True)
    query_names = read_query_names(args.query_csv, qf.shape[0])
    output_path = write_submission(args.track, indices, query_names)
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
