import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_split_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["cameraID"], row["imageName"]))
    return rows


def _read_dict_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_class_csv(path):
    if not path.exists():
        return {}
    return {row["imageName"]: row["Class"] for row in _read_dict_csv(path)}


def _norm_class(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _normalize(x):
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def _load_features(qf_paths, gf_paths):
    parts = []
    for qf_path, gf_path in zip(qf_paths, gf_paths):
        qf = np.load(qf_path)
        gf = np.load(gf_path)
        parts.append(_normalize(np.vstack([qf, gf])))
    return _normalize(np.concatenate(parts, axis=1))


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _mutual_knn_clusters(features, topk, sim_threshold, min_size, max_size):
    if features.shape[0] < min_size:
        return [], 0

    sim = features @ features.T
    np.fill_diagonal(sim, -np.inf)
    topk = min(topk, sim.shape[0] - 1)
    topk_idx = np.argpartition(-sim, kth=topk - 1, axis=1)[:, :topk]
    topk_sets = [set(row.tolist()) for row in topk_idx]

    uf = UnionFind(sim.shape[0])
    edge_count = 0
    for i in range(sim.shape[0]):
        for j in topk_idx[i]:
            if i < j and i in topk_sets[j] and sim[i, j] >= sim_threshold:
                uf.union(i, j)
                edge_count += 1

    raw = defaultdict(list)
    for i in range(sim.shape[0]):
        raw[uf.find(i)].append(i)

    clusters = [
        sorted(v)
        for v in raw.values()
        if len(v) >= min_size and (max_size <= 0 or len(v) <= max_size)
    ]
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters, edge_count


def _cluster_features(features, groups, topk, sim_threshold, min_size, max_size):
    if groups is None:
        return _mutual_knn_clusters(features, topk, sim_threshold, min_size, max_size)

    clusters = []
    edge_count = 0
    by_group = defaultdict(list)
    for idx, group in enumerate(groups):
        by_group[group].append(idx)

    for _, indices in sorted(by_group.items()):
        if len(indices) < min_size:
            continue
        sub_clusters, sub_edges = _mutual_knn_clusters(
            features[np.asarray(indices)],
            topk=min(topk, len(indices) - 1),
            sim_threshold=sim_threshold,
            min_size=min_size,
            max_size=max_size,
        )
        clusters.extend([[indices[i] for i in cluster] for cluster in sub_clusters])
        edge_count += sub_edges

    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters, edge_count


def _safe_symlink(src, dst):
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)


def _copy_eval_split(root, out, image_dir, csv_name, class_csv_name):
    src_dir = root / image_dir
    dst_dir = out / image_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / csv_name, out / csv_name)
    for _, image_name in _read_split_csv(root / csv_name):
        _safe_symlink(src_dir / image_name, dst_dir / image_name)
    class_csv = root / class_csv_name
    if class_csv.exists():
        shutil.copy2(class_csv, out / class_csv_name)


def build_pseudo_root(args):
    root = Path(args.challenge_root)
    out = Path(args.output_root)
    image_train = out / "image_train"
    image_train.mkdir(parents=True, exist_ok=True)

    query_rows = _read_split_csv(root / "query.csv")
    gallery_rows = _read_split_csv(root / "test.csv")
    query_classes = _read_class_csv(root / "query_classes.csv")
    gallery_classes = _read_class_csv(root / "test_classes.csv")
    features = _load_features(args.qf, args.gf)

    expected = len(query_rows) + len(gallery_rows)
    if features.shape[0] != expected:
        raise ValueError(
            "Feature row count {} does not match query+gallery count {}".format(
                features.shape[0], expected
            )
        )

    groups = None
    if args.class_aware:
        groups = []
        for _, image_name in query_rows:
            groups.append(_norm_class(query_classes.get(image_name, "unknown")))
        for _, image_name in gallery_rows:
            groups.append(_norm_class(gallery_classes.get(image_name, "unknown")))

    clusters, edge_count = _cluster_features(
        features,
        groups=groups,
        topk=args.topk,
        sim_threshold=args.sim_threshold,
        min_size=args.min_cluster_size,
        max_size=args.max_cluster_size,
    )

    items = []
    for idx, (cam, name) in enumerate(query_rows):
        if not args.include_query:
            continue
        items.append(
            {
                "index": idx,
                "cameraID": cam,
                "imageName": "q_" + name,
                "src": root / "image_query" / name,
                "split": "query",
                "class": query_classes.get(name, ""),
            }
        )
    offset = len(query_rows)
    for idx, (cam, name) in enumerate(gallery_rows, start=offset):
        items.append(
            {
                "index": idx,
                "cameraID": cam,
                "imageName": "g_" + name,
                "src": root / "image_test" / name,
                "split": "gallery",
                "class": gallery_classes.get(name, ""),
            }
        )

    index_to_item = {item["index"]: item for item in items}
    kept_clusters = []
    for cluster in clusters:
        cluster_items = [index_to_item[i] for i in cluster if i in index_to_item]
        if len(cluster_items) >= args.min_cluster_size:
            kept_clusters.append(cluster_items)

    next_pid = 0
    with open(out / "train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cameraID", "imageName", "Corresponding Indexes"])

        if args.include_real_train:
            train_rows = _read_dict_csv(root / "train.csv")
            for row in train_rows:
                pid = int(row["Corresponding Indexes"])
                next_pid = max(next_pid, pid + 1)
                image_name = "real_" + row["imageName"]
                _safe_symlink(root / "image_train" / row["imageName"], image_train / image_name)
                writer.writerow([row["cameraID"], image_name, pid])

        for local_pid, cluster_items in enumerate(kept_clusters):
            pid = next_pid + local_pid
            for item in cluster_items:
                _safe_symlink(item["src"], image_train / item["imageName"])
                writer.writerow([item["cameraID"], item["imageName"], pid])

    with open(out / "pseudo_clusters.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "split", "cameraID", "imageName", "class", "source_path"])
        for local_pid, cluster_items in enumerate(kept_clusters):
            pid = next_pid + local_pid
            for item in cluster_items:
                writer.writerow([pid, item["split"], item["cameraID"], item["imageName"], item["class"], item["src"]])

    if args.include_eval:
        _copy_eval_split(root, out, "image_query", "query.csv", "query_classes.csv")
        _copy_eval_split(root, out, "image_test", "test.csv", "test_classes.csv")

    sizes = [len(c) for c in kept_clusters]
    print("features:", features.shape)
    print("class_aware:", args.class_aware)
    print("include_real_train:", args.include_real_train)
    print("mutual_edges:", edge_count)
    print("clusters:", len(sizes))
    print("pseudo_images:", sum(sizes))
    if args.include_real_train:
        print("real_train_images:", len(_read_dict_csv(root / "train.csv")))
    if sizes:
        print("size_min/mean/max:", min(sizes), round(float(np.mean(sizes)), 2), max(sizes))
        hist = defaultdict(int)
        for size in sizes:
            hist[size] += 1
        print("size_hist:", dict(sorted(hist.items())))
    print("output_root:", out)


def main():
    parser = argparse.ArgumentParser(description="Build a pseudo-labeled UrbanElementsReID root from exported test features.")
    parser.add_argument("--challenge-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--qf", action="append", required=True, help="Query feature .npy path. Repeat with --gf for feature fusion.")
    parser.add_argument("--gf", action="append", required=True, help="Gallery feature .npy path. Repeat with --qf for feature fusion.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--sim-threshold", type=float, default=0.74)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--max-cluster-size", type=int, default=20)
    parser.add_argument("--include-query", action="store_true")
    parser.add_argument("--class-aware", action="store_true", help="Cluster query/gallery images only within their semantic class.")
    parser.add_argument("--include-real-train", action="store_true", help="Add the original challenge train set to train.csv before pseudo IDs.")
    parser.add_argument("--include-eval", action="store_true", help="Symlink query/test splits into the generated root for normal train.py evaluation setup.")
    args = parser.parse_args()
    if len(args.qf) != len(args.gf):
        raise ValueError("--qf and --gf must be supplied the same number of times")
    build_pseudo_root(args)


if __name__ == "__main__":
    main()
