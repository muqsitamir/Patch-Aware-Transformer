#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_submission(path):
    rows = []
    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("{} has no header".format(path))
        columns = {name.strip(): name for name in reader.fieldnames}
        image_col = columns.get("imageName") or columns.get("image_name")
        pred_col = columns.get("Corresponding Indexes") or columns.get("predictions")
        if image_col is None or pred_col is None:
            raise ValueError("{} must contain imageName and Corresponding Indexes".format(path))
        for row in reader:
            image_name = str(row[image_col]).strip()
            indexes = [int(value) for value in str(row[pred_col]).split()]
            rows.append((image_name, indexes))
    return rows


def parse_weights(raw, count):
    if raw:
        weights = [float(value) for value in raw.split(",")]
        if len(weights) != count:
            raise ValueError("--weights must have one comma-separated value per input")
        return weights
    return [1.0] * count


def fuse_rows(inputs, weights, rrf_k, topk, preserve_head=0):
    base_names = [name for name, _ in inputs[0]]
    input_maps = []
    for rows in inputs:
        row_map = {name: indexes for name, indexes in rows}
        if [name for name, _ in rows] != base_names:
            missing = sorted(set(base_names) ^ set(row_map))
            if missing:
                raise ValueError("Submission query names do not match; first mismatch: {}".format(missing[:5]))
        input_maps.append(row_map)

    fused = []
    for image_name in base_names:
        scores = defaultdict(float)
        first_rank = {}
        for input_idx, row_map in enumerate(input_maps):
            weight = weights[input_idx]
            for rank, gallery_idx in enumerate(row_map[image_name], start=1):
                scores[gallery_idx] += weight / (rrf_k + rank)
                first_rank[gallery_idx] = min(first_rank.get(gallery_idx, rank), rank)
        ranked = sorted(scores, key=lambda idx: (-scores[idx], first_rank[idx], idx))
        if preserve_head > 0:
            head = input_maps[0][image_name][:preserve_head]
            head_set = set(head)
            ranked = head + [idx for idx in ranked if idx not in head_set]
        fused.append((image_name, ranked[:topk]))
    return fused


def write_submission(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["imageName", "Corresponding Indexes"])
        for image_name, indexes in rows:
            writer.writerow([image_name, " ".join(str(index) for index in indexes)])


def main():
    parser = argparse.ArgumentParser(description="Fuse Kaggle ReID submission rankings with RRF.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="", help="Comma-separated weights matching --inputs.")
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--preserve-head",
        type=int,
        default=0,
        help="Keep this many top predictions from the first input unchanged.",
    )
    args = parser.parse_args()

    inputs = [read_submission(path) for path in args.inputs]
    weights = parse_weights(args.weights, len(inputs))
    rows = fuse_rows(inputs, weights, args.rrf_k, args.topk, args.preserve_head)
    write_submission(rows, args.output)
    print("wrote {}".format(args.output))


if __name__ == "__main__":
    main()
