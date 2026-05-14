#!/usr/bin/env python3
import argparse
import csv
import os


def read_class_csv(path):
    image_to_class = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_to_class[row["imageName"]] = row["Class"]
    return image_to_class


def read_submission(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["imageName"], [int(x) for x in row["Corresponding Indexes"].split()]))
    return rows


def write_submission(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imageName", "Corresponding Indexes"])
        for image_name, ranking in rows:
            writer.writerow([image_name, " ".join(str(x) for x in ranking)])


def route_rows(
    base_rows,
    specialist_rows,
    query_classes,
    gallery_classes,
    target_classes,
    topk,
    specialist_topk,
    preserve_base_topn,
    require_specialist_top1_in_base_topn,
    require_overlap_topn,
    require_overlap_min,
):
    specialist_by_name = {name: ranking for name, ranking in specialist_rows}
    target_classes = {c.lower() for c in target_classes}
    output_rows = []
    replaced = 0

    for image_name, base_ranking in base_rows:
        query_class = query_classes.get(image_name, "").lower()
        if query_class not in target_classes or image_name not in specialist_by_name:
            output_rows.append((image_name, base_ranking[:topk]))
            continue

        specialist_class_ranking = []
        specialist_seen = set()
        for idx in specialist_by_name[image_name]:
            gallery_name = "{:06d}.jpg".format(idx)
            if gallery_classes.get(gallery_name, "").lower() not in target_classes:
                continue
            if idx in specialist_seen:
                continue
            specialist_class_ranking.append(idx)
            specialist_seen.add(idx)

        should_route = True
        if require_specialist_top1_in_base_topn > 0:
            base_gate = set(base_ranking[:require_specialist_top1_in_base_topn])
            should_route = bool(specialist_class_ranking and specialist_class_ranking[0] in base_gate)

        if should_route and require_overlap_topn > 0 and require_overlap_min > 0:
            base_gate = set(base_ranking[:require_overlap_topn])
            specialist_gate = set(specialist_class_ranking[:require_overlap_topn])
            should_route = len(base_gate & specialist_gate) >= require_overlap_min

        if not should_route:
            output_rows.append((image_name, base_ranking[:topk]))
            continue

        chosen = []
        seen = set()
        for idx in base_ranking[:preserve_base_topn]:
            chosen.append(idx)
            seen.add(idx)

        inserted = 0
        for idx in specialist_class_ranking:
            if idx not in seen:
                chosen.append(idx)
                seen.add(idx)
                inserted += 1
            if inserted == specialist_topk:
                break

        for idx in base_ranking:
            if idx not in seen:
                chosen.append(idx)
                seen.add(idx)
            if len(chosen) == topk:
                break

        output_rows.append((image_name, chosen[:topk]))
        replaced += 1

    return output_rows, replaced


def parse_args():
    parser = argparse.ArgumentParser(description="Route selected query classes to a specialist submission.")
    parser.add_argument("--base", required=True, help="Baseline submission CSV.")
    parser.add_argument("--specialist", required=True, help="Specialist submission CSV.")
    parser.add_argument("--output", required=True, help="Output routed submission CSV.")
    parser.add_argument("--query-classes", required=True, help="query_classes.csv for the challenge split.")
    parser.add_argument("--gallery-classes", required=True, help="test_classes.csv for the challenge split.")
    parser.add_argument("--classes", nargs="+", required=True, help="Class names to route to the specialist.")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--specialist-topk",
        type=int,
        default=None,
        help="Number of same-class specialist results to insert before baseline backfill. Defaults to --topk.",
    )
    parser.add_argument(
        "--preserve-base-topn",
        type=int,
        default=0,
        help="Keep this many baseline results at the front before inserting specialist results.",
    )
    parser.add_argument(
        "--require-specialist-top1-in-base-topn",
        type=int,
        default=0,
        help="Route only if the specialist top-1 same-class result is already in the baseline top N. Disabled at 0.",
    )
    parser.add_argument(
        "--require-overlap-topn",
        type=int,
        default=0,
        help="Route only if baseline top N and specialist same-class top N overlap enough. Disabled at 0.",
    )
    parser.add_argument(
        "--require-overlap-min",
        type=int,
        default=0,
        help="Minimum overlap required when --require-overlap-topn is enabled.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_rows = read_submission(args.base)
    specialist_rows = read_submission(args.specialist)
    query_classes = read_class_csv(args.query_classes)
    gallery_classes = read_class_csv(args.gallery_classes)

    routed_rows, replaced = route_rows(
        base_rows,
        specialist_rows,
        query_classes,
        gallery_classes,
        args.classes,
        args.topk,
        args.specialist_topk if args.specialist_topk is not None else args.topk,
        args.preserve_base_topn,
        args.require_specialist_top1_in_base_topn,
        args.require_overlap_topn,
        args.require_overlap_min,
    )
    write_submission(args.output, routed_rows)
    print("Wrote {} rows to {}; replaced {} query rows.".format(len(routed_rows), args.output, replaced))


if __name__ == "__main__":
    main()
