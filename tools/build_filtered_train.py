#!/usr/bin/env python3
"""
Build filtered training CSVs and annotate gallery resolution for inference.

Two operations in one script:

  1. TRAINING FILTER. Removes:
     - Images whose shorter side or area falls below per-class thresholds.
     - Identities with fewer than --min-instances remaining crops.
     Writes train_filtered.csv and train_classes_filtered.csv.

  2. GALLERY RESOLUTION ANNOTATION. Computes per-image resolution metrics
     for the gallery (image_test) and query (image_query) sets and writes
     them to gallery_resolution.csv and query_resolution.csv. These are
     consumed by the inference pipeline to down-weight tiny gallery items
     in the final ranking. Gallery rows are NOT removed (the submission
     format requires fixed gallery indexes), only annotated.

Per-class threshold rationale (based on Urban2026 EDA):

  trafficsignal: median min_side 27px, 51% of queries below 32px.
                 Use a low threshold (12px) so the model still sees small
                 sign crops during training.

  RubbishBins:   median min_side 36px, 45% of queries below 32px.
                 Similar reasoning - keep most of the long tail.

  Container:     median min_side 62px, only 8% of gallery below 32px.
                 Can afford a slightly higher threshold.

  Crosswalk:     ratio is 6.9 (panoramic), so min_side is misleading.
                 Filter on AREA instead.

Defaults below correspond to "drop pure-noise crops only" - keeping the
model exposed to the small-crop appearance present at test time.

Usage
-----
  python tools/build_filtered_train.py --root /path/to/dataset

  # custom thresholds:
  python tools/build_filtered_train.py --root /path/to/dataset \\
      --min-side-default 16 --min-side-trafficsignal 12 \\
      --min-instances 3
"""

import argparse
import csv
import os.path as osp
from collections import defaultdict

from PIL import Image


_SIZE_CACHE = "_image_sizes.csv"
_TRAIN_CSV = "train.csv"
_TRAIN_CLASSES_CSV = "train_classes.csv"
_QUERY_CSV = "query.csv"
_QUERY_CLASSES_CSV = "query_classes.csv"
_TEST_CSV = "test.csv"
_TEST_CLASSES_CSV = "test_classes.csv"
_FILTERED_CSV = "train_filtered.csv"
_FILTERED_CLASSES_CSV = "train_classes_filtered.csv"
_REPORT = "filter_report.txt"
_GALLERY_RES_CSV = "gallery_resolution.csv"
_QUERY_RES_CSV = "query_resolution.csv"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_size_cache(path):
    cache = {}
    if not osp.exists(path):
        return cache
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                cache[row[0]] = (int(row[1]), int(row[2]))
    return cache


def _save_size_cache(path, cache):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['imageName', 'width', 'height'])
        for name, (w, h) in sorted(cache.items()):
            writer.writerow([name, w, h])


def _read_csv(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def _write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _resolve_size(image_dir, image_name, cache):
    """Return (w, h) for image_name, using cache. None if file unreadable."""
    if image_name in cache:
        return cache[image_name]
    img_path = osp.join(image_dir, image_name)
    try:
        with Image.open(img_path) as img:
            w, h = img.size
    except Exception as exc:
        print(f"WARNING: cannot open {img_path}: {exc}")
        return None
    cache[image_name] = (w, h)
    return (w, h)


def _passes_threshold(w, h, cls, args):
    """
    Decide whether (w, h, cls) passes the per-class resolution threshold.
    Crosswalks are filtered by AREA (because they are panoramic).
    All others are filtered by min_side.
    """
    if cls.lower() == 'crosswalk':
        return (w * h) >= args.min_area_crosswalk
    threshold_map = {
        'container': args.min_side_container,
        'rubbishbins': args.min_side_rubbishbins,
        'trafficsignal': args.min_side_trafficsignal,
    }
    threshold = threshold_map.get(cls.lower(), args.min_side_default)
    return min(w, h) >= threshold


def _annotate_resolution(image_dir, csv_rows, classes_lookup, cache, name_col_idx):
    """Compute width/height/min_side/area for each row. Returns list of dicts."""
    annotations = []
    for row in csv_rows:
        image_name = row[name_col_idx]
        size = _resolve_size(image_dir, image_name, cache)
        if size is None:
            w, h = 0, 0
        else:
            w, h = size
        cls = classes_lookup.get(image_name, '').lower()
        annotations.append({
            'imageName': image_name,
            'width': w,
            'height': h,
            'min_side': min(w, h),
            'area': w * h,
            'class': cls,
        })
    return annotations


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True, help="Dataset root directory")

    # per-class min_side thresholds (and default)
    parser.add_argument('--min-side-default', type=int, default=16,
                        help="Default min(w,h) threshold for unclassified rows")
    parser.add_argument('--min-side-trafficsignal', type=int, default=12,
                        help="min(w,h) for traffic signs (default: 12)")
    parser.add_argument('--min-side-rubbishbins', type=int, default=14,
                        help="min(w,h) for rubbish bins (default: 14)")
    parser.add_argument('--min-side-container', type=int, default=20,
                        help="min(w,h) for containers (default: 20)")
    parser.add_argument('--min-area-crosswalk', type=int, default=2000,
                        help="Minimum area (w*h) for crosswalks (default: 2000)")

    # singleton filter
    parser.add_argument('--min-instances', type=int, default=3,
                        help="Drop identities with fewer than this many crops")

    # for low-res annotation (inference-side use)
    parser.add_argument('--lowres-default', type=int, default=24,
                        help="Default low-res threshold for unclassified rows")

    args = parser.parse_args()
    root = args.root
    image_dir_train = osp.join(root, 'image_train')
    image_dir_query = osp.join(root, 'image_query')
    image_dir_test = osp.join(root, 'image_test')
    cache_path = osp.join(root, _SIZE_CACHE)

    # -----------------------------------------------------------------------
    # load CSVs
    # -----------------------------------------------------------------------
    train_header, train_rows = _read_csv(osp.join(root, _TRAIN_CSV))
    classes_header, classes_rows = _read_csv(osp.join(root, _TRAIN_CLASSES_CSV))
    name_to_class = {row[1]: row[3] for row in classes_rows if len(row) >= 4}

    total_original = len(train_rows)
    print(f"Read {total_original} rows from {_TRAIN_CSV}")
    print(f"Read {len(classes_rows)} rows from {_TRAIN_CLASSES_CSV}")

    size_cache = _load_size_cache(cache_path)
    cache_size_before = len(size_cache)

    # =======================================================================
    # PART 1: TRAINING FILTER (per-class thresholds + singleton drop)
    # =======================================================================

    size_kept = []
    size_dropped_by_class = defaultdict(int)
    size_dropped_count = 0

    for row in train_rows:
        image_name = row[1]
        size = _resolve_size(image_dir_train, image_name, size_cache)
        if size is None:
            size_dropped_count += 1
            continue
        w, h = size
        cls = name_to_class.get(image_name, '')

        if _passes_threshold(w, h, cls, args):
            size_kept.append(row)
        else:
            size_dropped_count += 1
            size_dropped_by_class[cls] += 1

    after_size = len(size_kept)

    # singleton filter
    pid_counts = defaultdict(int)
    for row in size_kept:
        pid_counts[int(row[2])] += 1

    kept_pids = {pid for pid, cnt in pid_counts.items() if cnt >= args.min_instances}
    dropped_pids = sorted(pid for pid, cnt in pid_counts.items() if cnt < args.min_instances)

    final_rows = [row for row in size_kept if int(row[2]) in kept_pids]
    instance_dropped_count = after_size - len(final_rows)
    after_instances = len(final_rows)

    kept_names = {row[1] for row in final_rows}
    final_classes_rows = [row for row in classes_rows if row[1] in kept_names]

    _write_csv(osp.join(root, _FILTERED_CSV), train_header, final_rows)
    _write_csv(osp.join(root, _FILTERED_CLASSES_CSV), classes_header, final_classes_rows)

    final_per_class = defaultdict(int)
    for row in final_classes_rows:
        if len(row) >= 4:
            final_per_class[row[3]] += 1

    # =======================================================================
    # PART 2: GALLERY + QUERY RESOLUTION ANNOTATION
    # =======================================================================

    # query
    query_header, query_rows = _read_csv(osp.join(root, _QUERY_CSV))
    _, query_classes_rows = _read_csv(osp.join(root, _QUERY_CLASSES_CSV))
    query_class_lookup = {row[1]: row[2] for row in query_classes_rows if len(row) >= 3}
    query_annotations = _annotate_resolution(
        image_dir_query, query_rows, query_class_lookup, size_cache, name_col_idx=1)

    # gallery
    test_header, test_rows = _read_csv(osp.join(root, _TEST_CSV))
    _, test_classes_rows = _read_csv(osp.join(root, _TEST_CLASSES_CSV))
    test_class_lookup = {row[1]: row[2] for row in test_classes_rows if len(row) >= 3}
    gallery_annotations = _annotate_resolution(
        image_dir_test, test_rows, test_class_lookup, size_cache, name_col_idx=1)

    def _tag_lowres(annotations):
        for a in annotations:
            cls = a['class']
            if cls == 'crosswalk':
                a['lowres'] = a['area'] < args.min_area_crosswalk
            else:
                threshold_map = {
                    'container': args.min_side_container,
                    'rubbishbins': args.min_side_rubbishbins,
                    'trafficsignal': args.min_side_trafficsignal,
                }
                t = threshold_map.get(cls, args.lowres_default)
                a['lowres'] = a['min_side'] < t
        return annotations

    query_annotations = _tag_lowres(query_annotations)
    gallery_annotations = _tag_lowres(gallery_annotations)

    def _write_annot(path, rows):
        header = ['imageName', 'width', 'height', 'min_side', 'area', 'class', 'lowres']
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in rows:
                writer.writerow([
                    r['imageName'], r['width'], r['height'],
                    r['min_side'], r['area'], r['class'],
                    int(r['lowres']),
                ])

    _write_annot(osp.join(root, _GALLERY_RES_CSV), gallery_annotations)
    _write_annot(osp.join(root, _QUERY_RES_CSV), query_annotations)

    if len(size_cache) > cache_size_before:
        _save_size_cache(cache_path, size_cache)
        print(f"Size cache updated: {cache_size_before} -> {len(size_cache)} entries")

    # =======================================================================
    # report
    # =======================================================================
    lines = [
        "=" * 60,
        "Training Data Filter Report",
        "=" * 60,
        f"--root {root!r}",
        "",
        "Training filter thresholds:",
        f"  default min_side       : {args.min_side_default}",
        f"  trafficsignal min_side : {args.min_side_trafficsignal}",
        f"  rubbishbins min_side   : {args.min_side_rubbishbins}",
        f"  container min_side     : {args.min_side_container}",
        f"  crosswalk min_area     : {args.min_area_crosswalk}",
        f"  min instances per pid  : {args.min_instances}",
        "",
        f"Original images               : {total_original}",
        f"Dropped by resolution         : {size_dropped_count}",
    ]
    for cls, n in sorted(size_dropped_by_class.items()):
        lines.append(f"  - {cls or '(none)'}: {n}")
    lines += [
        f"After resolution filter       : {after_size}",
        f"Identities before filter      : {len(pid_counts)}",
        f"Dropped identities            : {len(dropped_pids)}",
        f"Images dropped (low-count id) : {instance_dropped_count}",
        f"Final images                  : {after_instances}",
        f"Final identities              : {len(kept_pids)}",
        "",
        "Final per-class image counts:",
    ]
    for cls in sorted(final_per_class):
        lines.append(f"  {cls:14s}: {final_per_class[cls]}")
    lines += [
        "",
        "=" * 60,
        "Gallery / Query Resolution Annotation",
        "=" * 60,
    ]
    n_query_lowres = sum(1 for a in query_annotations if a['lowres'])
    n_gallery_lowres = sum(1 for a in gallery_annotations if a['lowres'])
    lines += [
        f"Query images annotated     : {len(query_annotations)}",
        f"  flagged low-resolution   : {n_query_lowres} "
        f"({n_query_lowres/max(1,len(query_annotations)):.0%})",
        f"Gallery images annotated   : {len(gallery_annotations)}",
        f"  flagged low-resolution   : {n_gallery_lowres} "
        f"({n_gallery_lowres/max(1,len(gallery_annotations)):.0%})",
        "",
        "Per-class low-res counts (gallery):",
    ]
    cls_total = defaultdict(int)
    cls_lowres = defaultdict(int)
    for a in gallery_annotations:
        cls_total[a['class']] += 1
        if a['lowres']:
            cls_lowres[a['class']] += 1
    for cls in sorted(cls_total):
        t = cls_total[cls]
        l = cls_lowres[cls]
        lines.append(f"  {cls:14s}: {l}/{t} ({l/max(1,t):.0%})")

    report = "\n".join(lines)
    print()
    print(report)

    with open(osp.join(root, _REPORT), 'w') as f:
        f.write(report + "\n")

    print()
    print(f"Wrote -> {osp.join(root, _FILTERED_CSV)}")
    print(f"Wrote -> {osp.join(root, _FILTERED_CLASSES_CSV)}")
    print(f"Wrote -> {osp.join(root, _GALLERY_RES_CSV)}")
    print(f"Wrote -> {osp.join(root, _QUERY_RES_CSV)}")
    print(f"Wrote -> {osp.join(root, _REPORT)}")


if __name__ == '__main__':
    main()