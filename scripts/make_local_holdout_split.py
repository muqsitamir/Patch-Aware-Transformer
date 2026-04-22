#!/usr/bin/env python3
import argparse
import csv
import math
import os
import random
from collections import Counter, defaultdict


TRAIN_CSV = "train.csv"
TRAIN_CLASSES_CSV = "train_classes.csv"
IMAGE_TRAIN_DIR = "image_train"

LOCAL_TRAIN_CSV = "local_train.csv"
LOCAL_TRAIN_CLASSES_CSV = "local_train_classes.csv"
LOCAL_VAL_QUERY_CSV = "local_val_query.csv"
LOCAL_VAL_QUERY_CLASSES_CSV = "local_val_query_classes.csv"
LOCAL_VAL_GALLERY_CSV = "local_val_gallery.csv"
LOCAL_VAL_GALLERY_CLASSES_CSV = "local_val_gallery_classes.csv"


def _columns(fieldnames):
    return {name.strip(): name for name in fieldnames or []}


def _require_column(fieldnames, candidates, csv_path):
    columns = _columns(fieldnames)
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    raise ValueError(
        "{} is missing required column. Expected one of: {}".format(
            csv_path, ", ".join(candidates)
        )
    )


def _read_csv(path):
    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("{} has no header".format(path))
        rows = [dict(row) for row in reader]
        return reader.fieldnames, rows


def _parse_pid(value, csv_path, image_name):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(
            "Invalid identity '{}' for image '{}' in {}".format(
                value, image_name, csv_path
            )
        )


def _row_value(row, column):
    return str(row.get(column, "")).strip()


def _load_source_rows(root):
    train_path = os.path.join(root, TRAIN_CSV)
    classes_path = os.path.join(root, TRAIN_CLASSES_CSV)
    image_dir = os.path.join(root, IMAGE_TRAIN_DIR)

    for path in (train_path, classes_path, image_dir):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    train_header, train_rows = _read_csv(train_path)
    classes_header, classes_rows = _read_csv(classes_path)

    train_camera_col = _require_column(train_header, ("cameraID", "camera_id"), train_path)
    train_image_col = _require_column(train_header, ("imageName", "image_name"), train_path)
    train_pid_col = _require_column(
        train_header, ("Corresponding Indexes", "objectID", "object"), train_path
    )

    class_camera_col = _require_column(classes_header, ("cameraID", "camera_id"), classes_path)
    class_image_col = _require_column(classes_header, ("imageName", "image_name"), classes_path)
    class_pid_col = _require_column(
        classes_header, ("Corresponding Indexes", "objectID", "object"), classes_path
    )
    class_col = _require_column(classes_header, ("Class", "class"), classes_path)

    class_by_key = {}
    class_by_image = {}
    duplicate_images = set()
    for row in classes_rows:
        image_name = _row_value(row, class_image_col)
        pid = _parse_pid(_row_value(row, class_pid_col), classes_path, image_name)
        key = (_row_value(row, class_camera_col), image_name, pid)
        class_by_key[key] = row
        if image_name in class_by_image:
            duplicate_images.add(image_name)
        else:
            class_by_image[image_name] = row

    items = []
    missing_class_rows = []
    missing_images = []
    for index, row in enumerate(train_rows):
        camera_id = _row_value(row, train_camera_col)
        image_name = _row_value(row, train_image_col)
        pid = _parse_pid(_row_value(row, train_pid_col), train_path, image_name)
        class_row = class_by_key.get((camera_id, image_name, pid))
        if class_row is None and image_name not in duplicate_images:
            class_row = class_by_image.get(image_name)
        if class_row is None:
            missing_class_rows.append(image_name)
            continue
        if not os.path.exists(os.path.join(image_dir, image_name)):
            missing_images.append(image_name)

        items.append(
            {
                "index": index,
                "pid": pid,
                "camera": camera_id,
                "image": image_name,
                "class": _row_value(class_row, class_col),
                "train_row": row,
                "class_row": class_row,
            }
        )

    if missing_class_rows:
        raise ValueError(
            "{} train images have no matching train_classes row; first missing: {}".format(
                len(missing_class_rows), ", ".join(missing_class_rows[:10])
            )
        )
    if missing_images:
        raise FileNotFoundError(
            "{} train images are missing from {}; first missing: {}".format(
                len(missing_images), image_dir, ", ".join(missing_images[:10])
            )
        )

    return train_header, classes_header, items


def _majority_class(items):
    counts = Counter(item["class"] for item in items)
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]


def _has_cross_camera_positive(items):
    return len({item["camera"] for item in items}) >= 2


def _choose_heldout_ids(pid_to_items, val_id_ratio, seed):
    rng = random.Random(seed)

    pid_to_class = {
        pid: _majority_class(items)
        for pid, items in pid_to_items.items()
    }

    eligible_by_class = defaultdict(list)
    all_ids_by_class = defaultdict(set)
    for pid, class_name in pid_to_class.items():
        all_ids_by_class[class_name].add(pid)
        if len(pid_to_items[pid]) >= 2 and _has_cross_camera_positive(pid_to_items[pid]):
            eligible_by_class[class_name].append(pid)

    max_by_class = {}
    for class_name, eligible_ids in eligible_by_class.items():
        num_all = len(all_ids_by_class[class_name])
        num_eligible = len(eligible_ids)
        if num_all == num_eligible:
            max_by_class[class_name] = max(0, num_eligible - 1)
        else:
            max_by_class[class_name] = num_eligible

    total_eligible = sum(len(ids) for ids in eligible_by_class.values())
    max_total = sum(max_by_class.values())
    if total_eligible == 0 or max_total == 0:
        raise ValueError(
            "No eligible validation identities. Need at least one class with "
            "two identities and at least two images per held-out identity."
        )

    target = int(round(total_eligible * val_id_ratio))
    target = max(1, min(target, max_total))

    quotas = {}
    fractional = []
    for class_name in sorted(eligible_by_class):
        desired = len(eligible_by_class[class_name]) * val_id_ratio
        quota = min(int(math.floor(desired)), max_by_class[class_name])
        quotas[class_name] = quota
        capacity = max_by_class[class_name] - quota
        if capacity > 0:
            fractional.append((desired - math.floor(desired), rng.random(), class_name))

    remaining = target - sum(quotas.values())
    fractional.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, _, class_name in fractional:
        if remaining <= 0:
            break
        quotas[class_name] += 1
        remaining -= 1

    heldout_ids = set()
    for class_name, quota in quotas.items():
        ids = sorted(eligible_by_class[class_name])
        rng.shuffle(ids)
        heldout_ids.update(ids[:quota])

    return heldout_ids, pid_to_class, total_eligible


def _split_query_gallery(pid_to_items, heldout_ids, seed):
    rng = random.Random(seed + 17)
    query_indices = set()
    gallery_indices = set()

    for pid in sorted(heldout_ids):
        items = sorted(pid_to_items[pid], key=lambda item: item["index"])
        shuffled = list(items)
        rng.shuffle(shuffled)
        num_images = len(shuffled)
        if num_images < 2:
            raise AssertionError("Held-out identity {} has fewer than 2 images".format(pid))
        num_query = max(1, min(num_images - 1, int(num_images * 0.25 + 0.5)))
        query_for_pid = set()
        by_index = {item["index"]: item for item in items}
        for candidate in shuffled:
            if len(query_for_pid) >= num_query:
                break
            candidate_query = query_for_pid | {candidate["index"]}
            candidate_gallery = [
                item for item in items
                if item["index"] not in candidate_query
            ]
            if not candidate_gallery:
                continue
            if all(
                any(gallery_item["camera"] != by_index[query_index]["camera"]
                    for gallery_item in candidate_gallery)
                for query_index in candidate_query
            ):
                query_for_pid.add(candidate["index"])

        if not query_for_pid:
            raise AssertionError(
                "Held-out identity {} has no cross-camera query/gallery split".format(pid)
            )

        for item in items:
            if item["index"] in query_for_pid:
                query_indices.add(item["index"])
            else:
                gallery_indices.add(item["index"])

    return query_indices, gallery_indices


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _identity_counts(items):
    return len({item["pid"] for item in items})


def _class_counts(items, pid_to_class=None):
    ids_by_class = defaultdict(set)
    images_by_class = Counter()
    for item in items:
        class_name = pid_to_class[item["pid"]] if pid_to_class else item["class"]
        ids_by_class[class_name].add(item["pid"])
        images_by_class[class_name] += 1
    return {
        class_name: (len(ids_by_class[class_name]), images_by_class[class_name])
        for class_name in sorted(ids_by_class)
    }


def _print_class_table(train_items, val_items, query_items, gallery_items, pid_to_class):
    train_counts = _class_counts(train_items, pid_to_class)
    val_counts = _class_counts(val_items, pid_to_class)
    query_counts = _class_counts(query_items, pid_to_class)
    gallery_counts = _class_counts(gallery_items, pid_to_class)
    class_names = sorted(
        set(train_counts) | set(val_counts) | set(query_counts) | set(gallery_counts)
    )
    if not class_names:
        return

    print("Per-class counts (ids/images):")
    print("  class, local_train, heldout, query, gallery")
    for class_name in class_names:
        train_pair = train_counts.get(class_name, (0, 0))
        val_pair = val_counts.get(class_name, (0, 0))
        query_pair = query_counts.get(class_name, (0, 0))
        gallery_pair = gallery_counts.get(class_name, (0, 0))
        print(
            "  {}, {}/{}, {}/{}, {}/{}, {}/{}".format(
                class_name,
                train_pair[0],
                train_pair[1],
                val_pair[0],
                val_pair[1],
                query_pair[0],
                query_pair[1],
                gallery_pair[0],
                gallery_pair[1],
            )
        )


def _validate_split(train_items, query_items, gallery_items, heldout_ids):
    train_ids = {item["pid"] for item in train_items}
    query_ids = {item["pid"] for item in query_items}
    gallery_ids = {item["pid"] for item in gallery_items}

    if train_ids & heldout_ids:
        raise AssertionError("Held-out identities leaked into local_train")
    if query_ids != heldout_ids:
        raise AssertionError("local_val_query does not contain every held-out identity")
    if gallery_ids != heldout_ids:
        raise AssertionError("local_val_gallery does not contain every held-out identity")

    query_images = {item["image"] for item in query_items}
    gallery_images = {item["image"] for item in gallery_items}
    if query_images & gallery_images:
        raise AssertionError("Query and gallery images overlap")

    for pid in heldout_ids:
        pid_query_items = [item for item in query_items if item["pid"] == pid]
        pid_gallery_items = [item for item in gallery_items if item["pid"] == pid]
        if len(pid_query_items) < 1:
            raise AssertionError("Held-out identity {} has no query image".format(pid))
        if len(pid_gallery_items) < 1:
            raise AssertionError("Held-out identity {} has no gallery image".format(pid))
        for query_item in pid_query_items:
            if not any(
                gallery_item["camera"] != query_item["camera"]
                for gallery_item in pid_gallery_items
            ):
                raise AssertionError(
                    "Held-out identity {} has no cross-camera gallery match for {}".format(
                        pid, query_item["image"]
                    )
                )


def make_split(root, val_id_ratio, seed):
    train_header, classes_header, items = _load_source_rows(root)

    pid_to_items = defaultdict(list)
    for item in items:
        pid_to_items[item["pid"]].append(item)

    heldout_ids, pid_to_class, total_eligible = _choose_heldout_ids(
        pid_to_items, val_id_ratio, seed
    )
    query_indices, gallery_indices = _split_query_gallery(pid_to_items, heldout_ids, seed)

    train_items = [item for item in items if item["pid"] not in heldout_ids]
    query_items = [item for item in items if item["index"] in query_indices]
    gallery_items = [item for item in items if item["index"] in gallery_indices]
    val_items = query_items + gallery_items

    _validate_split(train_items, query_items, gallery_items, heldout_ids)

    outputs = (
        (LOCAL_TRAIN_CSV, train_header, [item["train_row"] for item in train_items]),
        (LOCAL_TRAIN_CLASSES_CSV, classes_header, [item["class_row"] for item in train_items]),
        (LOCAL_VAL_QUERY_CSV, train_header, [item["train_row"] for item in query_items]),
        (
            LOCAL_VAL_QUERY_CLASSES_CSV,
            classes_header,
            [item["class_row"] for item in query_items],
        ),
        (LOCAL_VAL_GALLERY_CSV, train_header, [item["train_row"] for item in gallery_items]),
        (
            LOCAL_VAL_GALLERY_CLASSES_CSV,
            classes_header,
            [item["class_row"] for item in gallery_items],
        ),
    )

    for filename, header, rows in outputs:
        _write_csv(os.path.join(root, filename), header, rows)

    print("Urban Elements ReID local holdout split")
    print("  root: {}".format(root))
    print("  seed: {}".format(seed))
    print("  val_id_ratio: {}".format(val_id_ratio))
    print("  source train identities/images: {}/{}".format(len(pid_to_items), len(items)))
    print("  eligible validation identities: {}".format(total_eligible))
    print("  local train identities/images: {}/{}".format(_identity_counts(train_items), len(train_items)))
    print("  held-out identities/images: {}/{}".format(len(heldout_ids), len(val_items)))
    print("  query identities/images: {}/{}".format(_identity_counts(query_items), len(query_items)))
    print("  gallery identities/images: {}/{}".format(_identity_counts(gallery_items), len(gallery_items)))
    print("  wrote:")
    for filename, _, rows in outputs:
        print("    {} ({})".format(filename, len(rows)))
    _print_class_table(train_items, val_items, query_items, gallery_items, pid_to_class)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an identity-disjoint local holdout split for Urban Elements ReID."
    )
    parser.add_argument("--root", required=True, help="Urban2026 dataset root")
    parser.add_argument("--val-id-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not 0.0 < args.val_id_ratio < 1.0:
        raise ValueError("--val-id-ratio must be between 0 and 1")

    args.root = os.path.abspath(args.root)
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    make_split(parsed_args.root, parsed_args.val_id_ratio, parsed_args.seed)
