import csv
import json
import logging
import os
import re

import numpy as np
import torch

from utils.class_aware import (
    _class_array,
    _valid_class_mask,
    build_class_to_idx,
    get_class_csv_name,
    normalize_class_name,
)


GROUP_RERANK_MODES = {"none", "soft_group_rerank", "hard_group_filter"}
GROUP_BONUS_MODES = {"hard", "prob"}


def dataset_root(cfg):
    dataset_mode = str(getattr(cfg.DATASETS, "MODE", "challenge_only")).lower()
    if dataset_mode == "external_only":
        root = getattr(cfg.DATASETS, "EXTERNAL_ROOT", "")
    else:
        root = getattr(cfg.DATASETS, "ROOT_DIR", "")
    if isinstance(root, (tuple, list)):
        root = root[0]
    return str(root)


def group_rerank_mode(cfg):
    return str(getattr(cfg.TEST, "RERANK_MODE", "none")).lower()


def _is_valid_label(label):
    labels = _class_array([label])
    if labels is None:
        return False
    valid = _valid_class_mask(labels)
    return bool(valid is not None and valid[0])


def _split_members(members):
    return [
        normalize_class_name(member)
        for member in re.split(r"\s+(?:u|U)\s+|[|,+;/]", str(members))
        if normalize_class_name(member)
    ]


def _parse_group_spec(spec):
    spec = str(spec).strip()
    if not spec:
        return None, []

    if "=" in spec:
        group_name, members = spec.split("=", 1)
        group_name = normalize_class_name(group_name)
    else:
        members = spec
        group_name = normalize_class_name(spec)
    return group_name, _split_members(members)


def _read_group_file(path):
    mapping = {}
    if not path or not os.path.exists(path):
        return mapping

    with open(path, newline="") as csv_file:
        sample = csv_file.read(2048)
        csv_file.seek(0)
        if "," not in sample and "\t" not in sample:
            for line in csv_file:
                group_name, members = _parse_group_spec(line)
                for member in members:
                    mapping[member] = group_name
            return mapping

        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            return mapping
        columns = {col.strip(): col for col in reader.fieldnames}
        class_col = columns.get("Class") or columns.get("class") or columns.get("label")
        group_col = columns.get("Group") or columns.get("group") or columns.get("class_group")
        if class_col is None or group_col is None:
            return mapping
        for row in reader:
            class_name = normalize_class_name(row.get(class_col, ""))
            group_name = normalize_class_name(row.get(group_col, ""))
            if class_name and group_name:
                mapping[class_name] = group_name
    return mapping


def class_group_mapping(cfg):
    mapping = {}
    root = dataset_root(cfg)
    group_file = str(getattr(cfg.TEST, "CLASS_GROUP_FILE", "") or "")
    if group_file:
        candidates = [group_file]
        if not os.path.isabs(group_file):
            candidates.insert(0, os.path.join(root, group_file))
        for candidate in candidates:
            mapping.update(_read_group_file(candidate))
            if mapping:
                break

    for spec in getattr(cfg.TEST, "CLASS_GROUPS", []):
        group_name, members = _parse_group_spec(spec)
        for member in members:
            mapping[member] = group_name
    return mapping


def labels_to_groups(labels, mapping=None):
    labels = _class_array(labels)
    if labels is None:
        return None

    mapping = mapping or {}
    groups = []
    for label in labels:
        if not _is_valid_label(label):
            groups.append("")
            continue
        key = normalize_class_name(label)
        groups.append(mapping.get(key, key))
    return np.asarray(groups, dtype=object)


def class_names_for_probabilities(cfg):
    root = dataset_root(cfg)
    train_csv = get_class_csv_name(cfg, "TRAIN_CSV", "train_classes.csv")
    class_to_idx = build_class_to_idx(root, train_csv)
    if not class_to_idx:
        return None
    names = [None] * len(class_to_idx)
    for class_name, class_idx in class_to_idx.items():
        names[class_idx] = class_name
    return names


def classes_for_paths(image_to_class, image_paths):
    if not image_to_class or not image_paths:
        return None

    classes = []
    found = 0
    for image_path in image_paths:
        image_name = os.path.basename(str(image_path))
        class_name = image_to_class.get(str(image_path)) or image_to_class.get(image_name)
        if class_name is None:
            classes.append("")
        else:
            classes.append(class_name)
            found += 1
    if found == 0:
        return None
    return np.asarray(classes, dtype=object)


def read_group_label_csv(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return {}

    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            return {}

        columns = {col.strip(): col for col in reader.fieldnames}
        image_col = columns.get("imageName") or columns.get("image_name")
        label_col = (
            columns.get("Group")
            or columns.get("group")
            or columns.get("ClassGroup")
            or columns.get("class_group")
            or columns.get("Class")
            or columns.get("class")
        )
        if image_col is None or label_col is None:
            return {}

        image_to_group = {}
        for row in reader:
            image_name = str(row.get(image_col, "")).strip()
            group_name = normalize_class_name(row.get(label_col, ""))
            if image_name and group_name:
                image_to_group[image_name] = group_name
        return image_to_group


def load_csv_query_gallery_classes(cfg, image_paths, num_query):
    if not image_paths or len(image_paths) < num_query:
        return None, None

    root = dataset_root(cfg)
    query_csv = str(getattr(cfg.TEST, "QUERY_GROUP_CSV", "") or get_class_csv_name(cfg, "QUERY_CSV", "query_classes.csv"))
    gallery_csv = str(getattr(cfg.TEST, "GALLERY_GROUP_CSV", "") or get_class_csv_name(cfg, "TEST_CSV", "test_classes.csv"))
    q_classes = classes_for_paths(
        read_group_label_csv(os.path.join(root, query_csv)),
        image_paths[:num_query],
    )
    g_classes = classes_for_paths(
        read_group_label_csv(os.path.join(root, gallery_csv)),
        image_paths[num_query:],
    )
    if q_classes is None or g_classes is None:
        return None, None
    return q_classes, g_classes


def resolve_query_gallery_classes(cfg, pred_classes, image_paths, num_query):
    source = str(getattr(cfg.TEST, "GROUP_CLASS_SOURCE", "auto")).lower()
    if source in ("auto", "csv"):
        q_csv, g_csv = load_csv_query_gallery_classes(cfg, image_paths, num_query)
        if q_csv is not None and g_csv is not None:
            return q_csv, g_csv, "csv"
        if source == "csv":
            return None, None, "csv_missing"

    if pred_classes is None or len(pred_classes) < num_query:
        return None, None, "missing"
    pred_classes = np.asarray(pred_classes)
    return pred_classes[:num_query], pred_classes[num_query:], "predicted"


def _probability_group_bonus(q_class_probs, g_groups, class_names, mapping):
    if q_class_probs is None or class_names is None:
        return None

    if torch.is_tensor(q_class_probs):
        q_class_probs = q_class_probs.detach().cpu().numpy()
    q_class_probs = np.asarray(q_class_probs, dtype=np.float32)
    if q_class_probs.ndim != 2 or q_class_probs.shape[1] != len(class_names):
        return None

    class_groups = labels_to_groups(class_names, mapping)
    if class_groups is None:
        return None

    group_names = sorted(set(str(group) for group in class_groups if _is_valid_label(group)))
    if not group_names:
        return None

    group_to_col = {group: idx for idx, group in enumerate(group_names)}
    q_group_probs = np.zeros((q_class_probs.shape[0], len(group_names)), dtype=np.float32)
    for class_idx, group in enumerate(class_groups):
        group = str(group)
        if group in group_to_col:
            q_group_probs[:, group_to_col[group]] += q_class_probs[:, class_idx]

    bonus = np.zeros((q_class_probs.shape[0], len(g_groups)), dtype=np.float32)
    for gallery_idx, group in enumerate(g_groups):
        group = str(group)
        if group in group_to_col:
            bonus[:, gallery_idx] = q_group_probs[:, group_to_col[group]]
    return bonus


def _hard_group_bonus(q_groups, g_groups):
    q_valid = _valid_class_mask(q_groups)
    g_valid = _valid_class_mask(g_groups)
    if q_valid is None or g_valid is None:
        return None
    valid = q_valid[:, np.newaxis] & g_valid[np.newaxis, :]
    return (valid & (q_groups[:, np.newaxis] == g_groups[np.newaxis, :])).astype(np.float32)


def apply_group_rerank_to_distmat(
    distmat,
    q_classes,
    g_classes,
    cfg,
    q_class_probs=None,
    class_names=None,
):
    mode = group_rerank_mode(cfg)
    if mode not in GROUP_RERANK_MODES:
        raise ValueError(
            "Unsupported TEST.RERANK_MODE '{}'. Expected one of {}.".format(
                mode, sorted(GROUP_RERANK_MODES)
            )
        )
    if mode == "none":
        return np.asarray(distmat, dtype=np.float32), {"mode": mode, "changed_queries": 0}

    mapping = class_group_mapping(cfg)
    q_groups = labels_to_groups(q_classes, mapping)
    g_groups = labels_to_groups(g_classes, mapping)
    if q_groups is None or g_groups is None:
        return np.asarray(distmat, dtype=np.float32), {
            "mode": mode,
            "skipped": True,
            "reason": "missing query/gallery classes",
            "changed_queries": 0,
        }

    distmat = np.asarray(distmat, dtype=np.float32)
    base_order = np.argsort(distmat, axis=1, kind="mergesort")

    if mode == "soft_group_rerank":
        bonus_mode = str(getattr(cfg.TEST, "GROUP_BONUS_MODE", "hard")).lower()
        if bonus_mode not in GROUP_BONUS_MODES:
            raise ValueError(
                "Unsupported TEST.GROUP_BONUS_MODE '{}'. Expected one of {}.".format(
                    bonus_mode, sorted(GROUP_BONUS_MODES)
                )
            )

        bonus = None
        if bonus_mode == "prob":
            bonus = _probability_group_bonus(q_class_probs, g_groups, class_names, mapping)
        if bonus is None:
            bonus_mode = "hard"
            bonus = _hard_group_bonus(q_groups, g_groups)
        if bonus is None:
            return distmat, {
                "mode": mode,
                "skipped": True,
                "reason": "missing usable group bonus",
                "changed_queries": 0,
            }

        lambda_group = float(getattr(cfg.TEST, "GROUP_RERANK_LAMBDA", 0.2))
        # Distances rank ascending, so retrieval_score is -distance.
        # Explicit requested formula:
        # final_score = retrieval_score + lambda_group * group_bonus
        final_score = -distmat + lambda_group * bonus
        adjusted = -final_score

    else:
        adjusted = distmat.copy()
        q_valid = _valid_class_mask(q_groups)
        g_valid = _valid_class_mask(g_groups)
        fallback_count = 0
        for query_idx, group in enumerate(q_groups):
            if q_valid is None or g_valid is None or not q_valid[query_idx]:
                fallback_count += 1
                continue
            keep = g_valid & (g_groups == group)
            if not np.any(keep):
                fallback_count += 1
                continue
            adjusted[query_idx, ~keep] = np.inf
        bonus_mode = "filter"
        lambda_group = None

    new_order = np.argsort(adjusted, axis=1, kind="mergesort")
    changed_queries = int(np.sum(base_order[:, 0] != new_order[:, 0])) if base_order.size else 0
    info = {
        "mode": mode,
        "bonus_mode": bonus_mode,
        "lambda_group": lambda_group,
        "class_group_source": str(getattr(cfg.TEST, "GROUP_CLASS_SOURCE", "auto")).lower(),
        "num_query_groups": int(len(set(q_groups.tolist()))),
        "num_gallery_groups": int(len(set(g_groups.tolist()))),
        "changed_queries": changed_queries,
        "total_queries": int(distmat.shape[0]),
    }
    if mode == "hard_group_filter":
        info["fallback_queries"] = int(fallback_count)
    return adjusted, info


def rank_change_examples(base_distmat, final_distmat, image_paths, num_query, q_classes, g_classes, limit=5, mapping=None):
    limit = int(limit)
    if limit <= 0 or image_paths is None or len(image_paths) < num_query:
        return []

    q_paths = list(image_paths[:num_query])
    g_paths = list(image_paths[num_query:])
    q_groups = labels_to_groups(q_classes, mapping)
    g_groups = labels_to_groups(g_classes, mapping)
    if q_groups is None or g_groups is None:
        return []

    base_order = np.argsort(np.asarray(base_distmat), axis=1, kind="mergesort")
    final_order = np.argsort(np.asarray(final_distmat), axis=1, kind="mergesort")
    examples = []
    for query_idx in range(base_order.shape[0]):
        old_top = int(base_order[query_idx, 0])
        new_top = int(final_order[query_idx, 0])
        if old_top == new_top:
            continue
        examples.append(
            {
                "query": os.path.basename(str(q_paths[query_idx])),
                "query_group": str(q_groups[query_idx]),
                "old_top1": os.path.basename(str(g_paths[old_top])),
                "old_top1_group": str(g_groups[old_top]),
                "new_top1": os.path.basename(str(g_paths[new_top])),
                "new_top1_group": str(g_groups[new_top]),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def log_group_rerank(logger, info, examples=None):
    if info is None:
        return
    logger.info("Group rerank: {}".format(info))
    for idx, example in enumerate(examples or [], start=1):
        logger.info("Group rerank changed example {}: {}".format(idx, example))


def save_metrics(cfg, dataset_name, cmc, mAP, group_info=None):
    metrics_path = str(getattr(cfg.TEST, "METRICS_PATH", "") or "")
    if not metrics_path:
        output_dir = os.path.join(str(getattr(cfg, "LOG_ROOT", "")), str(getattr(cfg, "LOG_NAME", "")))
        if not output_dir.strip() or output_dir == ".":
            return None
        safe_dataset = str(dataset_name or "dataset").replace("/", "_")
        safe_mode = group_rerank_mode(cfg).replace("/", "_")
        metrics_path = os.path.join(output_dir, "metrics_{}_{}.json".format(safe_dataset, safe_mode))

    parent = os.path.dirname(metrics_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = {
        "dataset": dataset_name,
        "mAP": float(mAP),
        "rank1": float(cmc[0]) if len(cmc) > 0 else None,
        "rank5": float(cmc[4]) if len(cmc) > 4 else None,
        "rank10": float(cmc[9]) if len(cmc) > 9 else None,
        "rerank_mode": group_rerank_mode(cfg),
        "group_rerank": group_info or {},
    }
    with open(metrics_path, "w") as metrics_file:
        json.dump(payload, metrics_file, indent=2, sort_keys=True)
    logging.getLogger("PAT.test").info("Saved metrics to {}".format(metrics_path))
    return metrics_path
