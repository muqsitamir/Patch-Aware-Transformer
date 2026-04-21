import numpy as np
import torch
import torch.nn.functional as F

from utils.class_aware import (
    _class_array,
    _valid_class_mask,
    class_match_matrix,
    class_mismatch_matrix,
)


CLASS_POSTPROCESS_MODES = {
    "off",
    "penalty_additive",
    "same_class_first",
    "same_class_only_topk",
    "hard_same_class_only",
    "strict_same_class_backfill",
}

CLASS_SCORE_MODES = {"additive", "multiplicative", "reorder"}


def normalize_features(features, eps=1e-12):
    if torch.is_tensor(features):
        return F.normalize(features, dim=1, p=2, eps=eps)

    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def apply_query_expansion(qf, gf, topk=5, alpha=1.0):
    """Average each query with its top-k gallery neighbors, then L2-normalize."""
    topk = int(topk)
    alpha = float(alpha)
    if gf.shape[0] == 0:
        return normalize_features(qf), normalize_features(gf)

    qf = normalize_features(qf)
    gf = normalize_features(gf)
    if topk <= 0:
        return qf, gf

    k = min(topk, gf.shape[0])
    if torch.is_tensor(qf):
        sim = torch.mm(qf, gf.t())
        top_indices = torch.topk(sim, k=k, dim=1).indices
        neighbor_mean = gf[top_indices].mean(dim=1)
        return normalize_features(alpha * qf + neighbor_mean), gf

    sim = np.dot(qf, gf.T)
    top_indices = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    neighbor_mean = gf[top_indices].mean(axis=1)
    return normalize_features(alpha * qf + neighbor_mean), gf


def apply_class_score_bias(distmat, q_classes, g_classes, penalty=0.0, score_mode="additive", scale=1.2):
    """Return a distance matrix where lower values still rank first."""
    score_mode = str(score_mode).lower()
    if score_mode not in CLASS_SCORE_MODES:
        raise ValueError("Unsupported CLASS_SCORE_MODE '{}'. Expected one of {}.".format(
            score_mode, sorted(CLASS_SCORE_MODES)))

    mismatch = class_mismatch_matrix(q_classes, g_classes)
    if mismatch is None:
        return np.asarray(distmat)

    adjusted = np.asarray(distmat, dtype=np.float32).copy()
    if score_mode == "additive":
        if penalty > 0:
            adjusted[mismatch] += float(penalty)
    elif score_mode == "multiplicative":
        adjusted[mismatch] *= float(scale)
    return adjusted


def _argsort_distmat(distmat):
    return np.argsort(np.asarray(distmat), axis=1, kind="mergesort")


def _same_class_first(base_order, same_mask, output_topk):
    result = np.empty((base_order.shape[0], output_topk), dtype=base_order.dtype)
    for i, order in enumerate(base_order):
        # Same-class candidates keep their visual-distance order, then all
        # different/unknown-class candidates keep their visual-distance order.
        same_order = order[same_mask[i, order]]
        other_order = order[~same_mask[i, order]]
        result[i] = np.concatenate([same_order, other_order])[:output_topk]
    return result


def _same_class_only_topk(base_order, same_mask, output_topk, class_topk):
    result = np.empty((base_order.shape[0], output_topk), dtype=base_order.dtype)
    class_topk = max(0, min(int(class_topk), output_topk))
    num_gallery = base_order.shape[1]

    for i, order in enumerate(base_order):
        # The head of the ranking is class-constrained: up to CLASS_TOPK slots
        # are filled from same-class gallery items in visual-distance order.
        # The tail falls back to the original order so the CSV keeps 100 ids.
        same_take = order[same_mask[i, order]][:class_topk]
        used = np.zeros(num_gallery, dtype=bool)
        used[same_take] = True
        remainder = order[~used[order]]
        result[i] = np.concatenate([same_take, remainder])[:output_topk]
    return result


def _hard_same_class_only(base_order, same_mask, output_topk):
    result = np.empty((base_order.shape[0], output_topk), dtype=base_order.dtype)

    for i, order in enumerate(base_order):
        # When enough same-class gallery items exist, the returned list is
        # entirely same-class. If not, we backfill from other classes in the
        # original visual-distance order to keep the submission length stable.
        same_order = order[same_mask[i, order]]
        if same_order.size >= output_topk:
            result[i] = same_order[:output_topk]
            continue
        other_order = order[~same_mask[i, order]]
        result[i] = np.concatenate([same_order, other_order])[:output_topk]
    return result


def _strict_same_class_backfill(base_order, same_mask, output_topk):
    result = np.empty((base_order.shape[0], output_topk), dtype=base_order.dtype)

    for i, order in enumerate(base_order):
        # Strict mode first filters to same-class candidates in visual-distance
        # order. Cross-class candidates are used only if that filtered pool is
        # too small to fill the submission row.
        same_order = order[same_mask[i, order]]
        same_take = same_order[:output_topk]
        if same_take.size >= output_topk:
            result[i] = same_take
            continue
        other_order = order[~same_mask[i, order]]
        result[i] = np.concatenate([same_take, other_order])[:output_topk]
    return result


def class_postprocess_stats(q_classes, g_classes, class_topk=100):
    q_classes = _class_array(q_classes)
    g_classes = _class_array(g_classes)
    if q_classes is None or g_classes is None:
        return None

    q_valid = _valid_class_mask(q_classes)
    g_valid = _valid_class_mask(g_classes)
    if q_valid is None or g_valid is None:
        return None

    gallery_counts = {}
    valid_gallery_classes = g_classes[g_valid]
    if valid_gallery_classes.size > 0:
        labels, counts = np.unique(valid_gallery_classes, return_counts=True)
        gallery_counts = {str(label): int(count) for label, count in zip(labels, counts)}

    same_counts = np.zeros(q_classes.shape[0], dtype=np.int64)
    for i, (class_name, is_valid) in enumerate(zip(q_classes, q_valid)):
        if is_valid:
            same_counts[i] = gallery_counts.get(str(class_name), 0)

    query_class_counts = {}
    for class_name in q_classes[q_valid]:
        key = str(class_name)
        if key not in query_class_counts:
            query_class_counts[key] = gallery_counts.get(key, 0)

    if same_counts.size > 0:
        min_count = int(same_counts.min())
        max_count = int(same_counts.max())
        avg_count = float(same_counts.mean())
    else:
        min_count = 0
        max_count = 0
        avg_count = 0.0

    class_topk = max(0, int(class_topk))
    return {
        "gallery_counts": gallery_counts,
        "query_class_counts": query_class_counts,
        "same_count_min": min_count,
        "same_count_max": max_count,
        "same_count_avg": avg_count,
        "backfill_query_count": int(np.sum(same_counts < class_topk)),
        "num_queries": int(q_classes.shape[0]),
        "num_gallery": int(g_classes.shape[0]),
        "class_topk": class_topk,
        "missing_query_classes": int(np.sum(~q_valid)),
        "missing_gallery_classes": int(np.sum(~g_valid)),
    }


def build_class_postprocess_indices(
    distmat,
    q_classes=None,
    g_classes=None,
    mode="off",
    output_topk=100,
    class_topk=100,
    mismatch_penalty=0.0,
    score_mode="additive",
    scale=1.2,
):
    """Build submission indices from a final query-gallery distance matrix."""
    mode = str(mode).lower()
    score_mode = str(score_mode).lower()
    if mode not in CLASS_POSTPROCESS_MODES:
        raise ValueError("Unsupported CLASS_POSTPROCESS '{}'. Expected one of {}.".format(
            mode, sorted(CLASS_POSTPROCESS_MODES)))

    distmat = np.asarray(distmat, dtype=np.float32)
    output_topk = min(int(output_topk), distmat.shape[1])
    if output_topk <= 0:
        return np.empty((distmat.shape[0], 0), dtype=np.int64)

    if mode == "off" or q_classes is None or g_classes is None:
        return _argsort_distmat(distmat)[:, :output_topk]

    same_mask = class_match_matrix(q_classes, g_classes)
    if same_mask is None:
        return _argsort_distmat(distmat)[:, :output_topk]

    if mode == "penalty_additive":
        if score_mode == "reorder":
            return _same_class_first(_argsort_distmat(distmat), same_mask, output_topk)
        adjusted = apply_class_score_bias(
            distmat,
            q_classes,
            g_classes,
            penalty=mismatch_penalty,
            score_mode=score_mode,
            scale=scale,
        )
        return _argsort_distmat(adjusted)[:, :output_topk]

    base_order = _argsort_distmat(distmat)
    if mode == "same_class_first":
        return _same_class_first(base_order, same_mask, output_topk)
    if mode == "same_class_only_topk":
        return _same_class_only_topk(base_order, same_mask, output_topk, class_topk)
    if mode == "strict_same_class_backfill":
        return _strict_same_class_backfill(base_order, same_mask, output_topk)
    return _hard_same_class_only(base_order, same_mask, output_topk)
