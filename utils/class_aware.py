import csv
import os

import numpy as np
import torch
import torch.nn.functional as F


CLASS_ID_KEY = "class_id"


def get_class_aware_cfg(cfg):
    model_cfg = getattr(cfg, "MODEL", None)
    if model_cfg is None:
        return None
    return getattr(model_cfg, "CLASS_AWARE", None)


def is_class_aware_enabled(cfg):
    class_cfg = get_class_aware_cfg(cfg)
    return bool(getattr(class_cfg, "ENABLED", False)) if class_cfg is not None else False


def get_class_loss_weight(cfg):
    class_cfg = get_class_aware_cfg(cfg)
    return float(getattr(class_cfg, "LOSS_WEIGHT", 1.0)) if class_cfg is not None else 1.0


def get_class_distance_penalty(cfg):
    class_cfg = get_class_aware_cfg(cfg)
    return float(getattr(class_cfg, "DISTANCE_PENALTY", 0.0)) if class_cfg is not None else 0.0


def get_retrieval_class_source(cfg):
    class_cfg = get_class_aware_cfg(cfg)
    return str(getattr(class_cfg, "RETRIEVAL_SOURCE", "predicted")).lower() if class_cfg is not None else "predicted"


def use_metadata_classes_for_retrieval(cfg):
    return get_retrieval_class_source(cfg) == "metadata"


def get_class_csv_name(cfg, key, default):
    class_cfg = get_class_aware_cfg(cfg)
    return str(getattr(class_cfg, key, default)) if class_cfg is not None else default


def normalize_class_name(class_name):
    return str(class_name).strip().lower()


def read_class_csv(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return {}

    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            return {}

        columns = {col.strip(): col for col in reader.fieldnames}
        image_col = columns.get("imageName") or columns.get("image_name")
        class_col = columns.get("Class") or columns.get("class")
        if image_col is None or class_col is None:
            return {}

        image_to_class = {}
        for row in reader:
            image_name = str(row.get(image_col, "")).strip()
            class_name = normalize_class_name(row.get(class_col, ""))
            if image_name and class_name:
                image_to_class[image_name] = class_name
        return image_to_class


def build_class_to_idx(dataset_root, train_class_csv="train_classes.csv"):
    csv_path = os.path.join(dataset_root, train_class_csv)
    image_to_class = read_class_csv(csv_path)
    class_names = sorted(set(image_to_class.values()))
    return {class_name: idx for idx, class_name in enumerate(class_names)}


def infer_num_semantic_classes(cfg):
    if not is_class_aware_enabled(cfg):
        return 0

    class_cfg = get_class_aware_cfg(cfg)
    configured_num = int(getattr(class_cfg, "NUM_CLASSES", 0)) if class_cfg is not None else 0
    if configured_num > 0:
        return configured_num

    dataset_root = getattr(cfg.DATASETS, "ROOT_DIR", "")
    if isinstance(dataset_root, (tuple, list)):
        dataset_root = dataset_root[0]
    train_csv = get_class_csv_name(cfg, "TRAIN_CSV", "train_classes.csv")
    return len(build_class_to_idx(str(dataset_root), train_csv))


def add_class_metadata(item, image_name, image_to_class, class_to_idx):
    class_name = image_to_class.get(image_name) or image_to_class.get(os.path.basename(image_name))
    if class_name is None:
        class_id = -1
    else:
        class_id = class_to_idx.get(normalize_class_name(class_name), -1)

    if len(item) > 3 and isinstance(item[-1], dict):
        metadata = dict(item[-1])
        base = tuple(item[:-1])
    else:
        metadata = {}
        base = tuple(item)

    metadata[CLASS_ID_KEY] = int(class_id)
    return base + (metadata,)


def get_batch_class_targets(informations, device=None):
    others = informations.get("others", {})
    if not isinstance(others, dict) or CLASS_ID_KEY not in others:
        return None

    targets = others[CLASS_ID_KEY]
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)
    targets = targets.long()
    if device is not None:
        targets = targets.to(device)
    return targets


def semantic_classification_loss(logits, targets):
    if logits is None or targets is None:
        return None
    valid = targets >= 0
    if not torch.any(valid):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets[valid])


def semantic_accuracy(logits, targets):
    if logits is None or targets is None:
        return None
    valid = targets >= 0
    if not torch.any(valid):
        return None
    return (logits[valid].argmax(1) == targets[valid]).float().mean()


def get_model_module(model):
    return model.module if hasattr(model, "module") else model


def model_is_class_aware(model):
    return bool(getattr(get_model_module(model), "class_aware_enabled", False))


def model_num_semantic_classes(model):
    return int(getattr(get_model_module(model), "num_semantic_classes", 0))


def split_inference_output(output):
    if isinstance(output, tuple) and len(output) == 2:
        return output
    return output, None


def apply_class_distance_penalty(distmat, q_classes, g_classes, penalty):
    if penalty <= 0 or q_classes is None or g_classes is None:
        return distmat

    penalty_matrix = class_distance_penalty_matrix(q_classes, g_classes, penalty)
    if penalty_matrix is None:
        return distmat

    return np.asarray(distmat) + penalty_matrix


def class_distance_penalty_matrix(q_classes, g_classes, penalty):
    if penalty <= 0 or q_classes is None or g_classes is None:
        return None

    if torch.is_tensor(q_classes):
        q_classes = q_classes.cpu().numpy()
    if torch.is_tensor(g_classes):
        g_classes = g_classes.cpu().numpy()

    q_classes = np.asarray(q_classes).reshape(-1)
    g_classes = np.asarray(g_classes).reshape(-1)
    if q_classes.size == 0 or g_classes.size == 0:
        return None

    valid = (q_classes[:, np.newaxis] >= 0) & (g_classes[np.newaxis, :] >= 0)
    mismatch = valid & (q_classes[:, np.newaxis] != g_classes[np.newaxis, :])
    penalty_matrix = np.zeros((q_classes.size, g_classes.size), dtype=np.float32)
    penalty_matrix[mismatch] = penalty
    return penalty_matrix
