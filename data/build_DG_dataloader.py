import os
import torch
import sys
import collections.abc as container_abcs
import logging
from types import SimpleNamespace

# from torch._six import container_abcs, string_classes, int_classes
int_classes = int
string_classes = str
from torch.utils.data import DataLoader
from utils import comm
import random

from . import samplers
from .common import CommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms
from utils.class_aware import (
    CLASS_ID_KEY,
    is_class_aware_enabled,
    normalize_class_name,
    read_class_csv,
)

_root = os.getenv("REID_DATASETS", "../../data")


def _uses_urban_class_csv(dataset_name):
    return dataset_name in (
        'UrbanElementsReID',
        'UrbanElementsReID_test',
        'UrbanElementsReID_localval_train',
        'UrbanElementsReID_localval_test',
    )


def _as_root(root):
    if isinstance(root, (tuple, list)):
        return root[0]
    return root


def _external_root(cfg):
    external_root = str(getattr(cfg.DATASETS, "EXTERNAL_ROOT", ""))
    if not external_root:
        raise ValueError("DATASETS.EXTERNAL_ROOT must be set when DATASETS.MODE uses external data")
    return external_root


def _dataset_root(cfg, dataset_name=None):
    dataset_mode = str(getattr(cfg.DATASETS, "MODE", "challenge_only")).lower()
    if dataset_mode == "challenge_only":
        return cfg.DATASETS.ROOT_DIR
    if dataset_mode == "external_only":
        return _external_root(cfg)
    if dataset_mode == "mixed_train":
        test_names = getattr(cfg.DATASETS, "TEST", ())
        if isinstance(test_names, str):
            test_names = (test_names,)
        if dataset_name is not None and test_names and str(dataset_name) != str(test_names[0]):
            return _external_root(cfg)
        return cfg.DATASETS.ROOT_DIR
    raise ValueError("Unsupported DATASETS.MODE '{}'. Expected 'challenge_only', 'external_only', or 'mixed_train'.".format(dataset_mode))


def _default_class_cfg():
    return SimpleNamespace(
        TRAIN_CSV="train_classes.csv",
        QUERY_CSV="query_classes.csv",
        TEST_CSV="test_classes.csv",
    )


def _class_cfg_for_dataset(cfg, dataset_name, root, primary_root):
    if not is_class_aware_enabled(cfg) or not _uses_urban_class_csv(dataset_name):
        return None
    if os.path.abspath(str(_as_root(root))) != os.path.abspath(str(_as_root(primary_root))):
        return _default_class_cfg()
    return cfg.MODEL.CLASS_AWARE


def _dataset_kwargs(cfg, dataset_name, root, primary_root, combineall=False):
    kwargs = {"root": root}
    if combineall:
        kwargs["combineall"] = cfg.DATASETS.COMBINEALL
    class_cfg = _class_cfg_for_dataset(cfg, dataset_name, root, primary_root)
    if class_cfg is not None:
        kwargs["class_aware"] = True
        kwargs["class_aware_cfg"] = class_cfg
    return kwargs


def _num_ids(items):
    return len({item[1] for item in items})


def _offset_pids(items, offset):
    adjusted = []
    for item in items:
        item = tuple(item)
        if len(item) > 3 and isinstance(item[-1], dict):
            adjusted.append((item[0], item[1] + offset, item[2], dict(item[-1])))
        else:
            adjusted.append((item[0], item[1] + offset, item[2]))
    return adjusted


def _sample_to_external_ratio(challenge_items, external_items, ratio, seed):
    ratio = max(0.0, min(float(ratio), 0.95))
    if ratio <= 0.0 or not external_items:
        return [], 0.0
    desired = int(round((ratio / (1.0 - ratio)) * len(challenge_items)))
    desired = max(1, desired)
    rng = random.Random(seed)
    if len(external_items) >= desired:
        selected = list(external_items)
        rng.shuffle(selected)
        selected = selected[:desired]
    else:
        repeats = desired // len(external_items)
        remainder = desired % len(external_items)
        selected = list(external_items) * repeats
        extra = list(external_items)
        rng.shuffle(extra)
        selected.extend(extra[:remainder])
    actual_ratio = len(selected) / max(1, len(challenge_items) + len(selected))
    return selected, actual_ratio


def _class_name_for_item(item, image_to_class):
    image_name = os.path.basename(str(item[0]))
    return image_to_class.get(image_name) or image_to_class.get(str(item[0]))


def _remap_semantic_classes(items, root, class_csv, unified_class_to_idx):
    image_to_class = read_class_csv(os.path.join(str(root), class_csv))
    remapped = []
    for item in items:
        item = tuple(item)
        metadata = dict(item[-1]) if len(item) > 3 and isinstance(item[-1], dict) else {}
        base = item[:-1] if metadata else item
        class_name = _class_name_for_item(base, image_to_class)
        if class_name is not None:
            metadata[CLASS_ID_KEY] = unified_class_to_idx.get(normalize_class_name(class_name), -1)
        if metadata:
            remapped.append(tuple(base) + (metadata,))
        else:
            remapped.append(tuple(base))
    return remapped


def _mixed_class_map(cfg, challenge_root, external_root):
    challenge_csv = getattr(cfg.MODEL.CLASS_AWARE, "TRAIN_CSV", "train_classes.csv")
    external_csv = "train_classes.csv"
    class_names = set()
    class_names.update(read_class_csv(os.path.join(str(challenge_root), challenge_csv)).values())
    class_names.update(read_class_csv(os.path.join(str(external_root), external_csv)).values())
    normalized = sorted({normalize_class_name(name) for name in class_names if normalize_class_name(name)})
    return {class_name: idx for idx, class_name in enumerate(normalized)}


def _apply_domain(items, domain_idx, camera_to_domain, camera_all):
    updated = []
    for item in items:
        item = tuple(item)
        if len(item) > 3 and isinstance(item[-1], dict):
            add_info = dict(item[-1])
            item = item[:-1]
        else:
            add_info = {}

        if camera_to_domain:
            add_info['domains'] = item[2]
            camera_all.append(item[2])
        else:
            add_info['domains'] = int(domain_idx)
        updated.append(tuple(item) + (add_info,))
    return updated


def _load_dataset(cfg, dataset_name, root, primary_root, combineall=False):
    if dataset_name == 'CUHK03_NP':
        return DATASET_REGISTRY.get('CUHK03')(root=root, cuhk03_labeled=False)
    return DATASET_REGISTRY.get(dataset_name)(
        **_dataset_kwargs(cfg, dataset_name, root, primary_root, combineall=combineall)
    )


def _build_mixed_train_items(cfg):
    logger = logging.getLogger('PAT')
    challenge_root = _as_root(cfg.DATASETS.ROOT_DIR)
    external_root = _external_root(cfg)
    train_names = getattr(cfg.DATASETS, "TRAIN", ())
    if isinstance(train_names, str):
        train_names = (train_names,)
    challenge_name = str(train_names[0]) if train_names else "UrbanElementsReID_localval_train"
    external_name = str(train_names[1]) if len(train_names) > 1 else "UrbanElementsReID"

    challenge_dataset = _load_dataset(cfg, challenge_name, challenge_root, challenge_root, combineall=True)
    external_dataset = _load_dataset(cfg, external_name, external_root, challenge_root, combineall=True)

    challenge_items = list(challenge_dataset.train)
    external_items = list(external_dataset.train)
    challenge_ids = _num_ids(challenge_items)
    external_ids = _num_ids(external_items)

    if is_class_aware_enabled(cfg):
        unified_class_to_idx = _mixed_class_map(cfg, challenge_root, external_root)
        challenge_csv = getattr(cfg.MODEL.CLASS_AWARE, "TRAIN_CSV", "train_classes.csv")
        challenge_items = _remap_semantic_classes(challenge_items, challenge_root, challenge_csv, unified_class_to_idx)
        external_items = _remap_semantic_classes(external_items, external_root, "train_classes.csv", unified_class_to_idx)

    pid_offset = max([item[1] for item in challenge_items], default=-1) + 1
    external_items = _offset_pids(external_items, pid_offset)
    selected_external, actual_ratio = _sample_to_external_ratio(
        challenge_items,
        external_items,
        getattr(cfg.DATALOADER, "MIXED_EXTERNAL_RATIO", 0.2),
        getattr(cfg.SOLVER, "SEED", 1234),
    )

    camera_all = []
    mixed_items = []
    mixed_items.extend(_apply_domain(challenge_items, 0, cfg.DATALOADER.CAMERA_TO_DOMAIN, camera_all))
    mixed_items.extend(_apply_domain(selected_external, 1, cfg.DATALOADER.CAMERA_TO_DOMAIN, camera_all))

    if comm.is_main_process():
        logger.info("=> Mixed train mode")
        logger.info("  challenge train dataset/root: {} / {}".format(challenge_name, challenge_root))
        logger.info("  external train dataset/root: {} / {}".format(external_name, external_root))
        logger.info("  challenge local train ids/images: {}/{}".format(challenge_ids, len(challenge_items)))
        logger.info("  external train ids/images: {}/{}".format(external_ids, len(external_items)))
        logger.info("  selected external images after ratio: {}".format(len(selected_external)))
        logger.info("  total mixed train ids/images: {}/{}".format(_num_ids(mixed_items), len(mixed_items)))
        logger.info("  target external ratio: {:.3f}; actual external ratio: {:.3f}".format(
            float(getattr(cfg.DATALOADER, "MIXED_EXTERNAL_RATIO", 0.2)),
            actual_ratio,
        ))

    return mixed_items


def build_reid_train_loader(cfg):
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('*'*100)
        print('Hmm, Big Debugger is watching me')
        print('*'*100)
        num_workers = 0
    else:
        num_workers = cfg.DATALOADER.NUM_WORKERS

    train_transforms = build_transforms(cfg, is_train=True, is_fake=False)
    train_items = list()
    domain_idx = 0
    camera_all = list()

    # load datasets
    if str(getattr(cfg.DATASETS, "MODE", "challenge_only")).lower() == "mixed_train":
        train_items = _build_mixed_train_items(cfg)
    else:
        _root = _dataset_root(cfg)
        primary_root = _root
        for d in cfg.DATASETS.TRAIN:
            dataset = _load_dataset(cfg, d, _root, primary_root, combineall=True)
            if comm.is_main_process():
                dataset.show_train()
            dataset.train = _apply_domain(dataset.train, domain_idx, cfg.DATALOADER.CAMERA_TO_DOMAIN, camera_all)
            domain_idx += 1
            train_items.extend(dataset.train)

    train_set = CommDataset(train_items, train_transforms, relabel=True)

    train_loader = make_sampler(
        train_set=train_set,
        num_batch=cfg.SOLVER.IMS_PER_BATCH,
        num_instance=cfg.DATALOADER.NUM_INSTANCE,
        num_workers=num_workers,
        mini_batch_size=cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size(),
        drop_last=cfg.DATALOADER.DROP_LAST,
        flag1=cfg.DATALOADER.NAIVE_WAY,
        flag2=cfg.DATALOADER.DELETE_REM,
        cfg = cfg)

    return train_loader


def build_reid_test_loader(cfg, dataset_name, opt=None, flag_test=True, shuffle=False, only_gallery=False, only_query=False, eval_time=False):
    test_transforms = build_transforms(cfg, is_train=False)
    _root = _dataset_root(cfg, dataset_name)
    primary_root = _dataset_root(cfg)
    dataset_kwargs = _dataset_kwargs(cfg, dataset_name, _root, primary_root, combineall=False)
    if opt is None:
        dataset = DATASET_REGISTRY.get(dataset_name)(**dataset_kwargs)
        if comm.is_main_process():
            if flag_test:
                dataset.show_test()
            else:
                dataset.show_train()
    else:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=[_root, opt])
    if flag_test:
        if only_gallery:
            test_items = dataset.gallery
        elif only_query:
            test_set = CommDataset([random.choice(dataset.query)], test_transforms, relabel=False)
            return test_set
        else:
            test_items = dataset.query + dataset.gallery
        if shuffle: # only for visualization
            random.shuffle(test_items)
    else:
        test_items = dataset.train

    test_set = CommDataset(test_items, test_transforms, relabel=False)

    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        num_workers = 0
    else:
        num_workers = cfg.DATALOADER.NUM_WORKERS

    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=fast_batch_collator)
    return test_loader, len(dataset.query)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
    elif isinstance(elem, list):
        out_g = []
        out_pt1 = []
        out_pt2 = []
        out_pt3 = []
        # out = torch.stack(elem, dim=0)
        for i, tensor_list in enumerate(batched_inputs):
            out_g.append(tensor_list[0])
            out_pt1.append(tensor_list[1])
            out_pt2.append(tensor_list[2])
            out_pt3.append(tensor_list[3])
        out = torch.stack(out_g, dim=0)
        out_pt1 = torch.stack(out_pt1, dim=0)
        out_pt2 = torch.stack(out_pt2, dim=0)
        out_pt3 = torch.stack(out_pt3, dim=0)
        return out, out_pt1, out_pt2, out_pt3


def make_sampler(train_set, num_batch, num_instance, num_workers,
                 mini_batch_size, drop_last=True, flag1=True, flag2=True, seed=None, cfg=None):

    if flag1:
        data_sampler = samplers.RandomIdentitySampler(train_set.img_items,
                                                      mini_batch_size, num_instance)
    else:
        data_sampler = samplers.DomainSuffleSampler(train_set.img_items,
                                                     num_batch, num_instance, flag2, seed, cfg)
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, drop_last)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )
    return train_loader
