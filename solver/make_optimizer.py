import logging
from collections import OrderedDict

import torch


_HEAD_PARAM_NAMES = ("classifier", "arcface", "semantic_head", "bottleneck")
_TOKEN_PARAM_NAMES = ("pos_embed", "cls_token", "part_token", "dist_token")


def _is_transformer_backbone(model):
    base = getattr(model, "base", None)
    return base is not None and hasattr(base, "blocks") and hasattr(base, "patch_embed")


def _transformer_num_layers(model):
    if not _is_transformer_backbone(model):
        return 0
    return len(model.base.blocks) + 1


def _transformer_layer_id(param_name, num_layers):
    if not param_name.startswith("base."):
        return num_layers

    base_name = param_name[len("base."):]
    if base_name.startswith("patch_embed") or base_name.startswith(_TOKEN_PARAM_NAMES):
        return 0
    if base_name.startswith("blocks."):
        parts = base_name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return min(num_layers, int(parts[1]) + 1)
    return num_layers


def _is_head_parameter(param_name):
    return any(param_name.startswith(name) or ".{}.".format(name) in param_name for name in _HEAD_PARAM_NAMES)


def _use_zero_weight_decay(cfg, param_name, param_value):
    if not bool(getattr(cfg.SOLVER, "ZERO_WD_1D", False)):
        return False
    if param_value.ndim <= 1:
        return True
    if param_name.endswith(".bias"):
        return True
    return any(token_name in param_name for token_name in _TOKEN_PARAM_NAMES)


def _param_group_key(lr, weight_decay):
    return "{:.12g}|{:.12g}".format(float(lr), float(weight_decay))


def _build_param_groups(cfg, model):
    base_lr = float(cfg.SOLVER.BASE_LR)
    bias_lr_factor = float(cfg.SOLVER.BIAS_LR_FACTOR)
    head_lr_factor = float(getattr(cfg.SOLVER, "HEAD_LR_FACTOR", 1.0))
    if bool(getattr(cfg.SOLVER, "LARGE_FC_LR", False)):
        head_lr_factor *= 2.0

    layer_decay_enabled = bool(getattr(cfg.SOLVER, "LAYER_DECAY_ENABLED", False)) and _is_transformer_backbone(model)
    layer_decay = float(getattr(cfg.SOLVER, "LAYER_DECAY", 0.75))
    num_layers = _transformer_num_layers(model)

    grouped = OrderedDict()
    stats = {
        "layer_decay_enabled": layer_decay_enabled,
        "layer_decay": layer_decay,
        "num_layers": num_layers,
        "head_lr_factor": head_lr_factor,
        "zero_wd_1d": bool(getattr(cfg.SOLVER, "ZERO_WD_1D", False)),
        "num_groups": 0,
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lr = base_lr
        weight_decay = float(cfg.SOLVER.WEIGHT_DECAY)

        if "bias" in name:
            lr = base_lr * bias_lr_factor
            weight_decay = float(cfg.SOLVER.WEIGHT_DECAY_BIAS)

        if layer_decay_enabled and name.startswith("base."):
            layer_id = _transformer_layer_id(name, num_layers)
            lr *= layer_decay ** (num_layers - layer_id)

        if _is_head_parameter(name):
            lr *= head_lr_factor

        if _use_zero_weight_decay(cfg, name, param):
            weight_decay = 0.0

        group_key = _param_group_key(lr, weight_decay)
        if group_key not in grouped:
            grouped[group_key] = {
                "params": [],
                "lr": lr,
                "weight_decay": weight_decay,
            }
        grouped[group_key]["params"].append(param)

    stats["num_groups"] = len(grouped)
    return list(grouped.values()), stats


def _log_optimizer_settings(cfg, stats):
    logger = logging.getLogger("PAT.train")
    logger.info(
        "Optimizer param groups: groups={} head_lr_factor={:.4f} layer_decay_enabled={} layer_decay={:.4f} transformer_layers={} zero_wd_1d={}".format(
            stats["num_groups"],
            stats["head_lr_factor"],
            stats["layer_decay_enabled"],
            stats["layer_decay"],
            stats["num_layers"],
            stats["zero_wd_1d"],
        )
    )
    if bool(getattr(cfg.SOLVER, "LARGE_FC_LR", False)):
        logger.info("Large FC LR enabled; effective head LR factor is {:.4f}".format(stats["head_lr_factor"]))


def make_optimizer(cfg, model):
    params, stats = _build_param_groups(cfg, model)
    _log_optimizer_settings(cfg, stats)

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
            params,
            momentum=cfg.SOLVER.MOMENTUM,
        )
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        betas = tuple(float(beta) for beta in getattr(cfg.SOLVER, "OPT_BETAS", (0.9, 0.999)))
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            betas=betas,
            eps=float(getattr(cfg.SOLVER, "OPT_EPS", 1e-8)),
        )
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    return optimizer
