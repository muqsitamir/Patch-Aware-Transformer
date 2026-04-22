import torch
import torch.nn.functional as F


def _as_scales(cfg):
    scales = getattr(cfg.TEST, "TTA_SCALES", [1.0])
    if isinstance(scales, (int, float)):
        scales = [float(scales)]

    clean_scales = []
    seen = set()
    for scale in scales:
        scale = float(scale)
        if scale <= 0:
            raise ValueError("TEST.TTA_SCALES values must be positive, got {}".format(scale))
        key = round(scale, 6)
        if key not in seen:
            clean_scales.append(scale)
            seen.add(key)
    return clean_scales or [1.0]


def tta_enabled(cfg):
    return bool(getattr(cfg.TEST, "TTA_ENABLED", False))


def tta_flip_enabled(cfg):
    return bool(getattr(cfg.TEST, "TTA_FLIP", True))


def tta_scales(cfg):
    if not tta_enabled(cfg):
        return [1.0]
    return _as_scales(cfg)


def tta_num_views(cfg):
    if not tta_enabled(cfg):
        return 1
    flip_multiplier = 2 if tta_flip_enabled(cfg) else 1
    return len(tta_scales(cfg)) * flip_multiplier


def tta_settings_string(cfg):
    return (
        "enabled={}, flip={}, scales={}, merge={}, views_per_image={}".format(
            tta_enabled(cfg),
            tta_flip_enabled(cfg) if tta_enabled(cfg) else False,
            tta_scales(cfg),
            str(getattr(cfg.TEST, "TTA_MERGE", "mean")),
            tta_num_views(cfg),
        )
    )


def log_tta_settings(logger, cfg):
    logger.info("TTA settings: {}".format(tta_settings_string(cfg)))


def _scaled_tensor(images, scale):
    if abs(float(scale) - 1.0) < 1e-6:
        return images

    original_size = images.shape[-2:]
    scaled = F.interpolate(
        images,
        scale_factor=float(scale),
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )
    if scaled.shape[-2:] != original_size:
        scaled = F.interpolate(
            scaled,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
    return scaled


def iter_tta_views(images, cfg):
    if not tta_enabled(cfg):
        yield images
        return

    merge = str(getattr(cfg.TEST, "TTA_MERGE", "mean")).lower()
    if merge != "mean":
        raise ValueError("Unsupported TEST.TTA_MERGE '{}'. Only 'mean' is supported.".format(merge))

    for scale in tta_scales(cfg):
        scaled = _scaled_tensor(images, scale)
        yield scaled
        if tta_flip_enabled(cfg):
            yield torch.flip(scaled, dims=[3])


def _split_inference_output(output):
    if isinstance(output, tuple) and len(output) == 2:
        return output
    return output, None


def extract_tta_features(model, images, cfg, return_class_logits=False):
    features = []
    class_logits = []

    for view in iter_tta_views(images, cfg):
        if return_class_logits:
            output = model(view, return_class_logits=True)
            feat, logits = _split_inference_output(output)
            if logits is not None:
                class_logits.append(logits.float())
        else:
            feat = model(view)

        features.append(F.normalize(feat.float(), p=2, dim=1))

    merged_feat = torch.stack(features, dim=0).mean(dim=0)
    merged_feat = F.normalize(merged_feat, p=2, dim=1)

    if return_class_logits and class_logits:
        merged_logits = torch.stack(class_logits, dim=0).mean(dim=0)
    else:
        merged_logits = None

    return merged_feat, merged_logits
