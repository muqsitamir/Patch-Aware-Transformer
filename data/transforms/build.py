import random
import torchvision.transforms as T

from .transforms import *
from .autoaugment import AutoAugment
from PIL import Image, ImageFilter, ImageOps

from .transforms import LGT

RESIZE_INTERPOLATION = 3
INTERPOLATION_NAMES = {
    0: "nearest",
    2: "bilinear",
    3: "bicubic",
}


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def _as_list(values):
    return list(values) if isinstance(values, (list, tuple)) else [values]


def _format_toggle(enabled, details):
    if enabled:
        return "on ({})".format(details)
    return "off"


def _resize_description(size):
    return "size={} interpolation={}".format(_as_list(size), INTERPOLATION_NAMES.get(RESIZE_INTERPOLATION, RESIZE_INTERPOLATION))


def _synthetic_flag(cfg):
    meta_cfg = getattr(cfg, "META", None)
    data_cfg = getattr(meta_cfg, "DATA", None)
    return getattr(data_cfg, "SYNTH_FLAG", "unknown")


def describe_train_transforms(cfg, is_fake=False):
    lines = [
        "Train Transform Config:",
        "  resize: {}".format(_resize_description(cfg.INPUT.SIZE_TRAIN)),
        "  horizontal_flip: {}".format(_format_toggle(cfg.INPUT.DO_FLIP, "p={}".format(cfg.INPUT.FLIP_PROB))),
        "  pad_crop: {}".format(_format_toggle(cfg.INPUT.DO_PAD, "padding={} mode={}".format(cfg.INPUT.PADDING, cfg.INPUT.PADDING_MODE))),
        "  local_grayscale_transform: enabled={} p={}".format(bool(cfg.INPUT.LGT.DO_LGT), cfg.INPUT.LGT.PROB),
        "  color_jitter: {}".format(_format_toggle(cfg.INPUT.CJ.ENABLED, "p={} brightness={} contrast={} saturation={} hue={}".format(
            cfg.INPUT.CJ.PROB,
            cfg.INPUT.CJ.BRIGHTNESS,
            cfg.INPUT.CJ.CONTRAST,
            cfg.INPUT.CJ.SATURATION,
            cfg.INPUT.CJ.HUE,
        ))),
        "  autoaugment: {}".format(_format_toggle(cfg.INPUT.DO_AUTOAUG, "schedule_steps={}".format(cfg.SOLVER.MAX_EPOCHS))),
        "  augmix: {}".format(_format_toggle(cfg.INPUT.DO_AUGMIX, "default_params")),
        "  random_patch: {}".format(_format_toggle(cfg.INPUT.RPT.ENABLED, "p={}".format(cfg.INPUT.RPT.PROB))),
        "  random_erasing: {}".format(_format_toggle(cfg.INPUT.REA.ENABLED, "p={} mode=pixel max_count=1 device=cpu".format(cfg.INPUT.REA.PROB))),
        "  synthetic_extra: {}".format("on ({})".format(_synthetic_flag(cfg)) if is_fake else "off (is_fake=False)"),
        "  to_tensor: on",
        "  normalize: mean={} std={}".format(_as_list(cfg.INPUT.PIXEL_MEAN), _as_list(cfg.INPUT.PIXEL_STD)),
    ]
    return "\n".join(lines)


def describe_test_transforms(cfg):
    lines = [
        "Test Transform Config:",
        "  resize: {}".format(_resize_description(cfg.INPUT.SIZE_TEST)),
        "  local_grayscale_transform: enabled=False p={} (not used in test pipeline)".format(cfg.INPUT.LGT.PROB),
        "  to_tensor: on",
        "  normalize: mean={} std={}".format(_as_list(cfg.INPUT.PIXEL_MEAN), _as_list(cfg.INPUT.PIXEL_STD)),
        "  tta: {}".format(_format_toggle(
            getattr(cfg.TEST, "TTA_ENABLED", False),
            "flip={} scales={}".format(getattr(cfg.TEST, "TTA_FLIP", True), _as_list(getattr(cfg.TEST, "TTA_SCALES", [1.0]))),
        )),
    ]
    return "\n".join(lines)

def build_transforms(cfg, is_train=True, is_fake=False):
    res = []
    pixel_mean = _as_list(cfg.INPUT.PIXEL_MEAN)
    pixel_std = _as_list(cfg.INPUT.PIXEL_STD)

    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN

        # augmix augmentation
        do_augmix = cfg.INPUT.DO_AUGMIX

        # auto augmentation
        do_autoaug = cfg.INPUT.DO_AUTOAUG
        # total_iter = cfg.SOLVER.MAX_ITER
        total_iter = cfg.SOLVER.MAX_EPOCHS

        # horizontal filp
        do_flip = cfg.INPUT.DO_FLIP
        flip_prob = cfg.INPUT.FLIP_PROB

        # padding
        do_pad = cfg.INPUT.DO_PAD
        padding = cfg.INPUT.PADDING
        padding_mode = cfg.INPUT.PADDING_MODE

        # Local Grayscale Transfomation
        do_lgt = cfg.INPUT.LGT.DO_LGT
        lgt_prob = cfg.INPUT.LGT.PROB

        # color jitter
        do_cj = cfg.INPUT.CJ.ENABLED
        cj_prob = cfg.INPUT.CJ.PROB
        cj_brightness = cfg.INPUT.CJ.BRIGHTNESS
        cj_contrast = cfg.INPUT.CJ.CONTRAST
        cj_saturation = cfg.INPUT.CJ.SATURATION
        cj_hue = cfg.INPUT.CJ.HUE

        # random erasing
        do_rea = cfg.INPUT.REA.ENABLED
        rea_prob = cfg.INPUT.REA.PROB
        rea_mean = cfg.INPUT.REA.MEAN
        # random patch
        do_rpt = cfg.INPUT.RPT.ENABLED
        rpt_prob = cfg.INPUT.RPT.PROB

        if do_autoaug:
            res.append(AutoAugment(total_iter))
        res.append(T.Resize(size_train, interpolation=RESIZE_INTERPOLATION))
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode),
                        T.RandomCrop(size_train)])
        if do_lgt:
            res.append(LGT(lgt_prob))
        if do_cj:
            res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
        if do_augmix:
            res.append(AugMix())
        # if do_rea:
        #     res.append(RandomErasing(probability=rea_prob, mean=rea_mean, sh=1/3))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))
        if is_fake:
            synth_flag = _synthetic_flag(cfg)
            if synth_flag == 'jitter':
                res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=1.0))
            elif synth_flag == 'augmix':
                res.append(AugMix())
            elif synth_flag == 'both':
                res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
                res.append(AugMix())
        res.extend([
            T.ToTensor(),
            T.Normalize(pixel_mean, pixel_std)
        ])
        if do_rea:
            from timm.data.random_erasing import RandomErasing as RE
            res.append(RE(probability=rea_prob, mode='pixel', max_count=1, device='cpu'))
    else:
        size_test = cfg.INPUT.SIZE_TEST
        res.append(T.Resize(size_test, interpolation=RESIZE_INTERPOLATION))
        res.extend([
            T.ToTensor(),
            T.Normalize(pixel_mean, pixel_std)
        ])
    return T.Compose(res)
