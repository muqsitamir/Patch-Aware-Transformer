import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .ce_labelSmooth import CrossEntropyLabelSmooth as CE_LS

PART_ATTENTION_MODEL_NAMES = {'part_attention_vit', 'local_attention_vit'}

feat_dim_dict = {
    'local_attention_vit': 768,
    'part_attention_vit': 768,
    'vit': 768,
    'resnet18': 512,
    'resnet34': 512
}


def _uses_part_attention_soft_labels(cfg, model_name):
    return model_name in PART_ATTENTION_MODEL_NAMES and cfg.MODEL.PC_LOSS


def build_loss(cfg, num_classes):
    fg_attn = getattr(cfg.MODEL, 'FOREGROUND_ATTN', None)
    if getattr(fg_attn, 'ENABLED', False) and cfg.MODEL.PC_LOSS:
        import logging
        logging.getLogger('PAT.train').warning(
            'MODEL.FOREGROUND_ATTN.ENABLED=True is incompatible with '
            'MODEL.PC_LOSS=True (part-classification assumes fixed stripe '
            'parts, which foreground attention replaces). Set PC_LOSS=False.'
        )

    name = cfg.MODEL.NAME
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.NAME not in feat_dim_dict.keys():
        feat_dim = 2048
    else:
        feat_dim = feat_dim_dict[cfg.MODEL.NAME]
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        if _uses_part_attention_soft_labels(cfg, name):
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        else:
            xent = CE_LS(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax': # softmax loss only
        def loss_func(score, feat, target):
            loss = F.cross_entropy(score, target)
            loss_func.last_components = {
                'id_loss': loss.detach(),
                'combined': loss.detach(),
            }
            return loss

    # softmax & triplet
    elif sampler in ('softmax_triplet', 'GS'):
        def loss_func(score, feat, target, domains=None, t_domains=None, all_posvid=None, soft_label=False, soft_weight=0.1, soft_lambda=0.2):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if _uses_part_attention_soft_labels(cfg, name):
                        ID_LOSS = xent(score, target, all_posvid=all_posvid, soft_label=soft_label,soft_weight=soft_weight, soft_lambda=soft_lambda)
                    else:
                        ID_LOSS = xent(score, target)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

                TRI_LOSS = triplet(feat, target)[0]
                # DOMAIN_LOSS = xent(domains, t_domains)
                loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                loss_func.last_components = {
                    'id_loss': ID_LOSS.detach(),
                    'tri_loss': TRI_LOSS.detach(),
                    'combined': loss.detach(),
                }
                return loss
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    id_loss = xent(score, target)
                else:
                    id_loss = F.cross_entropy(score, target)
                tri_loss = triplet(feat, target)[0]
                center_loss = cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                loss = id_loss + tri_loss + center_loss
                loss_func.last_components = {
                    'id_loss': id_loss.detach(),
                    'tri_loss': tri_loss.detach(),
                    'center_loss': center_loss.detach(),
                    'combined': loss.detach(),
                }
                return loss
            else:
                print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                    'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    loss_func.last_components = {}
    return loss_func, center_criterion
