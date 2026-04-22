import logging
import os
import random
# from threading import local
from model.backbones.vit_pytorch import deit_tiny_patch16_224_TransReID, part_attention_deit_small, part_attention_deit_tiny, part_attention_vit_base, part_attention_vit_base_p32, part_attention_vit_large, part_attention_vit_small, vit_base_patch32_224_TransReID, vit_large_patch16_224_TransReID
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.class_aware import is_class_aware_enabled, is_class_token_select_enabled

from .backbones.resnet import BasicBlock, ResNet, Bottleneck
from .backbones import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID

# alter this to your pre-trained file name
lup_path_name = {
    'vit_base_patch16_224_TransReID': 'vit_base_ics_cfs_lup.pth',
    'vit_small_patch16_224_TransReID': 'vit_base_ics_cfs_lup.pth',
}

# alter this to your pre-trained file name
imagenet_path_name = {
    'vit_large_patch16_224_TransReID': 'jx_vit_large_p16_224-4ee7a4dc.pth',
    'vit_base_patch16_224_TransReID': 'jx_vit_base_p16_224-80ecf9dd.pth',
    'vit_base_patch32_224_TransReID': 'jx_vit_base_patch32_224_in21k-8db57226.pth', 
    'deit_base_patch16_224_TransReID': 'deit_base_distilled_patch16_224-df68dfff.pth',
    'vit_small_patch16_224_TransReID': 'vit_small_p16_224-15ec54c9.pth',
    'deit_small_patch16_224_TransReID': 'deit_small_distilled_patch16_224-649709d9.pth',
    'deit_tiny_patch16_224_TransReID': 'deit_tiny_distilled_patch16_224-b40b3cf7.pth'
}

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ClassGuidedTokenSelector(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes,
        topk=16,
        fusion='add',
        beta=0.5,
        score_norm='softmax',
        mode='relevance',
        deviation_metric='cosine',
        relevance_weight=1.0,
        deviation_weight=1.0,
        relevance_positive_norm='softmax',
        deviation_norm='none',
        stats_log_period=100,
    ):
        super().__init__()
        self.in_planes = int(in_planes)
        self.num_classes = int(num_classes)
        self.topk = int(topk)
        self.fusion = str(fusion).lower()
        self.beta = float(beta)
        self.score_norm = str(score_norm).lower()
        self.mode = str(mode).lower()
        self.deviation_metric = str(deviation_metric).lower()
        self.relevance_weight = float(relevance_weight)
        self.deviation_weight = float(deviation_weight)
        self.relevance_positive_norm = str(relevance_positive_norm).lower()
        self.deviation_norm = str(deviation_norm).lower()
        if self.deviation_norm == 'centered':
            self.deviation_norm = 'center'
        self.stats_log_period = int(stats_log_period)
        self._stats_log_count = 0
        self._debug_logged = False
        self._fallback_logged = False
        self._invalid_logged = False

        if self.num_classes <= 0:
            raise ValueError("ClassGuidedTokenSelector requires num_classes > 0")
        if self.fusion not in ('add', 'concat'):
            raise ValueError("MODEL.CLASS_TOKEN_SELECT.FUSION must be 'add' or 'concat'")
        if self.score_norm not in ('softmax', 'sigmoid', 'none'):
            raise ValueError("MODEL.CLASS_TOKEN_SELECT.SCORE_NORM must be 'softmax', 'sigmoid', or 'none'")
        if self.mode not in ('relevance', 'deviation', 'relevance_x_deviation'):
            raise ValueError("MODEL.CLASS_TOKEN_SELECT.MODE must be 'relevance', 'deviation', or 'relevance_x_deviation'")
        if self.deviation_metric not in ('cosine', 'l2'):
            raise ValueError("MODEL.CLASS_TOKEN_SELECT.DEVIATION_METRIC must be 'cosine' or 'l2'")
        if self.relevance_positive_norm not in ('softmax', 'sigmoid'):
            raise ValueError("MODEL.CLASS_TOKEN_SELECT.RELEVANCE_POSITIVE_NORM must be 'softmax' or 'sigmoid'")
        if self.deviation_norm not in ('none', 'center', 'zscore'):
            raise ValueError("MODEL.CLASS_TOKEN_SELECT.DEVIATION_NORM must be 'none', 'center', or 'zscore'")
        if self.stats_log_period < 0:
            raise ValueError("MODEL.CLASS_TOKEN_SELECT.STATS_LOG_PERIOD must be >= 0")

        self.class_embed = nn.Embedding(self.num_classes, self.in_planes)
        self.class_prototype = nn.Embedding(self.num_classes, self.in_planes)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.class_prototype.weight, std=0.02)
        if self.fusion == 'concat':
            self.fusion_proj = nn.Linear(self.in_planes * 2, self.in_planes)
            self.fusion_proj.apply(weights_init_kaiming)
        else:
            self.fusion_proj = None

    def _log_once(self, attr_name, message):
        if getattr(self, attr_name):
            return
        setattr(self, attr_name, True)
        logging.getLogger("PAT.train").info(message)
        print(message)

    @staticmethod
    def _score_stats(scores):
        scores = scores.detach().float()
        return (
            scores.mean().item(),
            scores.min().item(),
            scores.max().item(),
            scores.std(unbiased=False).item(),
        )

    def _positive_relevance_weights(self, relevance_scores):
        if self.relevance_positive_norm == 'softmax':
            return F.softmax(relevance_scores, dim=1)
        return torch.sigmoid(relevance_scores)

    def _normalize_deviation(self, deviation_scores):
        if self.deviation_norm == 'none':
            return deviation_scores

        centered = deviation_scores - deviation_scores.mean(dim=1, keepdim=True)
        if self.deviation_norm == 'center':
            return centered

        std = deviation_scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return centered / std

    def forward(self, global_token, patch_tokens, class_labels=None):
        if class_labels is None:
            self._log_once(
                '_fallback_logged',
                'Class-token selection fallback: class labels are missing; using global tokens.'
            )
            return global_token
        if patch_tokens is None or patch_tokens.dim() != 3 or patch_tokens.size(1) == 0:
            self._log_once(
                '_fallback_logged',
                'Class-token selection fallback: patch tokens are unavailable; using global tokens.'
            )
            return global_token
        if global_token.dim() != 2 or patch_tokens.size(0) != global_token.size(0) or patch_tokens.size(2) != global_token.size(1):
            self._log_once(
                '_fallback_logged',
                'Class-token selection fallback: token shapes do not match; using global tokens.'
            )
            return global_token

        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=global_token.device)
        class_labels = class_labels.to(global_token.device).long().view(-1)
        if class_labels.numel() != global_token.size(0):
            self._log_once(
                '_fallback_logged',
                'Class-token selection fallback: class-label batch size does not match tokens; using global tokens.'
            )
            return global_token

        valid = (class_labels >= 0) & (class_labels < self.num_classes)
        if not torch.any(valid):
            self._log_once(
                '_invalid_logged',
                'Class-token selection fallback: all class labels are invalid; using global tokens.'
            )
            return global_token

        k = min(max(self.topk, 1), patch_tokens.size(1))
        safe_labels = class_labels.clamp(0, self.num_classes - 1)
        queries = self.class_embed(safe_labels).to(dtype=patch_tokens.dtype)
        prototypes = self.class_prototype(safe_labels).to(dtype=patch_tokens.dtype)
        query_norm = F.normalize(queries, dim=-1)
        patch_norm = F.normalize(patch_tokens, dim=-1)
        prototype_norm = F.normalize(prototypes, dim=-1)
        relevance_scores = torch.einsum('bnc,bc->bn', patch_norm, query_norm)
        prototype_similarity = torch.einsum('bnc,bc->bn', patch_norm, prototype_norm)
        if self.deviation_metric == 'cosine':
            deviation_scores = 1.0 - prototype_similarity
        else:
            deviation_scores = torch.sum((patch_tokens - prototypes.unsqueeze(1)).pow(2), dim=-1)
        normalized_deviation_scores = self._normalize_deviation(deviation_scores)

        weighted_relevance = self.relevance_weight * relevance_scores
        positive_relevance = self._positive_relevance_weights(weighted_relevance)
        weighted_deviation = self.deviation_weight * normalized_deviation_scores
        if self.mode == 'relevance':
            scores = weighted_relevance
        elif self.mode == 'deviation':
            scores = weighted_deviation
        else:
            scores = positive_relevance * weighted_deviation

        top_scores, top_indices = torch.topk(scores, k=k, dim=1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1))
        selected_tokens = torch.gather(patch_tokens, 1, gather_index)

        if self.score_norm == 'softmax':
            weights = F.softmax(top_scores, dim=1).unsqueeze(-1).to(dtype=selected_tokens.dtype)
            selected_local = torch.sum(selected_tokens * weights, dim=1)
        elif self.score_norm == 'sigmoid':
            weights = torch.sigmoid(top_scores).unsqueeze(-1).to(dtype=selected_tokens.dtype)
            selected_local = torch.sum(selected_tokens * weights, dim=1)
            selected_local = selected_local / weights.sum(dim=1).clamp_min(1e-6)
        else:
            selected_local = selected_tokens.mean(dim=1)

        selected_local = selected_local.to(dtype=global_token.dtype)
        if self.fusion == 'add':
            fused_valid = global_token + self.beta * selected_local
        else:
            fused_valid = self.fusion_proj(torch.cat([global_token, self.beta * selected_local], dim=1))
        fused = torch.where(valid.unsqueeze(1), fused_valid, global_token)

        if not self._debug_logged:
            self._debug_logged = True
            message = (
                'Class-token selection debug: mode={}, deviation_metric={}, deviation_norm={}, '
                'relevance_positive_norm={}, global={}, patches={}, relevance_scores={}, '
                'deviation_scores={}, final_scores={}, selected={}, topk={}, valid={}/{}'
            ).format(
                self.mode,
                self.deviation_metric,
                self.deviation_norm,
                self.relevance_positive_norm,
                tuple(global_token.shape),
                tuple(patch_tokens.shape),
                tuple(relevance_scores.shape),
                tuple(deviation_scores.shape),
                tuple(scores.shape),
                tuple(selected_tokens.shape),
                k,
                int(valid.sum().item()),
                int(valid.numel()),
            )
            logging.getLogger("PAT.train").info(message)
            print(message)
        self._stats_log_count += 1
        if self.stats_log_period > 0 and self._stats_log_count % self.stats_log_period == 0:
            valid_relevance = relevance_scores[valid]
            valid_positive_relevance = positive_relevance[valid]
            valid_deviation = normalized_deviation_scores[valid]
            valid_scores = scores[valid]
            rel_mean, rel_min, rel_max, rel_std = self._score_stats(valid_relevance)
            pos_rel_mean, pos_rel_min, pos_rel_max, pos_rel_std = self._score_stats(valid_positive_relevance)
            dev_mean, dev_min, dev_max, dev_std = self._score_stats(valid_deviation)
            score_mean, score_min, score_max, score_std = self._score_stats(valid_scores)
            logging.getLogger("PAT.train").info(
                "Class-token selection stats[%d]: mode=%s, deviation_metric=%s, deviation_norm=%s, "
                "relevance_positive_norm=%s, topk=%d, "
                "relevance(mean=%.4f,min=%.4f,max=%.4f,std=%.4f), "
                "positive_relevance(mean=%.4f,min=%.4f,max=%.4f,std=%.4f), "
                "deviation(mean=%.4f,min=%.4f,max=%.4f,std=%.4f), "
                "final_score(mean=%.4f,min=%.4f,max=%.4f,std=%.4f)",
                self._stats_log_count,
                self.mode,
                self.deviation_metric,
                self.deviation_norm,
                self.relevance_positive_norm,
                k,
                rel_mean,
                rel_min,
                rel_max,
                rel_std,
                pos_rel_mean,
                pos_rel_min,
                pos_rel_max,
                pos_rel_std,
                dev_mean,
                dev_min,
                dev_max,
                dev_std,
                score_mean,
                score_min,
                score_max,
                score_std,
            )
        return fused


def _make_class_token_selector(cfg, in_planes, num_semantic_classes, owner_name, supports_tokens=True):
    requested = is_class_token_select_enabled(cfg)
    if not requested:
        print('Class-token selection disabled for {}.'.format(owner_name))
        return None, False
    if not supports_tokens:
        print('Class-token selection requested for {}, but this backbone does not expose patch tokens; disabled.'.format(owner_name))
        return None, False

    select_cfg = cfg.MODEL.CLASS_TOKEN_SELECT
    configured_classes = int(getattr(select_cfg, 'NUM_CLASSES', 0))
    num_classes = max(int(num_semantic_classes), configured_classes)
    if num_classes <= 0:
        print('Class-token selection requested for {}, but NUM_CLASSES is 0; disabled.'.format(owner_name))
        return None, False

    selector = ClassGuidedTokenSelector(
        in_planes=in_planes,
        num_classes=num_classes,
        topk=int(getattr(select_cfg, 'TOPK', 16)),
        fusion=str(getattr(select_cfg, 'FUSION', 'add')),
        beta=float(getattr(select_cfg, 'BETA', 0.5)),
        score_norm=str(getattr(select_cfg, 'SCORE_NORM', 'softmax')),
        mode=str(getattr(select_cfg, 'MODE', 'relevance')),
        deviation_metric=str(getattr(select_cfg, 'DEVIATION_METRIC', 'cosine')),
        relevance_weight=float(getattr(select_cfg, 'RELEVANCE_WEIGHT', 1.0)),
        deviation_weight=float(getattr(select_cfg, 'DEVIATION_WEIGHT', 1.0)),
        relevance_positive_norm=str(getattr(select_cfg, 'RELEVANCE_POSITIVE_NORM', 'softmax')),
        deviation_norm=str(getattr(select_cfg, 'DEVIATION_NORM', 'none')),
        stats_log_period=int(getattr(select_cfg, 'STATS_LOG_PERIOD', 100)),
    )
    print(
        'Class-token selection enabled for {}: num_classes={}, topk={}, fusion={}, beta={}, '
        'score_norm={}, mode={}, deviation_metric={}, deviation_norm={}, '
        'relevance_weight={}, deviation_weight={}, relevance_positive_norm={}, stats_log_period={}'.format(
            owner_name,
            num_classes,
            selector.topk,
            selector.fusion,
            selector.beta,
            selector.score_norm,
            selector.mode,
            selector.deviation_metric,
            selector.deviation_norm,
            selector.relevance_weight,
            selector.deviation_weight,
            selector.relevance_positive_norm,
            selector.stats_log_period,
        )
    )
    return selector, True


def _load_finetune_params(model, model_path):
    try:
        param_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    except TypeError:
        param_dict = torch.load(model_path, map_location='cpu')
    if 'state_dict' in param_dict:
        param_dict = param_dict['state_dict']

    model_dict = model.state_dict()
    loaded_keys = []
    skipped_classifier_keys = []
    skipped_keys = []
    for i in param_dict:
        key = i.replace('module.', '')
        if key not in model_dict:
            skipped_keys.append(key)
            continue
        if model_dict[key].shape != param_dict[i].shape:
            if key.startswith('classifier.'):
                skipped_classifier_keys.append(key)
            else:
                skipped_keys.append(key)
            continue
        model_dict[key].copy_(param_dict[i])
        loaded_keys.append(key)

    print('Loading pretrained model for finetuning from {}'.format(model_path))
    print('Loaded {} matched parameter tensors; skipped {} tensors.'.format(len(loaded_keys), len(skipped_keys) + len(skipped_classifier_keys)))
    if skipped_classifier_keys:
        print('Reset ID classifier for finetuning due to shape mismatch: {}'.format(', '.join(skipped_classifier_keys)))


class Backbone(nn.Module):
    def __init__(self, model_name, num_classes, cfg, num_semantic_classes=0):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        # model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 2048
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
            model_path = os.path.join(model_path_base, \
                "resnet18-f37072fd.pth")
            print('using resnet18 as a backbone')
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet34-b627a593.pth")
            print('using resnet34 as a backbone')
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet50-0676ba61.pth")
            print('using resnet50 as a backbone')
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
            model_path = os.path.join(model_path_base, \
                "resnet101-63fe2227.pth")
            print('using resnet101 as a backbone')
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            model_path = os.path.join(model_path_base, \
                "resnet152-394f9c45.pth")
            print('using resnet152 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # self.pool = nn.Linear(in_features=16*8, out_features=1, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.num_semantic_classes = int(num_semantic_classes)
        self.class_aware_enabled = is_class_aware_enabled(cfg) and self.num_semantic_classes > 0
        self.class_token_selector, self.class_token_select_enabled = _make_class_token_selector(
            cfg, self.in_planes, self.num_semantic_classes, model_name, supports_tokens=False
        )

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        if self.class_aware_enabled:
            self.semantic_head = nn.Linear(self.in_planes, self.num_semantic_classes, bias=False)
            self.semantic_head.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, return_class_logits=False, class_labels=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x) # B, C, h, w
        
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # global_feat = self.pool(x.flatten(2)).squeeze() # is GAP harming generalization?

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            if return_class_logits and self.class_aware_enabled:
                return cls_score, global_feat, self.semantic_head(feat)
            return cls_score, global_feat
        else:
            output_feat = feat if self.neck_feat == 'after' else global_feat
            if return_class_logits and self.class_aware_enabled:
                return output_feat, self.semantic_head(feat)
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        model_dict = self.state_dict()
        loaded_semantic_head = not self.class_aware_enabled
        for i in param_dict:
            key = i.replace('module.', '')
            if 'classifier' in key: # drop classifier
                continue
            if key not in model_dict or model_dict[key].shape != param_dict[i].shape:
                continue
            model_dict[key].copy_(param_dict[i])
            if key.startswith('semantic_head.'):
                loaded_semantic_head = True
        if self.class_aware_enabled and not loaded_semantic_head:
            print('Semantic head not found in checkpoint; disabling class-aware inference for this model.')
            self.class_aware_enabled = False
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        _load_finetune_params(self, model_path)

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))


class build_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory, num_semantic_classes=0):
        super(build_vit, self).__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        self.model_path = os.path.join(model_path_base, path)
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.num_semantic_classes = int(num_semantic_classes)
        self.class_aware_enabled = is_class_aware_enabled(cfg) and self.num_semantic_classes > 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
            (img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate= cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
            self.in_planes = 192
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
            self.in_planes = 1024
        self.class_token_selector, self.class_token_select_enabled = _make_class_token_selector(
            cfg, self.in_planes, self.num_semantic_classes, 'vit'
        )
        if self.pretrain_choice == 'imagenet':
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
            
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        if self.class_aware_enabled:
            self.semantic_head = nn.Linear(self.in_planes, self.num_semantic_classes, bias=False)
            self.semantic_head.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, return_class_logits=False, class_labels=None):
        x = self.base(x) # B, N, C
        global_feat = x[:, 0] # cls token for global feature
        if self.class_token_select_enabled:
            patch_tokens = x[:, 1:]
            global_feat = self.class_token_selector(global_feat, patch_tokens, class_labels)

        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            if return_class_logits and self.class_aware_enabled:
                return cls_score, global_feat, self.semantic_head(feat)
            return cls_score, global_feat
        else:
            output_feat = feat if self.neck_feat == 'after' else global_feat
            if return_class_logits and self.class_aware_enabled:
                return output_feat, self.semantic_head(feat)
            return output_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        model_dict = self.state_dict()
        loaded_semantic_head = not self.class_aware_enabled
        for i in param_dict:
            key = i.replace('module.', '')
            if 'classifier' in key: # drop classifier
                continue
            if 'bottleneck' in key:
                continue
            if key not in model_dict or model_dict[key].shape != param_dict[i].shape:
                continue
            model_dict[key].copy_(param_dict[i])
            if key.startswith('semantic_head.'):
                loaded_semantic_head = True
        if self.class_aware_enabled and not loaded_semantic_head:
            print('Semantic head not found in checkpoint; disabling class-aware inference for this model.')
            self.class_aware_enabled = False
        print('Loading trained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        _load_finetune_params(self, model_path)

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

'''
part attention vit
'''
class build_part_attention_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory, pretrain_tag='imagenet', num_semantic_classes=0):
        super().__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        if pretrain_tag == 'lup':
            path = lup_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        else:
            path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        self.model_path = os.path.join(model_path_base, path)
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: part token vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.num_semantic_classes = int(num_semantic_classes)
        self.class_aware_enabled = is_class_aware_enabled(cfg) and self.num_semantic_classes > 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
            (img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate= cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            pretrain_tag=pretrain_tag)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
            self.in_planes = 192
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
            self.in_planes = 1024
        self.class_token_selector, self.class_token_select_enabled = _make_class_token_selector(
            cfg, self.in_planes, self.num_semantic_classes, 'part_attention_vit'
        )
        if self.pretrain_choice == 'imagenet':
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        if self.class_aware_enabled:
            self.semantic_head = nn.Linear(self.in_planes, self.num_semantic_classes, bias=False)
            self.semantic_head.apply(weights_init_classifier)

    def forward(self, x, return_class_logits=False, class_labels=None):
        layerwise_tokens = self.base(x) # B, N, C
        layerwise_cls_tokens = [t[:, 0] for t in layerwise_tokens] # cls token
        part_feat_list = layerwise_tokens[-1][:, 1: 4] # 3, 768

        layerwise_part_tokens = [[t[:, i] for i in range(1,4)] for t in layerwise_tokens] # 12 3 768
        if self.class_token_select_enabled:
            patch_tokens = layerwise_tokens[-1][:, 4:]
            layerwise_cls_tokens[-1] = self.class_token_selector(layerwise_cls_tokens[-1], patch_tokens, class_labels)
        feat = self.bottleneck(layerwise_cls_tokens[-1])

        if self.training:
            cls_score = self.classifier(feat)
            if return_class_logits and self.class_aware_enabled:
                return cls_score, layerwise_cls_tokens, layerwise_part_tokens, self.semantic_head(feat)
            return cls_score, layerwise_cls_tokens, layerwise_part_tokens
        else:
            output_feat = feat if self.neck_feat == 'after' else layerwise_cls_tokens[-1]
            if return_class_logits and self.class_aware_enabled:
                return output_feat, self.semantic_head(feat)
            return output_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        model_dict = self.state_dict()
        loaded_semantic_head = not self.class_aware_enabled
        for i in param_dict:
            key = i.replace('module.', '')
            if 'classifier' in key: # drop classifier
                continue
            if key not in model_dict or model_dict[key].shape != param_dict[i].shape:
                continue
            model_dict[key].copy_(param_dict[i])
            if key.startswith('semantic_head.'):
                loaded_semantic_head = True
        if self.class_aware_enabled and not loaded_semantic_head:
            print('Semantic head not found in checkpoint; disabling class-aware inference for this model.')
            self.class_aware_enabled = False
        print('Loading trained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        _load_finetune_params(self, model_path)

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))        

__factory_T_type = {
    'vit_large_patch16_224_TransReID': vit_large_patch16_224_TransReID,
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_base_patch32_224_TransReID': vit_base_patch32_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    'deit_tiny_patch16_224_TransReID': deit_tiny_patch16_224_TransReID,
}

__factory_LAT_type = {
    'vit_large_patch16_224_TransReID': part_attention_vit_large,
    'vit_base_patch16_224_TransReID': part_attention_vit_base,
    'vit_base_patch32_224_TransReID': part_attention_vit_base_p32,
    'deit_base_patch16_224_TransReID': part_attention_vit_base,
    'vit_small_patch16_224_TransReID': part_attention_vit_small,
    'deit_small_patch16_224_TransReID': part_attention_deit_small,
    'deit_tiny_patch16_224_TransReID': part_attention_deit_tiny,
}

def make_model(cfg, modelname, num_class, sd_flag=False, head_flag=False, camera_num=None, view_num=None, num_semantic_class=0):
    if modelname == 'vit':
        model = build_vit(num_class, cfg, __factory_T_type, num_semantic_classes=num_semantic_class)
        print('===========building vit===========')
    elif modelname == 'part_attention_vit':
        model = build_part_attention_vit(num_class, cfg, __factory_LAT_type, num_semantic_classes=num_semantic_class)
        print('===========building our part attention vit===========')
    else:
        model = Backbone(modelname, num_class, cfg, num_semantic_classes=num_semantic_class)
        print('===========building ResNet===========')
    ### count params
    model.compute_num_params()
    return model
