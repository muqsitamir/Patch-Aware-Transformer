import logging
import math
import os
import random
import time
import torch
import torch.nn as nn
from model.make_model import make_model
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter
from utils.eval_guard import (
    dummy_eval_warning,
    resolve_eval_dataset_name,
    should_skip_eval_if_dummy_ids,
)
from utils.class_aware import (
    get_batch_class_targets,
    get_class_distance_penalty,
    get_class_loss_weight,
    model_is_class_aware,
    model_num_semantic_classes,
    semantic_accuracy,
    semantic_classification_loss,
    use_metadata_classes_for_retrieval,
)
from utils.tta import extract_tta_features, log_tta_settings
from utils.group_rerank import save_metrics


def _test_names(cfg):
    names = getattr(cfg.DATASETS, "TEST", ())
    if isinstance(names, str):
        return (names,)
    return tuple(names)


def _log_validation_header(logger, label, dataset_name, epoch=None):
    if epoch is None:
        logger.info("{} validation dataset: {}".format(label, dataset_name))
    else:
        logger.info("{} validation dataset: {} - Epoch: {}".format(label, dataset_name, epoch))


def _iter_named_tensors(prefix, value):
    if value is None:
        return
    if torch.is_tensor(value):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = "{}.{}".format(prefix, key) if prefix else str(key)
            yield from _iter_named_tensors(child_prefix, item)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            child_prefix = "{}[{}]".format(prefix, idx) if prefix else "[{}]".format(idx)
            yield from _iter_named_tensors(child_prefix, item)


def _tensor_max_abs(tensor):
    if tensor.numel() == 0:
        return 0.0
    safe_tensor = torch.nan_to_num(tensor.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    return float(safe_tensor.abs().max().item())


def _tensor_stats(tensor):
    tensor = tensor.detach()
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "nan": int(torch.isnan(tensor).sum().item()),
        "inf": int(torch.isinf(tensor).sum().item()),
        "max_abs": _tensor_max_abs(tensor),
    }


def _find_nonfinite_tensors(items):
    problems = []
    for name, value in items:
        for tensor_name, tensor in _iter_named_tensors(name, value):
            if not torch.is_tensor(tensor) or not tensor.dtype.is_floating_point:
                continue
            if bool(torch.isfinite(tensor.detach()).all().item()):
                continue
            problems.append((tensor_name, _tensor_stats(tensor)))
    return problems


def _find_nonfinite_grads(model, limit=10):
    problems = []
    for name, param in model.named_parameters():
        if param.grad is None or not param.grad.dtype.is_floating_point:
            continue
        if bool(torch.isfinite(param.grad.detach()).all().item()):
            continue
        problems.append((name, _tensor_stats(param.grad)))
        if len(problems) >= limit:
            break
    return problems


def _log_nonfinite_state(logger, epoch, iteration, img_path, target, items, grad_problems=None, prefix="Non-finite training state"):
    problems = _find_nonfinite_tensors(items)
    if not problems and not grad_problems:
        return

    logger.error("{} at epoch {} iteration {}".format(prefix, epoch, iteration + 1))
    if img_path:
        logger.error("Batch sample image: {}".format(img_path[0]))
    if target is not None and torch.is_tensor(target) and target.numel() > 0:
        target_cpu = target.detach().cpu()
        logger.error(
            "Target stats: min={} max={} unique={}".format(
                int(target_cpu.min().item()),
                int(target_cpu.max().item()),
                int(target_cpu.unique().numel()),
            )
        )

    for name, stats in problems[:20]:
        logger.error(
            "Tensor {} shape={} dtype={} nan={} inf={} max_abs={:.6g}".format(
                name,
                stats["shape"],
                stats["dtype"],
                stats["nan"],
                stats["inf"],
                stats["max_abs"],
            )
        )

    for name, stats in (grad_problems or [])[:10]:
        logger.error(
            "Grad {} shape={} dtype={} nan={} inf={} max_abs={:.6g}".format(
                name,
                stats["shape"],
                stats["dtype"],
                stats["nan"],
                stats["inf"],
                stats["max_abs"],
            )
        )


def part_attention_vit_do_train_with_amp(cfg,
             model,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             num_query, local_rank,
             patch_centers = None,
             pc_criterion= None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    grad_clip_enabled = bool(getattr(cfg.SOLVER, "GRAD_CLIP_ENABLED", False))
    grad_clip_norm = float(getattr(cfg.SOLVER, "GRAD_CLIP_NORM", 0.0))

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("PAT.train")
    logger.info('start training')
    logger.info("Gradient clipping: enabled={} max_norm={:.3f}".format(grad_clip_enabled, grad_clip_norm))
    log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    best_checkpoint_path = os.path.join(log_path, cfg.MODEL.NAME + '_best.pth')
    tb_path = os.path.join(cfg.TB_LOG_ROOT, cfg.LOG_NAME)
    tbWriter = SummaryWriter(tb_path)
    print("saving tblog to {}".format(tb_path))
    
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    total_loss_meter = AverageMeter()
    reid_loss_meter = AverageMeter()
    class_loss_meter = AverageMeter()
    class_acc_meter = AverageMeter()
    pc_loss_meter = AverageMeter()
    # ds_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(init_scale=512)
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    # train
    if cfg.MODEL.PC_LOSS:
        print('initialize the centers')
        model.train()
        for i, informations in enumerate(train_loader):
            # measure data loading time
            with torch.no_grad():
                #input = input.cuda(non_blocking=True)
                input = informations['images'].cuda(non_blocking=True)
                vid = informations['targets']
                camid = informations['camid']
                path = informations['img_path']
                #input = input.view(-1, input.size(2), input.size(3), input.size(4))

                # compute output
                _, _, layerwise_feat_list = model(input)
                patch_centers.get_soft_label(path, layerwise_feat_list[-1], vid=vid, camid=camid)
        print('initialization done')
    
    best_mAP = 0.0
    best_index = None
    last_checkpoint_epoch = None
    val_name = resolve_eval_dataset_name(cfg)
    secondary_val_names = _test_names(cfg)[1:]
    nonfinite_reported = False
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        total_loss_meter.reset()
        reid_loss_meter.reset()
        class_loss_meter.reset()
        class_acc_meter.reset()
        acc_meter.reset()
        pc_loss_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, informations in enumerate(train_loader):
            img = informations['images']
            vid = informations['targets']
            camid = informations['camid']
            img_path = informations['img_path']
            t_domains = informations['others']['domains']

            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = camid.to(device)
            t_domains = t_domains.to(device)
            class_targets = get_batch_class_targets(informations, device)
            use_class_aware = model_is_class_aware(model)
            class_logits = None
            class_loss = None
            part_feat = None

            model.to(device)
            with amp.autocast(enabled=True):
                if use_class_aware:
                    score, layerwise_global_feat, layerwise_feat_list, class_logits = model(img, return_class_logits=True)
                else:
                    score, layerwise_global_feat, layerwise_feat_list = model(img)
                
                ############## patch learning ######################
                patch_agent, position = patch_centers.get_soft_label(img_path, layerwise_feat_list[-1], vid=vid, camid=camid)
                l_ploss = cfg.MODEL.PC_LR
                if cfg.MODEL.PC_LOSS:
                    feat = torch.stack(layerwise_feat_list[-1], dim=0)
                    feat = feat[:,::1,:]
                    part_feat = feat
                    '''
                    loss1: clustering loss(for patch centers)
                    '''
                    ploss, all_posvid = pc_criterion(feat, patch_agent, position, patch_centers, vid=target, camid=target_cam)
                    '''
                    loss2: reid-specific loss
                    (ID + Triplet loss)
                    '''
                    reid_loss = loss_fn(score, layerwise_global_feat[-1], target, all_posvid=all_posvid, soft_label=cfg.MODEL.SOFT_LABEL, soft_weight=cfg.MODEL.SOFT_WEIGHT, soft_lambda=cfg.MODEL.SOFT_LAMBDA)
                else:
                    ploss = torch.tensor([0.]).cuda()
                    reid_loss = loss_fn(score, layerwise_global_feat[-1], target, soft_label=cfg.MODEL.SOFT_LABEL)
                
                class_loss = semantic_classification_loss(class_logits, class_targets)
                total_loss = reid_loss + l_ploss*ploss
                if class_loss is not None:
                    total_loss = total_loss + get_class_loss_weight(cfg) * class_loss

            loss_components = dict(getattr(loss_fn, "last_components", {}))
            monitored_items = [
                ("score", score),
                ("global_feat", layerwise_global_feat[-1]),
                ("part_feat", part_feat),
                ("patch_agent", patch_agent),
                ("id_loss", loss_components.get("id_loss")),
                ("tri_loss", loss_components.get("tri_loss")),
                ("center_loss", loss_components.get("center_loss")),
                ("reid_loss", reid_loss),
                ("pc_loss", ploss),
                ("class_loss", class_loss),
                ("total_loss", total_loss),
            ]
            if _find_nonfinite_tensors(monitored_items):
                if not nonfinite_reported:
                    _log_nonfinite_state(
                        logger,
                        epoch,
                        n_iter,
                        img_path,
                        target,
                        monitored_items,
                        prefix="Non-finite forward/loss tensors detected",
                    )
                    nonfinite_reported = True
                logger.warning(
                    "Skipping optimizer step at epoch {} iteration {} because forward/loss tensors are non-finite.".format(
                        epoch, n_iter + 1
                    )
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = None
            if grad_clip_enabled and grad_clip_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            grad_norm_value = None
            if grad_norm is not None:
                grad_norm_value = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

            grad_problems = _find_nonfinite_grads(model)
            if grad_problems or (grad_norm_value is not None and not math.isfinite(grad_norm_value)):
                if not nonfinite_reported:
                    if grad_norm_value is not None and not math.isfinite(grad_norm_value):
                        logger.error("Gradient norm is non-finite: {}".format(grad_norm_value))
                    _log_nonfinite_state(
                        logger,
                        epoch,
                        n_iter,
                        img_path,
                        target,
                        monitored_items,
                        grad_problems=grad_problems,
                        prefix="Non-finite gradients detected",
                    )
                    nonfinite_reported = True
                logger.warning(
                    "Skipping optimizer step at epoch {} iteration {} because gradients are non-finite.".format(
                        epoch, n_iter + 1
                    )
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

            # score = scores[-1]
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            total_loss_meter.update(total_loss.item(), img.shape[0])
            reid_loss_meter.update(reid_loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            pc_loss_meter.update(ploss.item(), img.shape[0])
            if use_class_aware and class_logits is not None and class_loss is not None:
                class_loss_meter.update(class_loss.item(), img.shape[0])
                class_acc = semantic_accuracy(class_logits, class_targets)
                if class_acc is not None:
                    class_acc_meter.update(class_acc.item(), 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if use_class_aware:
                    logger.info("Epoch[{}] Iteration[{}/{}] total_loss: {:.3f}, reid_loss: {:.3f}, class_loss: {:.3f}, pc_loss: {:.3f}, Acc: {:.3f}, Class Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, n_iter+1, len(train_loader), total_loss_meter.avg,
                    reid_loss_meter.avg, class_loss_meter.avg, pc_loss_meter.avg, acc_meter.avg, class_acc_meter.avg, scheduler._get_lr(epoch)[0]))
                else:
                    logger.info("Epoch[{}] Iteration[{}/{}] total_loss: {:.3f}, reid_loss: {:.3f}, pc_loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, n_iter+1, len(train_loader), total_loss_meter.avg,
                    reid_loss_meter.avg, pc_loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                tbWriter.add_scalar('train/reid_loss', reid_loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar('train/acc', acc_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar("train/pc_loss", pc_loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                if use_class_aware:
                    tbWriter.add_scalar('train/class_loss', class_loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                    tbWriter.add_scalar('train/class_acc', class_acc_meter.avg, n_iter+1+(epoch-1)*len(train_loader))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
        
        if epoch % eval_period == 0:
            if should_skip_eval_if_dummy_ids(cfg, val_loader, num_query, val_name):
                if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                    logger.warning(dummy_eval_warning(val_name))
            elif cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, informations in enumerate(val_loader):
                        with torch.no_grad():
                            img = informations['images'].to(device)
                            vid = informations['targets']
                            camid = informations['camid']
                            feat = model(img)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    if best_index is None or best_mAP < mAP:
                        best_mAP = mAP
                        best_index = epoch
                        torch.save(model.state_dict(), best_checkpoint_path)
                        logger.info("=====best epoch: {} mAP: {:.1%}; saved {}=====".format(best_index, best_mAP, best_checkpoint_path))
                    torch.cuda.empty_cache()
            else:
                _log_validation_header(logger, "Primary", val_name, epoch)
                cmc, mAP = do_inference(cfg, model, val_loader, num_query, dataset_name=val_name)
                if cmc is not None and mAP is not None:
                    tbWriter.add_scalar('val/Rank@1', cmc[0], epoch)
                    tbWriter.add_scalar('val/mAP', mAP, epoch)
                    if best_index is None or best_mAP < mAP:
                        best_mAP = mAP
                        best_index = epoch
                        torch.save(model.state_dict(), best_checkpoint_path)
                        logger.info("=====best epoch: {} mAP: {:.1%}; saved {}=====".format(best_index, best_mAP, best_checkpoint_path))
                for secondary_name in secondary_val_names:
                    _log_validation_header(logger, "Secondary", secondary_name, epoch)
                    secondary_loader, secondary_num_query = build_reid_test_loader(cfg, secondary_name)
                    secondary_cmc, secondary_mAP = do_inference(
                        cfg,
                        model,
                        secondary_loader,
                        secondary_num_query,
                        dataset_name=secondary_name,
                    )
                    if secondary_cmc is not None and secondary_mAP is not None:
                        scalar_name = str(secondary_name).replace("/", "_")
                        tbWriter.add_scalar('val_secondary/{}/Rank@1'.format(scalar_name), secondary_cmc[0], epoch)
                        tbWriter.add_scalar('val_secondary/{}/mAP'.format(scalar_name), secondary_mAP, epoch)

        if epoch % checkpoint_period == 0:
            last_checkpoint_epoch = epoch
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        torch.cuda.empty_cache()

    # final evaluation
    eval_model = None
    if best_index is not None:
        load_path = best_checkpoint_path
        if os.path.exists(load_path):
            eval_model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=0, camera_num=None, view_num=None, num_semantic_class=model_num_semantic_classes(model))
            eval_model.load_param(load_path)
            print('load best weights from {} (epoch {}, mAP {:.1%})'.format(load_path, best_index, best_mAP))
        else:
            logger.warning("Best checkpoint was not found: {}".format(load_path))
    elif last_checkpoint_epoch is not None:
        load_path = os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(last_checkpoint_epoch))
        if os.path.exists(load_path):
            eval_model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=0, camera_num=None, view_num=None, num_semantic_class=model_num_semantic_classes(model))
            eval_model.load_param(load_path)
            logger.warning("No valid validation metric was recorded; using latest checkpoint for final evaluation: {}".format(load_path))
        else:
            logger.warning("Skipping final evaluation because checkpoint was not found: {}".format(load_path))
    else:
        logger.warning("Skipping final evaluation because no checkpoint was saved.")

    if eval_model is not None:
        for testname in cfg.DATASETS.TEST:
            if 'ALL' in testname:
                testname = 'DG_' + testname.split('_')[1]
            val_loader, num_query = build_reid_test_loader(cfg, testname)
            _log_validation_header(logger, "Final", testname)
            do_inference(cfg, eval_model, val_loader, num_query, dataset_name=testname)
    
    if cfg.SOLVER.DELETE_OLD_CHECKPOINTS and eval_model is not None:
        # remove useless path files
        del_list = os.listdir(log_path)
        for fname in del_list:
            checkpoint_path = os.path.join(log_path, fname)
            if '.pth' in fname and checkpoint_path != best_checkpoint_path:
                os.remove(checkpoint_path)
                print('removing {}. '.format(checkpoint_path))
        # save final checkpoint
        print('saving final checkpoint.\nDo not interrupt the program!!!')
        torch.save(eval_model.state_dict(), os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        print('done!')
    else:
        logger.info("Keeping all checkpoint files in {}".format(log_path))

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 dataset_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("PAT.test")
    logger.info("Enter inferencing")
    logger.info("Validation dataset: {}".format(dataset_name))
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    else:
        logger.warning("CUDA unavailable; using CPU for inference.")
    model.to(device)

    model.eval()
    if should_skip_eval_if_dummy_ids(cfg, val_loader, num_query, dataset_name):
        logger.warning(dummy_eval_warning(resolve_eval_dataset_name(cfg, dataset_name)))
        return None, None

    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        class_penalty=get_class_distance_penalty(cfg),
        query_expansion=bool(getattr(cfg.TEST, "QUERY_EXPANSION", False)),
        qe_topk=int(getattr(cfg.TEST, "QE_TOPK", 5)),
        qe_alpha=float(getattr(cfg.TEST, "QE_ALPHA", 1.0)),
        cfg=cfg,
        dataset_name=dataset_name,
    )

    evaluator.reset()
    use_class_aware = model_is_class_aware(model)
    use_metadata_classes = use_metadata_classes_for_retrieval(cfg)
    log_tta_settings(logger, cfg)
    img_path_list = []
    t0 = time.time()
    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        pid = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']
        # domains = informations['others']['domains']
        with torch.no_grad():
            img = img.to(device)
            # camids = camids.to(device)
            if use_metadata_classes:
                feat, _ = extract_tta_features(model, img, cfg)
                pred_classes = get_batch_class_targets(informations)
                evaluator.update((feat, pid, camids, pred_classes, None, imgpath))
            elif use_class_aware:
                feat, class_logits = extract_tta_features(model, img, cfg, return_class_logits=True)
                pred_probs = torch.softmax(class_logits.float(), dim=1).cpu() if class_logits is not None else None
                pred_classes = pred_probs.argmax(1) if pred_probs is not None else None
                evaluator.update((feat, pid, camids, pred_classes, pred_probs, imgpath))
            else:
                feat, _ = extract_tta_features(model, img, cfg)
                evaluator.update((feat, pid, camids, None, None, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    save_metrics(cfg, dataset_name, cmc, mAP, getattr(evaluator, "last_group_info", None))
    logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP
