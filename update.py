import os
import csv
import torch
import argparse

import numpy as np
from config import cfg
from model import make_model
from utils.logger import setup_logger
from utils.re_ranking import re_ranking
from data.build_DG_dataloader import build_reid_test_loader
from processor.ori_vit_processor_with_amp import do_inference as do_inf
from processor.part_attention_vit_processor import do_inference as do_inf_pat
from utils.class_aware import (
    class_distance_penalty_matrix,
    get_class_distance_penalty,
    get_batch_class_targets,
    infer_num_semantic_classes,
    model_is_class_aware,
    split_inference_output,
    use_metadata_classes_for_retrieval,
)

#from torch.backends import cudnn

def extract_feature(model, dataloaders, num_query, cfg):
    features = []
    pred_classes = []
    count = 0
    img_path = []
    use_class_aware = model_is_class_aware(model)
    use_metadata_classes = use_metadata_classes_for_retrieval(cfg)
    model.eval()

    for data in dataloaders:
        img = data['images']
        #obtain values form dict data
        n, c, h, w = img.size()
        count += n
        ff = None
        class_logits_sum = None
        for i in range(2):
            input_img = img.cuda()
            if use_class_aware and not use_metadata_classes:
                outputs = model(input_img, return_class_logits=True)
                f, class_logits = split_inference_output(outputs)
                if class_logits is not None:
                    class_logits = class_logits.float()
                    class_logits_sum = class_logits if class_logits_sum is None else class_logits_sum + class_logits
            else:
                f = model(input_img)
            f = f.float()
            if ff is None:
                ff = torch.zeros_like(f).cuda()
            ff = ff + f
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features.append(ff)
        if use_metadata_classes:
            class_targets = get_batch_class_targets(data)
            if class_targets is not None:
                pred_classes.append(class_targets.cpu())
        elif use_class_aware and class_logits_sum is not None:
            pred_classes.append(class_logits_sum.argmax(1).cpu())
    features = torch.cat(features, 0)
    if pred_classes:
        pred_classes = torch.cat(pred_classes, 0).numpy()
    else:
        pred_classes = None

    # query
    qf = features[:num_query]
    # gallery
    gf = features[num_query:]
    q_pred_classes = pred_classes[:num_query] if pred_classes is not None else None
    g_pred_classes = pred_classes[num_query:] if pred_classes is not None else None
    return qf, gf, q_pred_classes, g_pred_classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/PAT.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        "--track", default="./config/PAT.yml", help="path to config file", type=str
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0,0,0, num_semantic_class=infer_num_semantic_classes(cfg))
    model.load_param(cfg.TEST.WEIGHT)

    for testname in cfg.DATASETS.TEST:
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        if cfg.MODEL.NAME == 'part_attention_vit':
            do_inf_pat(cfg, model, val_loader, num_query)
        else:
            do_inf(cfg, model, val_loader, num_query)
    with torch.no_grad():
        qf, gf, q_pred_classes, g_pred_classes = extract_feature(model, val_loader, num_query, cfg)

    # save feature
    qf=qf.cpu().numpy()
    gf=gf.cpu().numpy()
    np.save("./qf.npy", qf)
    np.save("./gf.npy", gf)

    q_g_dist = np.dot(qf, np.transpose(gf))
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))

    penalty_matrix = class_distance_penalty_matrix(
        q_pred_classes,
        g_pred_classes,
        get_class_distance_penalty(cfg),
    )
    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, q_g_penalty=penalty_matrix)

    indices = np.argsort(re_rank_dist, axis=1)[:, :100]

    m, n = indices.shape
    # # print('m: {}  n: {}'.format(m, n))
    with open(args.track, 'wb') as f_w:
        for i in range(m):
            write_line = indices[i] + 1
            write_line = ' '.join(map(str, write_line.tolist())) + '\n'
            f_w.write(write_line.encode())


    lista_nombres = ["{:06d}.jpg".format(i) for i in range(1, len(indices) + 1)]
    output_path = args.track.split(".txt")[0] + "_submission.csv"

    with open(output_path, 'w', newline='') as archivo_csv:
        csv_writter = csv.writer(archivo_csv)
        csv_writter.writerow(['imageName', 'Corresponding Indexes'])
        for numero, track in zip(lista_nombres, indices):
            track_str = ' '.join(map(str, track + 1))
            csv_writter.writerow([numero, track_str])
