#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import timm
from timm.data import create_transform, resolve_data_config


class CsvImageDataset(Dataset):
    def __init__(self, root, csv_name, image_dir, transform):
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.transform = transform
        self.rows = self._read_rows(self.root / csv_name)

    @staticmethod
    def _read_rows(csv_path):
        with open(csv_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                raise ValueError("{} has no header".format(csv_path))
            columns = {name.strip(): name for name in reader.fieldnames}
            image_col = columns.get("imageName") or columns.get("image_name")
            if image_col is None:
                raise ValueError("{} is missing imageName column".format(csv_path))
            return [str(row[image_col]).strip() for row in reader]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_name = self.rows[index]
        image_path = self.image_dir / image_name
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), image_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot timm image retrieval exporter for Urban Elements ReID."
    )
    parser.add_argument("--root", required=True, help="Challenge dataset root.")
    parser.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--output", required=True, help="Output submission CSV path.")
    parser.add_argument("--query-csv", default="query.csv")
    parser.add_argument("--gallery-csv", default="test.csv")
    parser.add_argument("--query-dir", default="image_query")
    parser.add_argument("--gallery-dir", default="image_test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=0, help="Optional square input size override.")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tta-flip", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-features-prefix", default="")
    return parser.parse_args()


def make_model(model_name, device, img_size=0):
    kwargs = {"pretrained": True, "num_classes": 0}
    if img_size > 0:
        kwargs["img_size"] = img_size
    model = timm.create_model(model_name, **kwargs)
    model.eval()
    model.to(device)
    return model


def extract_features(model, loader, device, use_amp=False, tta_flip=False, label="images"):
    features = []
    names = []
    autocast_enabled = bool(use_amp and device.type == "cuda")
    with torch.no_grad():
        for batch_idx, (images, batch_names) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                feat = model(images)
                if tta_flip:
                    flip_feat = model(torch.flip(images, dims=[3]))
                    feat = 0.5 * (feat + flip_feat)
            feat = F.normalize(feat.float(), dim=1)
            features.append(feat.cpu())
            names.extend(batch_names)
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(loader):
                print(
                    "{}: batch {}/{} ({}/{})".format(
                        label,
                        batch_idx,
                        len(loader),
                        len(names),
                        len(loader.dataset),
                    ),
                    flush=True,
                )
    return torch.cat(features, dim=0).numpy(), names


def numeric_gallery_indexes(gallery_names):
    indexes = []
    for name in gallery_names:
        stem = os.path.splitext(os.path.basename(name))[0]
        try:
            indexes.append(int(stem))
        except ValueError:
            return np.arange(1, len(gallery_names) + 1, dtype=np.int64)
    indexes = np.asarray(indexes, dtype=np.int64)
    sequential = np.arange(1, len(gallery_names) + 1, dtype=np.int64)
    if sorted(indexes.tolist()) != sequential.tolist():
        return sequential
    return indexes


def rank_and_write(qf, gf, query_names, gallery_names, output_path, topk):
    qf = qf.astype(np.float32, copy=False)
    gf = gf.astype(np.float32, copy=False)
    topk = min(int(topk), gf.shape[0])
    gallery_indexes = numeric_gallery_indexes(gallery_names)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["imageName", "Corresponding Indexes"])
        for start in range(0, qf.shape[0], 128):
            scores = np.matmul(qf[start:start + 128], gf.T)
            part = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
            part_scores = np.take_along_axis(scores, part, axis=1)
            order = np.argsort(-part_scores, axis=1)
            ranked = np.take_along_axis(part, order, axis=1)
            for offset, row in enumerate(ranked):
                indexes = gallery_indexes[row]
                writer.writerow([
                    query_names[start + offset],
                    " ".join(str(int(index)) for index in indexes),
                ])


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print("device={}".format(device))
    print("model={}".format(args.model))

    model = make_model(args.model, device, args.img_size)
    data_config = resolve_data_config({}, model=model)
    if args.img_size > 0:
        data_config["input_size"] = (3, args.img_size, args.img_size)
    transform = create_transform(**data_config, is_training=False)
    print("data_config={}".format(data_config))

    query_set = CsvImageDataset(args.root, args.query_csv, args.query_dir, transform)
    gallery_set = CsvImageDataset(args.root, args.gallery_csv, args.gallery_dir, transform)
    query_loader = DataLoader(
        query_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    gallery_loader = DataLoader(
        gallery_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    qf, query_names = extract_features(
        model, query_loader, device, args.amp, args.tta_flip, label="query"
    )
    gf, gallery_names = extract_features(
        model, gallery_loader, device, args.amp, args.tta_flip, label="gallery"
    )
    print("features: query={} gallery={} dim={}".format(qf.shape[0], gf.shape[0], qf.shape[1]))

    if args.save_features_prefix:
        prefix = Path(args.save_features_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(prefix) + "_qf.npy", qf)
        np.save(str(prefix) + "_gf.npy", gf)

    rank_and_write(qf, gf, query_names, gallery_names, args.output, args.topk)
    print("wrote {}".format(args.output))


if __name__ == "__main__":
    main()
