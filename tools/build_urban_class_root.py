import argparse
import csv
import os
import shutil
from pathlib import Path


CLASS_ALIASES = {
    "traffic": {"trafficsignal", "trafficsign", "trafficsignal", "trafficsigns", "trafficsignals"},
    "traffic_signal": {"trafficsignal", "trafficsign", "trafficsigns", "trafficsignals"},
    "crosswalk": {"crosswalk"},
    "container": {"container"},
    "rubbishbins": {"rubbishbins", "rubbishbin", "rubbish_bins", "rubbish_bin"},
    "tall_objects": {
        "trafficsignal",
        "trafficsign",
        "trafficsigns",
        "trafficsignals",
        "container",
        "rubbishbins",
        "rubbishbin",
        "rubbish_bins",
        "rubbish_bin",
    },
    "all": None,
}


def _norm_class(value):
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _selected_set(class_names):
    selected = set()
    for name in class_names:
        key = name.lower()
        if key == "all":
            return None
        aliases = CLASS_ALIASES.get(key)
        if aliases is None:
            selected.add(_norm_class(name))
        else:
            selected.update(_norm_class(v) for v in aliases)
    return selected


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def _safe_symlink(src, dst):
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)


def _copy_or_link_split(src_root, out_root, image_dir, csv_name, classes_name=None):
    src_root = Path(src_root)
    out_root = Path(out_root)
    (out_root / image_dir).mkdir(parents=True, exist_ok=True)

    csv_src = src_root / csv_name
    if csv_src.exists():
        shutil.copy2(csv_src, out_root / csv_name)
        rows = _read_csv(csv_src)
        for row in rows:
            image = row["imageName"]
            _safe_symlink(src_root / image_dir / image, out_root / image_dir / image)

    if classes_name and (src_root / classes_name).exists():
        shutil.copy2(src_root / classes_name, out_root / classes_name)


def build_class_root(args):
    src_root = Path(args.source_root)
    out_root = Path(args.output_root)
    selected = _selected_set(args.classes)
    out_train = out_root / "image_train"
    out_train.mkdir(parents=True, exist_ok=True)

    train_rows = _read_csv(src_root / "train.csv")
    class_rows = _read_csv(src_root / "train_classes.csv")
    class_by_image = {row["imageName"]: row["Class"] for row in class_rows}
    class_row_by_image = {row["imageName"]: row for row in class_rows}

    kept_train = []
    kept_classes = []
    for row in train_rows:
        class_name = class_by_image.get(row["imageName"])
        if class_name is None:
            continue
        if selected is not None and _norm_class(class_name) not in selected:
            continue
        kept_train.append(row)
        kept_classes.append(class_row_by_image[row["imageName"]])
        _safe_symlink(src_root / "image_train" / row["imageName"], out_train / row["imageName"])

    _write_csv(out_root / "train.csv", train_rows[0].keys(), kept_train)
    _write_csv(out_root / "train_classes.csv", class_rows[0].keys(), kept_classes)

    if args.include_eval:
        _copy_or_link_split(src_root, out_root, "image_query", "query.csv", "query_classes.csv")
        _copy_or_link_split(src_root, out_root, "image_test", "test.csv", "test_classes.csv")
        for name in [
            "sample_submission.csv",
            "local_train.csv",
            "local_train_classes.csv",
            "local_val_query.csv",
            "local_val_query_classes.csv",
            "local_val_gallery.csv",
            "local_val_gallery_classes.csv",
        ]:
            if (src_root / name).exists():
                shutil.copy2(src_root / name, out_root / name)

    print("source_root:", src_root)
    print("output_root:", out_root)
    print("classes:", ",".join(args.classes))
    print("train_rows:", len(kept_train))
    print("unique_ids:", len({row.get("Corresponding Indexes") or row.get("objectID") or row.get("pid") for row in kept_train}))


def main():
    parser = argparse.ArgumentParser(description="Create a class-filtered UrbanElementsReID root with symlinked images.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--include-eval", action="store_true")
    build_class_root(parser.parse_args())


if __name__ == "__main__":
    main()
