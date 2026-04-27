"""
Diagnostic notebook for Urban Elements ReID 2026.

Run this BEFORE committing to an architectural change. The answers to these
questions determine which approach actually has a chance of helping.

Designed to be run cell-by-cell in a notebook (each # %% block is a cell).
Or just `python diagnostics.py` to run all of them sequentially.

Inputs expected:
    --root <ROOT>     directory containing train.csv, query.csv, test.csv,
                      *_classes.csv, image_train/, image_query/, image_test/
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True,
                   help='Dataset root directory')
    p.add_argument('--checkpoint', default=None,
                   help='Optional: path to a trained PAT checkpoint for '
                        'per-class mAP analysis. If not provided, that '
                        'section is skipped.')
    return p.parse_args()


# %% ============================================================
# CELL 1 — Load all CSVs
# ================================================================
def load_csvs(root):
    train = pd.read_csv(os.path.join(root, 'train.csv'))
    train_cls = pd.read_csv(os.path.join(root, 'train_classes.csv'))
    query = pd.read_csv(os.path.join(root, 'query.csv'))
    query_cls = pd.read_csv(os.path.join(root, 'query_classes.csv'))
    test = pd.read_csv(os.path.join(root, 'test.csv'))
    test_cls = pd.read_csv(os.path.join(root, 'test_classes.csv'))
    return train, train_cls, query, query_cls, test, test_cls


# %% ============================================================
# CELL 2 — Q1: How are identities distributed across cameras?
# ================================================================
# This answers: "are pids tracked across cameras, or is each camera its own
# identity space?" Without this we don't know if cross-camera matching is
# even *attempted* in the labeling.
def q1_pid_camera_coverage(train, train_cls):
    print('=' * 60)
    print('Q1. PID coverage across train cameras (c001-c003)')
    print('=' * 60)
    m = train.merge(train_cls[['imageName', 'Class']], on='imageName')

    pid_to_cams = m.groupby('Corresponding Indexes')['cameraID'].apply(
        lambda s: tuple(sorted(s.unique())))
    n_cams = pid_to_cams.apply(len)

    print(f'\nTotal unique pids in train: {len(pid_to_cams)}')
    print('\nDistribution of #cameras each pid appears in:')
    print(n_cams.value_counts().sort_index().to_string())

    print('\nWhich camera-combinations cover most pids:')
    print(pid_to_cams.value_counts().head(10).to_string())

    print('\nINTERPRETATION:')
    print('  If most pids appear in 3 cameras → cross-camera linking is real.')
    print('  If most pids appear in only 1 camera → identities are camera-')
    print('  local, and the "ReID" task is essentially within-camera.')
    print()

    # EXPECTED on this dataset (computed from CSVs in chat):
    # 927 of 1088 pids appear in all 3 cameras → cross-camera linking IS real
    # for forward-pass cameras. Good news.
    return pid_to_cams, m


# %% ============================================================
# CELL 3 — Q2: Per-class pid breakdown
# ================================================================
def q2_per_class_pid_coverage(merged_train):
    print('=' * 60)
    print('Q2. Per-class breakdown of pid camera coverage')
    print('=' * 60)
    pid_class = merged_train.groupby('Corresponding Indexes')['Class'].first()
    pid_to_cams = merged_train.groupby('Corresponding Indexes')['cameraID'].apply(
        lambda s: len(s.unique()))
    combo = pd.DataFrame({'class': pid_class, 'n_cams': pid_to_cams})
    table = combo.groupby(['class', 'n_cams']).size().unstack(fill_value=0)
    print(table)
    print('\nINTERPRETATION:')
    print('  Per-class fraction with full 3-camera coverage tells you which')
    print('  classes you have the cleanest cross-camera signal for.')
    print()


# %% ============================================================
# CELL 4 — Q3: The big question — front/back labeling for c004 reverse pass
# ================================================================
# CRITICAL: query.csv and test.csv DO NOT have pid columns. So we cannot
# directly verify whether c004 (reverse) and c001-c003 (forward) views of
# the same physical sign share a pid. The ground-truth linking lives only
# in the Kaggle scoring backend.
#
# But there are two indirect signals:
#   (a) c004 doesn't appear in train at all → model never sees reverse views
#       during training. Whatever cross-direction matching happens is purely
#       from generalization.
#   (b) The number of c004 queries (928) vs gallery (2844) tells us roughly
#       how many physical objects are expected to be reidentified.
def q3_reverse_pass_linking(train, query, test):
    print('=' * 60)
    print('Q3. Reverse-pass (c004) linking — what we can know')
    print('=' * 60)
    print('\nTrain cameras:', sorted(train['cameraID'].unique()))
    print('Query cameras:', sorted(query['cameraID'].unique()))
    print('Test  cameras:', sorted(test['cameraID'].unique()))
    print()
    print(f'Query images:  {len(query)}  (all c004)')
    print(f'Gallery images: {len(test)}  (forward-pass)')
    print(f'\nINTERPRETATION:')
    print('  The model NEVER sees c004 (reverse-pass) crops during training.')
    print('  Whatever reverse-to-forward matching happens is generalization,')
    print('  not learned. This is the core domain shift in this challenge.')
    print()
    print('  Front/back labeling (whether the same physical sign in c004 has')
    print('  the same pid as in c001-c003) is NOT visible in the CSVs. It')
    print('  exists only in the Kaggle scoring backend.')
    print()
    print('  For the paper: be honest that the back-of-sign case is')
    print('  unsolvable without explicit cross-direction supervision in the')
    print('  training data, which this challenge does not provide.')
    print()


# %% ============================================================
# CELL 5 — Q4: How many crops per pid?
# ================================================================
# Singletons can't form positive triplet pairs. Identifies how much your
# filter removed.
def q4_crops_per_pid(merged_train):
    print('=' * 60)
    print('Q4. Crops per identity (training data quality)')
    print('=' * 60)
    counts = merged_train.groupby('Corresponding Indexes').size()
    print(f'\nTotal pids: {len(counts)}')
    print(f'  with 1 crop:    {(counts == 1).sum()}')
    print(f'  with 2 crops:   {(counts == 2).sum()}')
    print(f'  with 3-5 crops: {((counts >= 3) & (counts <= 5)).sum()}')
    print(f'  with 6+ crops:  {(counts >= 6).sum()}')
    print(f'\n  Median: {int(counts.median())}, mean: {counts.mean():.1f}, '
          f'max: {counts.max()}')

    print('\n  Per-class mean crops/pid:')
    pid_class = merged_train.groupby('Corresponding Indexes')['Class'].first()
    df = pd.DataFrame({'count': counts, 'class': pid_class})
    print(df.groupby('class')['count'].agg(['mean', 'median', 'min', 'max']))
    print()


# %% ============================================================
# CELL 6 — Q5: Crop size distribution (the resolution problem)
# ================================================================
# This is where the unsalvageable-low-res-crops question gets answered.
# Computes width/height for every train image. Caches to disk because it
# takes a few minutes on 11k images.
def q5_crop_sizes(root, train, train_cls):
    print('=' * 60)
    print('Q5. Crop size distribution (where the resolution wall is)')
    print('=' * 60)

    cache = os.path.join(root, '_diag_image_sizes.csv')
    if os.path.exists(cache):
        sizes = pd.read_csv(cache)
        print(f'  loaded cached sizes from {cache}')
    else:
        rows = []
        image_dir = os.path.join(root, 'image_train')
        names = train['imageName'].tolist()
        for i, name in enumerate(names):
            try:
                with Image.open(os.path.join(image_dir, name)) as im:
                    rows.append((name, im.size[0], im.size[1]))
            except Exception:
                rows.append((name, 0, 0))
            if (i + 1) % 2000 == 0:
                print(f'  ... read {i+1}/{len(names)}')
        sizes = pd.DataFrame(rows, columns=['imageName', 'w', 'h'])
        sizes.to_csv(cache, index=False)
        print(f'  cached sizes to {cache}')

    sizes['min_side'] = sizes[['w', 'h']].min(axis=1)
    sizes['area'] = sizes['w'] * sizes['h']
    sizes['ratio'] = sizes['w'] / sizes['h'].clip(lower=1)
    sizes_with_class = sizes.merge(
        train_cls[['imageName', 'Class']], on='imageName')

    print('\nOverall min_side percentiles (train):')
    pcts = [5, 10, 25, 50, 75, 90, 95]
    print('  ', list(zip(pcts, np.percentile(sizes['min_side'], pcts).astype(int))))

    print('\nFraction of train below threshold T (where T is min(w,h)):')
    for thresh in [16, 24, 32, 48, 64, 96]:
        frac = (sizes['min_side'] < thresh).mean()
        print(f'  min_side < {thresh:3d}: {frac:6.1%}')

    print('\nPer-class min_side median (smaller = harder class):')
    for cls, grp in sizes_with_class.groupby('Class'):
        print(f'  {cls:14s}: median={int(grp["min_side"].median())}, '
              f'p10={int(grp["min_side"].quantile(0.10))}, '
              f'p90={int(grp["min_side"].quantile(0.90))}')

    print('\nPer-class width/height ratio (>>1 = wide, <<1 = tall):')
    for cls, grp in sizes_with_class.groupby('Class'):
        print(f'  {cls:14s}: median ratio={grp["ratio"].median():.2f}, '
              f'p10={grp["ratio"].quantile(0.10):.2f}, '
              f'p90={grp["ratio"].quantile(0.90):.2f}')

    print('\nINTERPRETATION:')
    print('  If >30% of crops have min_side < 32, those are essentially')
    print('  unsalvageable — they have no information for fine-grained ReID')
    print('  and any "extract internal symbol" architecture will fail on them.')
    print('  Per-class ratio tells you whether square input is reasonable.')
    print('  Crosswalks at ratio > 3 cannot be properly fit into a square.')
    print()
    return sizes_with_class


# %% ============================================================
# CELL 7 — Q6: Compute size distribution for QUERY and GALLERY too
# ================================================================
# Critical: even if your training data is OK, if the query crops are tiny
# you can't ReID them. This separates "method limited" from "data limited".
def q6_query_gallery_sizes(root, query, query_cls, test, test_cls):
    print('=' * 60)
    print('Q6. Query and gallery crop sizes (the actual eval bottleneck)')
    print('=' * 60)

    def compute_sizes(image_dir, df, cls_df, cache_path):
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path)
        rows = []
        for name in df['imageName']:
            try:
                with Image.open(os.path.join(image_dir, name)) as im:
                    rows.append((name, im.size[0], im.size[1]))
            except Exception:
                rows.append((name, 0, 0))
        s = pd.DataFrame(rows, columns=['imageName', 'w', 'h'])
        s.to_csv(cache_path, index=False)
        return s

    qsizes = compute_sizes(
        os.path.join(root, 'image_query'), query, query_cls,
        os.path.join(root, '_diag_query_sizes.csv'))
    qsizes['min_side'] = qsizes[['w', 'h']].min(axis=1)
    qsizes = qsizes.merge(query_cls[['imageName', 'Class']], on='imageName')

    gsizes = compute_sizes(
        os.path.join(root, 'image_test'), test, test_cls,
        os.path.join(root, '_diag_test_sizes.csv'))
    gsizes['min_side'] = gsizes[['w', 'h']].min(axis=1)
    gsizes = gsizes.merge(test_cls[['imageName', 'Class']], on='imageName')

    print('\nQUERY (c004 reverse-pass) per-class min_side:')
    for cls, grp in qsizes.groupby('Class'):
        below_32 = (grp['min_side'] < 32).mean()
        print(f'  {cls:14s}: median={int(grp["min_side"].median())}, '
              f'fraction with min_side<32: {below_32:.0%}')

    print('\nGALLERY (forward-pass) per-class min_side:')
    for cls, grp in gsizes.groupby('Class'):
        below_32 = (grp['min_side'] < 32).mean()
        print(f'  {cls:14s}: median={int(grp["min_side"].median())}, '
              f'fraction with min_side<32: {below_32:.0%}')

    print('\nINTERPRETATION:')
    print('  Compare query and gallery medians per class. If queries are')
    print('  systematically smaller than gallery items, c004 cropping was')
    print('  more aggressive — adds another domain mismatch on top of the')
    print('  forward/reverse direction.')
    print('  If a large fraction of queries have min_side<32, those queries')
    print('  cannot be re-identified by any visual method. Cap your')
    print('  expectations of method-driven mAP improvements.')
    print()
    return qsizes, gsizes


# %% ============================================================
# CELL 8 — Q7 (optional): per-class mAP on the existing checkpoint
# ================================================================
# This is the MOST informative diagnostic, but requires running inference
# on local validation. Skipped by default — runs only if you pass a
# checkpoint path. Skips here entirely; the principle is described:
def q7_per_class_map_template():
    print('=' * 60)
    print('Q7. Per-class mAP on local validation (template)')
    print('=' * 60)
    print('\nNot computed in this script — requires running inference.')
    print('To compute manually:')
    print('  1. Load your trained checkpoint.')
    print('  2. Run inference on local validation set (image_train as')
    print('     both query and gallery — leaky but informative for relative')
    print('     comparisons).')
    print('  3. Modify utils/metrics.py:R1_mAP_eval.compute() to compute')
    print('     mAP separately for each class, by masking the distmat to')
    print('     same-class queries/gallery only.')
    print('  4. Look at per-class mAP. Where the gap is biggest, that\'s')
    print('     where method changes can move the most.')
    print()
    print('  Key question: which class has LOWEST mAP?')
    print('  - If signs (the majority class) → most fixes should target signs')
    print('  - If crosswalks → square-input is hurting them, address geometry')
    print('  - If everything is around 30-40% → it\'s the domain shift')
    print('    (c001-c003 → c004), not the architecture')


# %% ============================================================
# CELL 9 — Summary: what this means for your architectural choice
# ================================================================
def summarize():
    print('=' * 60)
    print('Decision framework based on the answers above')
    print('=' * 60)
    print('''
Q1+Q2: cross-camera pid linking
  - If most pids span all 3 train cameras → forward-pass ReID works,
    your model has enough cross-camera signal in the train data.
  - If pids are mostly camera-local → there\'s a labeling problem.

Q3: reverse-pass (c004) is unseen at training
  - Always true for this challenge. The c004 → forward gallery match
    is the structural domain shift. No amount of architecture redesign
    on the model side fixes this; you need either (a) more training
    data with reverse views, or (b) accept the score ceiling.

Q4: crops-per-pid
  - Singletons (1-2 crops) are useless for triplet loss. Filter them.
  - If a class has tiny mean crops/pid (e.g., 3 vs 10 for another class),
    that class is the bottleneck for that class\'s mAP.

Q5: training crop sizes
  - The fraction below ~32px tells you what fraction of training data is
    noise. If it\'s 20%+, filtering is high-priority.
  - The per-class ratio tells you whether square input is OK.

Q6: query+gallery sizes — the eval-time bottleneck
  - If query crops are systematically tiny, the score has a hard ceiling
    that no architecture can break through. Be honest in the paper.

Q7: per-class mAP — the architecture decision
  - Where you have headroom is where to target. A method that helps signs
    only matters if signs are the bottleneck.

For the SUPERPIXEL / part-based architecture decision specifically:
  - Helps medium-to-large crops with internal structure (signs > 64px,
    containers, signs with clear icons).
  - Doesn\'t help: small crops (no info), back-of-sign (uniform texture),
    crosswalks (can be panoramic, no point parts).
  - Estimate the fraction of queries it can plausibly help by combining
    Q5 and the class distribution. If 30%+ of queries are
    medium-size signs/containers, the architecture is worth trying.
  - If <15%, focus on the cleanup run + post-processing instead.
''')


# %% ============================================================
# CELL 10 — Tie it all together
# ================================================================
def main():
    args = parse_args()
    train, train_cls, query, query_cls, test, test_cls = load_csvs(args.root)

    pid_to_cams, merged_train = q1_pid_camera_coverage(train, train_cls)
    q2_per_class_pid_coverage(merged_train)
    q3_reverse_pass_linking(train, query, test)
    q4_crops_per_pid(merged_train)
    sizes = q5_crop_sizes(args.root, train, train_cls)
    qsizes, gsizes = q6_query_gallery_sizes(
        args.root, query, query_cls, test, test_cls)
    q7_per_class_map_template()
    summarize()


if __name__ == '__main__':
    main()
