#!/usr/bin/env python3
"""Compute dataset statistics from `annot/full_label.txt`.

Usage:
  python scripts/label_stats.py --labels annot/full_label.txt --images images --out results_stats

Outputs:
- Printed summary
- JSON report: <out>/report.json
- Plots: <out>/brand_hist.png
- Sparse matrix: <out>/brand_color_counts.npz
- CSVs: <out>/brand_counts.csv, <out>/color_counts.csv, <out>/pair_counts.csv

"""
import argparse
import os
import json
from collections import Counter, defaultdict
import math
import statistics
import csv
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, save_npz


def parse_labels(path):
    filenames = []
    brands = []
    colors = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 3:
                # try comma separated or csv
                toks = [t.strip() for t in line.split(',') if t.strip()]
                if len(toks) < 3:
                    print(f"Skipping malformed line {i+1}: {line}")
                    continue
            filename = ' '.join(toks[:-2])
            brand = toks[-2]
            color = toks[-1]
            filenames.append(filename)
            brands.append(brand)
            colors.append(color)
    return filenames, brands, colors


def compute_stats(counts):
    vals = sorted(counts.values())
    if not vals:
        return {}
    return {
        'min': int(min(vals)),
        'max': int(max(vals)),
        'mean': float(sum(vals) / len(vals)),
        'median': float(statistics.median(vals)),
        'n_categories': len(vals)
    }


def top_k_share(counts, k, total):
    most = [c for _, c in Counter(counts).most_common(k)]
    return sum(most) / total * 100.0


def plot_histogram(vals, outpath, title='Histogram', xlabel='count', bins=None):
    plt.figure(figsize=(8,5))
    if bins is None:
        bins = max(10, int(np.sqrt(len(vals))))
    plt.hist(vals, bins=bins, color='C0', edgecolor='k')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Number of categories')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def make_sparse_matrix(pair_counts, brands_list, colors_list):
    brand2idx = {b:i for i,b in enumerate(brands_list)}
    color2idx = {c:i for i,c in enumerate(colors_list)}
    rows = []
    cols = []
    data = []
    for (b,c), cnt in pair_counts.items():
        rows.append(brand2idx[b])
        cols.append(color2idx[c])
        data.append(cnt)
    mat = coo_matrix((data, (rows, cols)), shape=(len(brands_list), len(colors_list)), dtype=np.int32)
    return mat, brand2idx, color2idx


def gather_resolutions(filenames, images_dir):
    widths = []
    heights = []
    missing = []
    for fn in filenames:
        # try fn as-is, then join with images_dir
        if os.path.isabs(fn):
            cand = fn
        else:
            cand = fn
            if not os.path.exists(cand):
                cand = os.path.join(images_dir, fn)
        if not os.path.exists(cand):
            missing.append(fn)
            continue
        try:
            with Image.open(cand) as im:
                w,h = im.size
                widths.append(w)
                heights.append(h)
        except Exception as e:
            missing.append(fn)
    return widths, heights, missing


def save_csv_counts(counts, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['category','count'])
        for k,v in sorted(counts.items(), key=lambda x:-x[1]):
            writer.writerow([k,v])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', '-l', default='annot/clean_label.txt', help='Path to label file')
    parser.add_argument('--images', '-i', default='images', help='Path to images directory')
    parser.add_argument('--out', '-o', default='results_stats', help='Output directory')
    parser.add_argument('--min-pair-thresh', type=int, default=20, help='Threshold for small pairs')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    filenames, brands, colors = parse_labels(args.labels)
    total = len(filenames)
    print(f"Total labeled images: {total}")

    brand_counts = Counter(brands)
    color_counts = Counter(colors)
    pair_counts = Counter(zip(brands, colors))

    # Brand stats
    bstats = compute_stats(brand_counts)
    b_lt_50 = sum(1 for v in brand_counts.values() if v < 50)
    b_lt_100 = sum(1 for v in brand_counts.values() if v < 100)
    b_top5_share = sum([c for _,c in brand_counts.most_common(5)]) / total * 100.0
    b_top10_share = sum([c for _,c in brand_counts.most_common(10)]) / total * 100.0

    # Color stats
    cstats = compute_stats(color_counts)
    c_top3_share = sum([c for _,c in color_counts.most_common(3)]) / total * 100.0
    c_lt_100 = sum(1 for v in color_counts.values() if v < 100)

    # Pair stats
    n_pairs = len(pair_counts)
    pairs_lt_thresh = sum(1 for v in pair_counts.values() if v < args.min_pair_thresh)

    # Brands with <=2 colors
    brand_to_colors = defaultdict(set)
    color_to_brands = defaultdict(set)
    for b,c in zip(brands, colors):
        brand_to_colors[b].add(c)
        color_to_brands[c].add(b)
    brands_le2_colors = sum(1 for b,s in brand_to_colors.items() if len(s) <= 2)
    colors_le2_brands = sum(1 for c,s in color_to_brands.items() if len(s) <= 2)

    # Sparse matrix
    brands_list = sorted(brand_counts.keys())
    colors_list = sorted(color_counts.keys())
    mat, brand2idx, color2idx = make_sparse_matrix(pair_counts, brands_list, colors_list)
    save_npz(os.path.join(args.out, 'brand_color_counts.npz'), mat)
    # Also save dense CSV for easy viewing and a heatmap PNG
    dense = mat.toarray()
    csv_matrix_path = os.path.join(args.out, 'brand_color_matrix.csv')
    with open(csv_matrix_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['brand\\color'] + colors_list)
        for i, b in enumerate(brands_list):
            writer.writerow([b] + dense[i].tolist())
    # Heatmap
    try:
        plt.figure(figsize=(max(6, len(colors_list)*0.5), max(4, len(brands_list)*0.3)))
        plt.imshow(dense, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='count')
        plt.xticks(range(len(colors_list)), colors_list, rotation=45, ha='right')
        plt.yticks(range(len(brands_list)), brands_list)
        plt.title('Brand × Color counts heatmap')
        plt.tight_layout()
        heatmap_path = os.path.join(args.out, 'brand_color_heatmap.png')
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
    except Exception:
        heatmap_path = None

    # Resolutions
    widths, heights, missing = gather_resolutions(filenames, args.images)
    areas = [w*h for w,h in zip(widths, heights)]
    res_stats = {}
    if areas:
        res_stats['area_min'] = int(min(areas))
        res_stats['area_max'] = int(max(areas))
        res_stats['area_median'] = float(statistics.median(areas))
        res_stats['width_min'] = int(min(widths))
        res_stats['width_max'] = int(max(widths))
        res_stats['width_median'] = float(statistics.median(widths))
        res_stats['height_min'] = int(min(heights))
        res_stats['height_max'] = int(max(heights))
        res_stats['height_median'] = float(statistics.median(heights))
    else:
        res_stats['note'] = 'No images found for resolution analysis; check images path or filenames.'

    # Plots
    plot_histogram(list(brand_counts.values()), os.path.join(args.out, 'brand_hist.png'), title='Distribution of images per brand', xlabel='images per brand')

    # Save counts CSVs
    save_csv_counts(brand_counts, os.path.join(args.out, 'brand_counts.csv'))
    save_csv_counts(color_counts, os.path.join(args.out, 'color_counts.csv'))
    with open(os.path.join(args.out, 'pair_counts.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['brand','color','count'])
        for (b,c),cnt in sorted(pair_counts.items(), key=lambda x:-x[1]):
            writer.writerow([b,c,cnt])

    # Build report
    report = {
        'total_images': total,
        'brands': {
            'n_brands': bstats.get('n_categories'),
            'min': bstats.get('min'), 'max': bstats.get('max'), 'mean': bstats.get('mean'), 'median': bstats.get('median'),
            'n_brands_lt_50': b_lt_50, 'n_brands_lt_100': b_lt_100,
            'top_5_share_percent': b_top5_share, 'top_10_share_percent': b_top10_share
        },
        'colors': {
            'n_colors': cstats.get('n_categories'),
            'min': cstats.get('min'), 'max': cstats.get('max'), 'mean': cstats.get('mean'), 'median': cstats.get('median'),
            'top_3_share_percent': c_top3_share, 'n_colors_lt_100': c_lt_100
        },
        'pairs': {
            'n_pairs': n_pairs,
            f'n_pairs_lt_{args.min_pair_thresh}': pairs_lt_thresh
        },
        'brand_color_matrix': {
            'n_rows': len(brands_list), 'n_cols': len(colors_list),
            'npz': os.path.join(args.out, 'brand_color_counts.npz'),
            'csv': os.path.join(args.out, 'brand_color_matrix.csv'),
            'heatmap': os.path.join(args.out, 'brand_color_heatmap.png')
        },
        'brand_color_small_stats': {
            'brands_le2_colors': brands_le2_colors,
            'colors_le2_brands': colors_le2_brands
        },
        'resolutions': res_stats,
        'missing_images_for_resolution': len(missing)
    }

    with open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print concise human-friendly summary
    print('\n=== Summary ===')
    print(f"Brands: {report['brands']['n_brands']} (min={report['brands']['min']}, max={report['brands']['max']}, mean={report['brands']['mean']:.2f}, median={report['brands']['median']})")
    print(f"  Brands with <50 images: {report['brands']['n_brands_lt_50']}")
    print(f"  Brands with <100 images: {report['brands']['n_brands_lt_100']}")
    print(f"  Top-5 brands cover {report['brands']['top_5_share_percent']:.2f}%")
    print(f"  Top-10 brands cover {report['brands']['top_10_share_percent']:.2f}%")

    print(f"\nColors: {report['colors']['n_colors']} (min={report['colors']['min']}, max={report['colors']['max']}, mean={report['colors']['mean']:.2f}, median={report['colors']['median']})")
    print(f"  Top-3 colors cover {report['colors']['top_3_share_percent']:.2f}%")
    print(f"  Colors with <100 images: {report['colors']['n_colors_lt_100']}")

    print(f"\nPairs (brand,color): {report['pairs']['n_pairs']}")
    print(f"  Pairs with <{args.min_pair_thresh} images: {pairs_lt_thresh}")

    print(f"\nBrands that appear with ≤2 colors: {brands_le2_colors}")
    print(f"Colors that appear with ≤2 brands: {colors_le2_brands}")

    print(f"\nSaved histogram to {os.path.join(args.out, 'brand_hist.png')}")
    print(f"Saved matrices and CSVs to {args.out}")
    print(f"Resolution analysis: missing images={len(missing)}; resolutions saved to report.json")

    print('\nReport JSON: ' + os.path.join(args.out, 'report.json'))

if __name__ == '__main__':
    main()
