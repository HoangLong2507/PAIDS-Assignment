"""
export_static.py
================
Pre-generates all data files needed by the static GitHub Pages site.

Run once from the project root:
    cd "Programming for Artificial Intelligence and Data Science"
    python export_static.py

Output: docs/ directory ready to deploy on GitHub Pages.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
DOCS_DIR = ROOT / "docs"
EDA_DIR = APP_DIR / "templates" / "assignment1" / "EDA"


def _load_eda(name: str) -> object:
    """Import an EDA module directly from its .py file (Python 3.14 compatible)."""
    modname = f"eda_{name.lower()}"
    path = EDA_DIR / name / "eda.py"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    # Must register in sys.modules BEFORE exec so dataclasses resolves __module__
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class _SafeEncoder(json.JSONEncoder):
    """Handle numpy scalars, pandas Timestamp, NaN, NaT."""
    def default(self, obj):
        import math
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                v = float(obj)
                return None if math.isnan(v) or math.isinf(v) else v
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        try:
            import pandas as pd
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat() if not pd.isnull(obj) else None
        except ImportError:
            pass
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, cls=_SafeEncoder)
    print(f"  ✓ {path.relative_to(ROOT)}")


def fig_to_json_file(fig, path: Path) -> None:
    """Serialize a Plotly figure to JSON and save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fig.to_json(), encoding="utf-8")
    print(f"  ✓ {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 0. Assignment 2 (Netflix ML) Plotly export (from executed notebook outputs)
# ---------------------------------------------------------------------------

# Looks for HTML outputs containing:
#   <script>(function(){var fig={...};Plotly.newPlot('plotly-x', ...);})();</script>
_PLOTLY_FIG_RE = re.compile(
    r"var\s+fig\s*=\s*(\{.*?\})\s*;\s*Plotly\.newPlot",
    flags=re.S,
)


def _iter_nb_html_outputs(nb: dict) -> "list[str]":
    htmls: list[str] = []
    for cell in nb.get("cells", []) or []:
        for out in cell.get("outputs", []) or []:
            data = (out.get("data") or {})
            html = data.get("text/html")
            if not html:
                continue
            if isinstance(html, list):
                html = "".join(html)
            if isinstance(html, str):
                htmls.append(html)
    return htmls


def export_assignment2_netflix(out_dir: Path) -> None:
    """Export Plotly figures from the executed Netflix notebook into JSON files."""
    print("\n=== Assignment 2: Netflix ML (Plotly JSON export) ===")
    ensure_dir(out_dir)

    nb_path = ROOT / "Netflix_ML_Pipeline.executed.ipynb"
    if not nb_path.exists():
        print(f"  ! Skip: notebook not found: {nb_path.relative_to(ROOT)}")
        return

    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    figs: list[dict] = []
    for html in _iter_nb_html_outputs(nb):
        for m in _PLOTLY_FIG_RE.finditer(html):
            blob = m.group(1)
            try:
                fig = json.loads(blob)
            except json.JSONDecodeError:
                continue
            # cleanup noise
            if isinstance(fig, dict) and isinstance(fig.get("config"), dict):
                fig["config"].pop("plotlyServerURL", None)
            figs.append(fig)

    if not figs:
        print("  ! No Plotly figures found in notebook outputs.")
        return

    manifest = {"count": len(figs), "figures": []}
    for i, fig in enumerate(figs, start=1):
        fname = f"fig_{i}.json"
        (out_dir / fname).write_text(json.dumps(fig, ensure_ascii=False), encoding="utf-8")
        manifest["figures"].append({"file": fname, "index": i})
        print(f"  ✓ {out_dir.relative_to(ROOT) / fname}")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Assignment 2 export complete: {len(figs)} figure(s).")


# ---------------------------------------------------------------------------
# 1. Text EDA Export
# ---------------------------------------------------------------------------

def export_text(out_dir: Path) -> None:
    print("\n=== Text EDA ===")
    ensure_dir(out_dir)

    eda = _load_eda("Text")

    df = eda.load_df()

    # Overview
    ov = eda.get_overview(df)
    write_json(out_dir / "overview.json", {
        "total_reviews": ov.total_reviews,
        "total_features": ov.total_features,
        "rating_min": ov.rating_min,
        "rating_max": ov.rating_max,
        "missing_review": ov.missing_review,
        "missing_rating": ov.missing_rating,
        "duplicates": ov.duplicates,
        "avg_words": round(ov.avg_words, 2),
        "avg_chars": round(ov.avg_chars, 2),
    })

    # Per-rating stats
    per_rating = eda.per_rating_length_stats(df).to_dict(orient="records")
    write_json(out_dir / "per_rating_stats.json", per_rating)

    # Rating table
    rating_table = eda.rating_distribution(df).to_dict(orient="records")
    write_json(out_dir / "rating_table.json", rating_table)

    # Figures
    fig_to_json_file(eda.fig_rating_pie(df), out_dir / "fig_rating_pie.json")

    fig_stop, total_words, total_stop = eda.fig_stopwords_raw(df)
    fig_to_json_file(fig_stop, out_dir / "fig_stopwords.json")
    write_json(out_dir / "text_stats.json", {
        "total_words": total_words,
        "total_stop": total_stop,
        "stop_pct": round((total_stop / total_words * 100) if total_words else 0.0, 4),
    })

    print("  Cleaning text (may take a moment)...")
    cleaned = eda.clean_text_series(df)
    vocab_before, vocab_after = eda.vocab_sizes(df, cleaned)
    write_json(out_dir / "vocab_sizes.json", {
        "vocab_before": vocab_before,
        "vocab_after": vocab_after,
        "vocab_reduction_pct": round((1 - vocab_after / vocab_before) * 100 if vocab_before else 0.0, 2),
    })

    stats_df = eda.vocab_richness(df, cleaned)
    write_json(out_dir / "vocab_table.json", stats_df.to_dict(orient="records"))
    fig_to_json_file(eda.fig_vocab_richness(stats_df), out_dir / "fig_vocab_richness.json")

    print("  Computing top-50 words per rating...")
    for rating, fig in eda.fig_top50_words_by_rating(df, cleaned).items():
        fig_to_json_file(fig, out_dir / f"fig_top50_{rating}.json")

    print("  Computing TF-IDF (slow — ~2 min)...")
    for rating, fig in eda.fig_tfidf_top_terms(df, cleaned).items():
        fig_to_json_file(fig, out_dir / f"fig_tfidf_{rating}.json")

    print("  Computing bigrams...")
    for rating, fig in eda.fig_bigrams(df, cleaned).items():
        fig_to_json_file(fig, out_dir / f"fig_bigrams_{rating}.json")

    print("  Computing similarity matrix (slow)...")
    sim_fig, sim_labels, sim_matrix = eda.fig_rating_similarity(df)
    fig_to_json_file(sim_fig, out_dir / "fig_similarity.json")
    write_json(out_dir / "sim_matrix.json", {
        "labels": sim_labels,
        "matrix": sim_matrix,
    })

    # Hardcoded insights HTML
    insights_html = """
<h3>1. Dataset Characteristics</h3>
<ul>
  <li><b>Rating Distribution:</b> moderately imbalanced (Rating 5★ leads at 44.19%.)</li>
  <li><b>Review Length:</b> Overall average of 104.38 words and average of 724.90 characters per review (raw, all words included). Negative reviews are notably longer with average 126.60 words (867 characters) for 2★ reviews, and 5★ reviews at only 93.96 words (661 characters). Frustrated guests write more; satisfied guests write less.</li>
  <li><b>Corpus Size:</b> 2,138,765 total raw words across all 20,491 reviews</li>
  <li><b>Stop Words Density:</b> Only 45,273 stop words detected (2.1% of corpus), which far below typical English prose (30–40%).</li>
  <li><b>Vocabulary Size:</b> 102,008 tokens before cleaning → 75,048 after removing stop words &amp; non-alpha (26.4% reduction)</li>
  <li><b>Data Quality:</b> 0 missing values, 0 duplicate rows, which is an exceptionally clean dataset</li>
</ul>

<h3>2. Vocabulary &amp; Language Patterns</h3>
<ul>
  <li><b>Vocabulary Richness by Rating (after cleaning):</b> 5★ has ~5 times more total words than 1★ (reflects class imbalance), and also ~3 times more unique vocabulary, i.e., positive reviews have richer, more varied language</li>
</ul>

<h3>3. Data Processing</h3>
<ul>
  <li><b>Raw Stats:</b> Word/character counts include ALL words (not, no, nothing, none, cant, etc.)</li>
  <li><b>Vocabulary Stats:</b> Word frequency, TF-IDF, and bigram analyses computed AFTER removing stop words, punctuation, numbers</li>
  <li><b>Stop Word Anomaly:</b> 2.1% stop word rate (vs typical 30–40%) suggests text is already preprocessed or written in a telegraphic/informal style common to review platforms — affects absolute stop word counts but not relative vocabulary analysis</li>
  <li><b>Why TF-IDF + Bigrams:</b> Top raw frequencies (hotel, room) are identical across all ratings, which is meaningless for discrimination. TF-IDF surfaces weighted terms (nt, staff) while Bigrams restore context (&quot;credit card&quot; vs &quot;great location&quot;)</li>
</ul>
""".strip()
    write_json(out_dir / "insights.json", {"html": insights_html})

    print("Text EDA export complete.")


# ---------------------------------------------------------------------------
# 2. Tabular EDA Export
# ---------------------------------------------------------------------------

def export_tabular(out_dir: Path) -> None:
    print("\n=== Tabular EDA ===")
    ensure_dir(out_dir)

    tabular_eda = _load_eda("Tabular")

    df = tabular_eda.load_df()

    overview = tabular_eda.get_overview(df)
    write_json(out_dir / "overview.json", {
        "total_samples": overview.total_samples,
        "total_features": overview.total_features,
        "numerical_features": overview.numerical_features,
        "categorical_features": overview.categorical_features,
    })

    missing_df = tabular_eda.missing_values_table(df)
    write_json(out_dir / "missing_table.json", missing_df.to_dict(orient="records"))
    fig_to_json_file(tabular_eda.fig_missing_bar(missing_df), out_dir / "fig_missing.json")

    fig_release, release_stats = tabular_eda.fig_release_year_distribution(df)
    fig_to_json_file(fig_release, out_dir / "fig_release_year.json")
    write_json(out_dir / "release_stats.json", {
        "mean": float(release_stats["mean"]),
        "median": float(release_stats["median"]),
        "std": float(release_stats["std"]),
        "min": float(release_stats["min"]),
        "max": float(release_stats["max"]),
    })

    print("  Computing categorical features...")
    cat_results = tabular_eda.fig_categorical_top15(df)
    cat_payload = {}
    for feature, (fig, stats) in cat_results.items():
        cat_payload[feature] = {
            "fig": json.loads(fig.to_json()),
            "stats": stats,
        }
    write_json(out_dir / "cat_payload.json", cat_payload)

    fig_target, target_dist = tabular_eda.fig_target_pie(df)
    fig_to_json_file(fig_target, out_dir / "fig_target.json")
    write_json(out_dir / "target_dist.json", target_dist)

    corr_fig, numerical_cols, high_corrs = tabular_eda.fig_corr_matrix(df)
    if corr_fig:
        fig_to_json_file(corr_fig, out_dir / "fig_corr.json")
    else:
        write_json(out_dir / "fig_corr.json", None)
    write_json(out_dir / "high_corrs.json", {"numerical_cols": numerical_cols, "high_corrs": high_corrs})

    fig_outlier, outlier_stats = tabular_eda.outlier_iqr_release_year(df)
    fig_to_json_file(fig_outlier, out_dir / "fig_outlier.json")
    write_json(out_dir / "outlier_stats.json", outlier_stats)

    print("  Computing target relationships...")
    rel_results = tabular_eda.fig_target_relationships(df)
    rel_payload = {}
    for feature, (fig, stats) in rel_results.items():
        rel_payload[feature] = {
            "fig": json.loads(fig.to_json()),
            "stats": stats,
        }
    write_json(out_dir / "rel_figs.json", rel_payload)

    samples = tabular_eda.sample_rows(df, n=3)
    write_json(out_dir / "samples.json", samples)

    print("Tabular EDA export complete.")


# ---------------------------------------------------------------------------
# 3. Image EDA Export
# ---------------------------------------------------------------------------

def export_image(out_dir: Path) -> None:
    print("\n=== Image EDA ===")
    ensure_dir(out_dir)

    image_eda = _load_eda("Image")

    image_df = image_eda.load_image_statistic()
    bbox_df = image_eda.load_bbox_statistic()
    spatial_df = image_eda.load_spatial_distribution()
    quality_df = image_eda.load_quality_metrics()
    pixel_df = image_eda.load_pixel_distribution()
    shape_df = image_eda.load_shape_distribution()
    mask_df = image_eda.load_mask_statistics()
    segq_df = image_eda.load_boundary_quality()
    tsne_df = image_eda.load_tsne_embeddings()

    labels_df = image_eda.load_class_labels_txt()
    classes_df = image_eda.load_classes_txt()
    split_df = image_eda.load_train_val_test_split_txt()

    overview = image_eda.get_overview(image_df)
    write_json(out_dir / "overview.json", {
        "total_images": overview.total_images,
        "total_breeds": overview.total_breeds,
        "splits": overview.splits,
        "part_locations": overview.part_locations,
        "binary_attributes": overview.binary_attributes,
        "bounding_box": overview.bounding_box,
        "avg_width": round(overview.avg_width, 2),
        "avg_height": round(overview.avg_height, 2),
        "avg_aspect_ratio": round(overview.avg_aspect_ratio, 4),
        "avg_file_size_kb": round(overview.avg_file_size_kb, 2),
        "avg_brightness": round(overview.avg_brightness, 2),
        "avg_contrast": round(overview.avg_contrast, 2),
        "avg_sharpness": round(overview.avg_sharpness, 2),
    })

    # Part 1
    fig_to_json_file(image_eda.fig_split_bar(image_df), out_dir / "fig_split.json")
    fig_to_json_file(image_eda.fig_p1_size_scatter(image_df), out_dir / "fig_p1_size_scatter.json")
    fig_to_json_file(image_eda.fig_p1_file_size_hist(image_df), out_dir / "fig_p1_file_size_hist.json")
    fig_to_json_file(image_eda.fig_p1_aspect_ratio_hist(image_df), out_dir / "fig_p1_aspect_ratio_hist.json")
    fig_to_json_file(image_eda.fig_p1_rgb_3d(image_df), out_dir / "fig_p1_rgb_3d.json")
    fig_to_json_file(image_eda.fig_p1_sharp_vs_contrast(image_df), out_dir / "fig_p1_sharp_vs_contrast.json")
    fig_to_json_file(image_eda.fig_p1_rgb_box(image_df), out_dir / "fig_p1_rgb_box.json")
    fig_to_json_file(image_eda.fig_p1_brightness_hist(image_df), out_dir / "fig_p1_brightness_hist.json")
    fig_to_json_file(image_eda.fig_p1_contrast_hist(image_df), out_dir / "fig_p1_contrast_hist.json")

    # Part 2
    fig_to_json_file(image_eda.fig_p2_bbox_mean_median_bars(bbox_df), out_dir / "fig_p2_bbox_mean_median_bars.json")
    fig_to_json_file(image_eda.fig_p2_bbox_ar_donut(bbox_df), out_dir / "fig_p2_bbox_ar_donut.json")
    fig_to_json_file(image_eda.fig_p2_bbox_area_hist(bbox_df), out_dir / "fig_p2_bbox_area_hist.json")
    fig_to_json_file(image_eda.fig_p2_bbox_wh_scatter(bbox_df), out_dir / "fig_p2_bbox_wh_scatter.json")
    fig_to_json_file(image_eda.fig_p2_spatial_heatmap(spatial_df), out_dir / "fig_p2_spatial_heatmap.json")
    fig_to_json_file(image_eda.fig_p2_spatial_dist_hist(spatial_df), out_dir / "fig_p2_spatial_dist_hist.json")
    fig_to_json_file(image_eda.fig_p2_quality_area_cv_barh(quality_df), out_dir / "fig_p2_quality_area_cv_barh.json")
    fig_to_json_file(image_eda.fig_p2_quality_coverage_barh(quality_df), out_dir / "fig_p2_quality_coverage_barh.json")

    # Part 3
    fig_to_json_file(image_eda.fig_p3_pixel_donut(pixel_df), out_dir / "fig_p3_pixel_donut.json")
    fig_to_json_file(image_eda.fig_p3_pixel_mean_std_bars(pixel_df), out_dir / "fig_p3_pixel_mean_std_bars.json")
    fig_to_json_file(image_eda.fig_p3_shape_metrics_bars(shape_df), out_dir / "fig_p3_shape_metrics_bars.json")
    fig_to_json_file(image_eda.fig_p3_boundary_thickness_bar(mask_df), out_dir / "fig_p3_boundary_thickness_bar.json")
    fig_to_json_file(image_eda.fig_p3_boundary_smoothness_bar(mask_df), out_dir / "fig_p3_boundary_smoothness_bar.json")
    fig_to_json_file(image_eda.fig_p3_boundary_complexity_bar(mask_df), out_dir / "fig_p3_boundary_complexity_bar.json")
    fig_to_json_file(image_eda.fig_p3_segq_coverage_barh(segq_df), out_dir / "fig_p3_segq_coverage_barh.json")
    fig_to_json_file(image_eda.fig_p3_segq_coverage_cv_barh(segq_df), out_dir / "fig_p3_segq_coverage_cv_barh.json")
    fig_to_json_file(image_eda.fig_p3_segq_fg_cv_barh(segq_df), out_dir / "fig_p3_segq_fg_cv_barh.json")

    # Part 4
    fig_to_json_file(image_eda.fig_p4_class_hist(labels_df), out_dir / "fig_p4_class_hist.json")
    fig_to_json_file(image_eda.fig_p4_top10_classes_barh(labels_df, classes_df), out_dir / "fig_p4_top10_classes_barh.json")
    fig_to_json_file(image_eda.fig_p4_split_counts_bar(split_df), out_dir / "fig_p4_split_counts_bar.json")
    fig_to_json_file(image_eda.fig_p4_split_ratio_pie(split_df), out_dir / "fig_p4_split_ratio_pie.json")
    fig_to_json_file(image_eda.fig_p4_tsne_scatter(tsne_df), out_dir / "fig_p4_tsne_scatter.json")
    fig_to_json_file(image_eda.fig_p4_umap_scatter(tsne_df), out_dir / "fig_p4_umap_scatter.json")
    fig_to_json_file(image_eda.fig_p4_pca_cumvar(tsne_df), out_dir / "fig_p4_pca_cumvar.json")
    # Skipping similarity_heatmap: the source CSV is ~790KB and produces a very large JSON

    print("Image EDA export complete.")


# ---------------------------------------------------------------------------
# 4. Copy static assets
# ---------------------------------------------------------------------------

def copy_static(docs_dir: Path) -> None:
    print("\n=== Static Assets ===")
    src = APP_DIR / "static" / "assignment1.styles.css"
    dst = docs_dir / "static" / "assignment1.styles.css"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  ✓ {dst.relative_to(ROOT)}")

    src2 = APP_DIR / "static" / "styles.css"
    if src2.exists():
        dst2 = docs_dir / "static" / "styles.css"
        shutil.copy2(src2, dst2)
        print(f"  ✓ {dst2.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Project root : {ROOT}")
    print(f"Output dir   : {DOCS_DIR}")

    copy_static(DOCS_DIR)
    export_text(DOCS_DIR / "assignment1" / "text" / "data")
    export_tabular(DOCS_DIR / "assignment1" / "tabular" / "data")
    export_image(DOCS_DIR / "assignment1" / "image" / "data")

    print("\n✅ All data exported to docs/")
    print("   Next: open docs/index.html in a browser to verify.")
