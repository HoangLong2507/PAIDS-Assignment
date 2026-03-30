from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .templates.assignment1.EDA.Text import eda
from .templates.assignment1.EDA.Tabular import eda as tabular_eda
from .templates.assignment1.EDA.Image import eda as image_eda


BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="CO3135 Projects")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Static pages (for scalable structure: assignment1/, assignment2/, ...)
REPO_ROOT = BASE_DIR.parent


@app.get("/", response_class=HTMLResponse)
def main_page(request: Request):
    return templates.TemplateResponse(
        "main/index.html",
        {
            "request": request,
        },
    )


@app.get("/assignment1", response_class=HTMLResponse)
def assignment1(request: Request):
    return FileResponse(str(REPO_ROOT / "app" / "templates" / "assignment1" / "index.html"))


@app.get("/assignment1/text", response_class=HTMLResponse)
def assignment1_text(request: Request):
    df = eda.load_df()

    overview = eda.get_overview(df)

    fig_bar = eda.fig_category_bar(df).to_html(include_plotlyjs=False, full_html=False)
    fig_pie = eda.fig_category_pie(df).to_html(include_plotlyjs=False, full_html=False)
    fig_lengths = eda.fig_length_distributions(df).to_html(
        include_plotlyjs=False, full_html=False
    )

    fig_stop, total_words, total_stop = eda.fig_stopwords_raw(df)
    fig_stop_html = fig_stop.to_html(include_plotlyjs=False, full_html=False)

    cleaned_text = eda.clean_text_series(df)

    stats_df = eda.vocab_richness(df, cleaned_text)
    vocab_table = stats_df.to_dict(orient="records")
    fig_vocab = eda.fig_vocab_richness(stats_df).to_html(
        include_plotlyjs=False, full_html=False
    )

    top50_figs = {
        k: v.to_html(include_plotlyjs=False, full_html=False)
        for k, v in eda.fig_top50_words_by_type(df, cleaned_text).items()
    }

    tfidf_figs = {
        k: v.to_html(include_plotlyjs=False, full_html=False)
        for k, v in eda.fig_tfidf_top_terms(df, cleaned_text).items()
    }

    bigram_figs = {
        k: v.to_html(include_plotlyjs=False, full_html=False)
        for k, v in eda.fig_bigrams(df, cleaned_text).items()
    }

    sim_fig, categories, sim = eda.fig_category_similarity(df)
    sim_html = sim_fig.to_html(include_plotlyjs=False, full_html=False)

    return templates.TemplateResponse(
        "/assignment1/EDA/Text/index.html",
        {
            "request": request,
            "overview": overview,
            "fig_bar": fig_bar,
            "fig_pie": fig_pie,
            "fig_lengths": fig_lengths,
            "fig_stop": fig_stop_html,
            "total_words": total_words,
            "total_stop": total_stop,
            "stop_pct": (total_stop / total_words * 100) if total_words else 0.0,
            "vocab_table": vocab_table,
            "fig_vocab": fig_vocab,
            "top50_figs": top50_figs,
            "tfidf_figs": tfidf_figs,
            "bigram_figs": bigram_figs,
            "sim_fig": sim_html,
            "categories": categories,
            "sim": sim,
        },
    )


@app.get("/assignment1/tabular", response_class=HTMLResponse)
def assignment1_tabular(request: Request):
    df = tabular_eda.load_df()

    overview = tabular_eda.get_overview(df)

    missing_df = tabular_eda.missing_values_table(df)
    missing_table = missing_df.to_dict(orient="records")
    fig_missing = tabular_eda.fig_missing_bar(missing_df).to_html(
        include_plotlyjs=False, full_html=False
    )

    fig_release, release_stats = tabular_eda.fig_release_year_distribution(df)
    fig_release_html = fig_release.to_html(include_plotlyjs=False, full_html=False)

    cat_results = tabular_eda.fig_categorical_top15(df)
    cat_payload = {
        feature: {
            "fig": fig.to_html(include_plotlyjs=False, full_html=False),
            "stats": stats,
        }
        for feature, (fig, stats) in cat_results.items()
    }

    fig_target, target_dist = tabular_eda.fig_target_pie(df)
    fig_target_html = fig_target.to_html(include_plotlyjs=False, full_html=False)

    corr_fig, numerical_cols, high_corrs = tabular_eda.fig_corr_matrix(df)
    corr_html = (
        corr_fig.to_html(include_plotlyjs=False, full_html=False) if corr_fig else None
    )

    fig_outlier, outlier_stats = tabular_eda.outlier_iqr_release_year(df)
    fig_outlier_html = fig_outlier.to_html(include_plotlyjs=False, full_html=False)

    rel_figs = {
        feature: fig.to_html(include_plotlyjs=False, full_html=False)
        for feature, fig in tabular_eda.fig_target_relationships(df).items()
    }

    samples = tabular_eda.sample_rows(df, n=3)

    return templates.TemplateResponse(
        "/assignment1/EDA/Tabular/index.html",
        {
            "request": request,
            "overview": overview,
            "missing_table": missing_table,
            "fig_missing": fig_missing,
            "fig_release_year": fig_release_html,
            "release_stats": release_stats,
            "cat_payload": cat_payload,
            "fig_target": fig_target_html,
            "target_dist": target_dist,
            "fig_corr": corr_html,
            "numerical_cols": numerical_cols,
            "high_corrs": high_corrs,
            "fig_outlier": fig_outlier_html,
            "outlier_stats": outlier_stats,
            "rel_figs": rel_figs,
            "samples": samples,
        },
    )


@app.get("/assignment1/image", response_class=HTMLResponse)
def assignment1_image(request: Request):
    image_df = image_eda.load_image_statistic()
    bbox_df = image_eda.load_bbox_statistic()
    spatial_df = image_eda.load_spatial_distribution()
    quality_df = image_eda.load_quality_metrics()
    pixel_df = image_eda.load_pixel_distribution()
    shape_df = image_eda.load_shape_distribution()
    mask_df = image_eda.load_mask_statistics()
    segq_df = image_eda.load_boundary_quality()
    tsne_df = image_eda.load_tsne_embeddings()
    sim_df = image_eda.load_similarity_matrix()

    labels_df = image_eda.load_class_labels_txt()
    classes_df = image_eda.load_classes_txt()
    split_df = image_eda.load_train_val_test_split_txt()

    overview = image_eda.get_overview(image_df)

    # Part 1
    fig_split = image_eda.fig_split_bar(image_df).to_html(include_plotlyjs=False, full_html=False)
    fig_p1_size_scatter = image_eda.fig_p1_size_scatter(image_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p1_file_size_hist = image_eda.fig_p1_file_size_hist(image_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p1_aspect_ratio_hist = image_eda.fig_p1_aspect_ratio_hist(image_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p1_rgb_3d = image_eda.fig_p1_rgb_3d(image_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p1_sharp_vs_contrast = image_eda.fig_p1_sharp_vs_contrast(image_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p1_rgb_box = image_eda.fig_p1_rgb_box(image_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p1_brightness_hist = image_eda.fig_p1_brightness_hist(image_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p1_contrast_hist = image_eda.fig_p1_contrast_hist(image_df).to_html(
        include_plotlyjs=False, full_html=False
    )

    # Part 2
    fig_p2_bbox_mean_median_bars = image_eda.fig_p2_bbox_mean_median_bars(bbox_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p2_bbox_ar_donut = image_eda.fig_p2_bbox_ar_donut(bbox_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p2_bbox_area_hist = image_eda.fig_p2_bbox_area_hist(bbox_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p2_bbox_wh_scatter = image_eda.fig_p2_bbox_wh_scatter(bbox_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p2_spatial_heatmap = image_eda.fig_p2_spatial_heatmap(spatial_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p2_spatial_dist_hist = image_eda.fig_p2_spatial_dist_hist(spatial_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p2_quality_area_cv_barh = image_eda.fig_p2_quality_area_cv_barh(quality_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p2_quality_coverage_barh = image_eda.fig_p2_quality_coverage_barh(quality_df).to_html(
        include_plotlyjs=False, full_html=False
    )

    # Part 3
    fig_p3_pixel_donut = image_eda.fig_p3_pixel_donut(pixel_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p3_pixel_mean_std_bars = image_eda.fig_p3_pixel_mean_std_bars(pixel_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p3_shape_metrics_bars = image_eda.fig_p3_shape_metrics_bars(shape_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p3_boundary_thickness_bar = image_eda.fig_p3_boundary_thickness_bar(mask_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p3_boundary_smoothness_bar = image_eda.fig_p3_boundary_smoothness_bar(mask_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p3_boundary_complexity_bar = image_eda.fig_p3_boundary_complexity_bar(mask_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p3_segq_coverage_barh = image_eda.fig_p3_segq_coverage_barh(segq_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p3_segq_coverage_cv_barh = image_eda.fig_p3_segq_coverage_cv_barh(segq_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p3_segq_fg_cv_barh = image_eda.fig_p3_segq_fg_cv_barh(segq_df).to_html(
        include_plotlyjs=False, full_html=False
    )

    # Part 4
    fig_p4_class_hist = image_eda.fig_p4_class_hist(labels_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p4_top10_classes_barh = image_eda.fig_p4_top10_classes_barh(labels_df, classes_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p4_split_counts_bar = image_eda.fig_p4_split_counts_bar(split_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p4_split_ratio_pie = image_eda.fig_p4_split_ratio_pie(split_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p4_tsne_scatter = image_eda.fig_p4_tsne_scatter(tsne_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p4_umap_scatter = image_eda.fig_p4_umap_scatter(tsne_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p4_pca_cumvar = image_eda.fig_p4_pca_cumvar(tsne_df).to_html(
        include_plotlyjs=False, full_html=False
    )
    fig_p4_similarity_heatmap = image_eda.fig_p4_similarity_heatmap(sim_df).to_html(
        include_plotlyjs=False, full_html=False
    )

    return templates.TemplateResponse(
        "/assignment1/EDA/Image/index.html",
        {
            "request": request,
            "overview": overview,
            "fig_split": fig_split,
            "fig_p1_size_scatter": fig_p1_size_scatter,
            "fig_p1_file_size_hist": fig_p1_file_size_hist,
            "fig_p1_aspect_ratio_hist": fig_p1_aspect_ratio_hist,
            "fig_p1_rgb_3d": fig_p1_rgb_3d,
            "fig_p1_sharp_vs_contrast": fig_p1_sharp_vs_contrast,
            "fig_p1_rgb_box": fig_p1_rgb_box,
            "fig_p1_brightness_hist": fig_p1_brightness_hist,
            "fig_p1_contrast_hist": fig_p1_contrast_hist,
            "fig_p2_bbox_mean_median_bars": fig_p2_bbox_mean_median_bars,
            "fig_p2_bbox_ar_donut": fig_p2_bbox_ar_donut,
            "fig_p2_bbox_area_hist": fig_p2_bbox_area_hist,
            "fig_p2_bbox_wh_scatter": fig_p2_bbox_wh_scatter,
            "fig_p2_spatial_heatmap": fig_p2_spatial_heatmap,
            "fig_p2_spatial_dist_hist": fig_p2_spatial_dist_hist,
            "fig_p2_quality_area_cv_barh": fig_p2_quality_area_cv_barh,
            "fig_p2_quality_coverage_barh": fig_p2_quality_coverage_barh,
            "fig_p3_pixel_donut": fig_p3_pixel_donut,
            "fig_p3_pixel_mean_std_bars": fig_p3_pixel_mean_std_bars,
            "fig_p3_shape_metrics_bars": fig_p3_shape_metrics_bars,
            "fig_p3_boundary_thickness_bar": fig_p3_boundary_thickness_bar,
            "fig_p3_boundary_smoothness_bar": fig_p3_boundary_smoothness_bar,
            "fig_p3_boundary_complexity_bar": fig_p3_boundary_complexity_bar,
            "fig_p3_segq_coverage_barh": fig_p3_segq_coverage_barh,
            "fig_p3_segq_coverage_cv_barh": fig_p3_segq_coverage_cv_barh,
            "fig_p3_segq_fg_cv_barh": fig_p3_segq_fg_cv_barh,
            "fig_p4_class_hist": fig_p4_class_hist,
            "fig_p4_top10_classes_barh": fig_p4_top10_classes_barh,
            "fig_p4_split_counts_bar": fig_p4_split_counts_bar,
            "fig_p4_split_ratio_pie": fig_p4_split_ratio_pie,
            "fig_p4_tsne_scatter": fig_p4_tsne_scatter,
            "fig_p4_umap_scatter": fig_p4_umap_scatter,
            "fig_p4_pca_cumvar": fig_p4_pca_cumvar,
            "fig_p4_similarity_heatmap": fig_p4_similarity_heatmap,
        },
    )
