from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


HERE = Path(__file__).resolve().parent
CSV_DIR = HERE / "csv"
TXT_DIR = HERE / "txt"


@dataclass(frozen=True)
class ImageOverview:
    total_images: int
    total_breeds: int
    splits: Dict[str, int]
    part_locations: int
    binary_attributes: int
    bounding_box: int
    avg_width: float
    avg_height: float
    avg_aspect_ratio: float
    avg_file_size_kb: float
    avg_brightness: float
    avg_contrast: float
    avg_sharpness: float


def _read_csv(name: str) -> pd.DataFrame:
    path = CSV_DIR / name
    return pd.read_csv(path)


@lru_cache(maxsize=2)
def load_image_statistic() -> pd.DataFrame:
    return _read_csv("image_statistic.csv")


@lru_cache(maxsize=2)
def load_bbox_statistic() -> pd.DataFrame:
    return _read_csv("bbox_statistic.csv")


@lru_cache(maxsize=2)
def load_mask_statistics() -> pd.DataFrame:
    return _read_csv("mask_statistics.csv")


@lru_cache(maxsize=2)
def load_quality_metrics() -> pd.DataFrame:
    return _read_csv("quality_metrics.csv")


@lru_cache(maxsize=2)
def load_pixel_distribution() -> pd.DataFrame:
    return _read_csv("pixel_distribution.csv")


@lru_cache(maxsize=2)
def load_shape_distribution() -> pd.DataFrame:
    return _read_csv("shape_distribution.csv")


@lru_cache(maxsize=2)
def load_boundary_quality() -> pd.DataFrame:
    return _read_csv("segmentation_boundary_quality_metric.csv")


@lru_cache(maxsize=2)
def load_spatial_distribution() -> pd.DataFrame:
    return _read_csv("spatial_distribution_analytics.csv")


@lru_cache(maxsize=2)
def load_tsne_embeddings() -> pd.DataFrame:
    return _read_csv("tsne_embeddings.csv")


@lru_cache(maxsize=2)
def load_similarity_matrix() -> pd.DataFrame:
    return _read_csv("similarity_matrix.csv")


@lru_cache(maxsize=2)
def load_class_labels_txt() -> pd.DataFrame:
    # columns: image_idx class_idx
    path = TXT_DIR / "image_class_labels.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=["image_idx", "class_idx"])


@lru_cache(maxsize=2)
def load_classes_txt() -> pd.DataFrame:
    path = TXT_DIR / "classes.txt"
    # format: class_idx class_name
    return pd.read_csv(path, sep=r"\s+", header=None, names=["class_idx", "class_name"], engine="python")


@lru_cache(maxsize=2)
def load_train_val_test_split_txt() -> pd.DataFrame:
    path = TXT_DIR / "train_val_test_split.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=["image_idx", "split"])


def get_overview(image_df: pd.DataFrame) -> ImageOverview:
    splits = image_df["split"].value_counts().to_dict()
    parts_dir = HERE / "parts"
    if parts_dir.exists() and parts_dir.is_dir():
        try:
            part_locations = len([p for p in parts_dir.iterdir() if p.is_file()])
        except Exception:
            part_locations = 15
    else:
        part_locations = 15

    attrs_dir = HERE / "attributes"
    if attrs_dir.exists() and attrs_dir.is_dir():
        try:
            binary_attributes = len([p for p in attrs_dir.iterdir() if p.is_file()])
        except Exception:
            binary_attributes = 312
    else:
        binary_attributes = 312

    bbox_path = CSV_DIR / "bbox_statistic.csv"
    bounding_box_val = 1
    try:
        bbox_df = pd.read_csv(bbox_path)
        boxes_per_image = bbox_df.groupby("image_id").size()
        mean_boxes = float(boxes_per_image.mean()) if len(boxes_per_image) else 0.0
        bounding_box_val = int(round(mean_boxes)) if mean_boxes >= 0.5 else int(mean_boxes)
    except Exception:
        bounding_box_val = 1

    return ImageOverview(
        total_images=int(len(image_df)),
        total_breeds=int(image_df["breed"].nunique()),
        splits={k: int(v) for k, v in splits.items()},
        part_locations=int(part_locations),
        binary_attributes=int(binary_attributes),
        bounding_box=int(bounding_box_val),
        avg_width=float(image_df["width"].mean()),
        avg_height=float(image_df["height"].mean()),
        avg_aspect_ratio=float(image_df["aspect_ratio"].mean()),
        avg_file_size_kb=float(image_df["file_size_kb"].mean()),
        avg_brightness=float(image_df["brightness"].mean()),
        avg_contrast=float(image_df["contrast"].mean()),
        avg_sharpness=float(image_df["sharpness"].mean()),
    )


def fig_split_bar(image_df: pd.DataFrame) -> go.Figure:
    counts = image_df["split"].value_counts().reset_index()
    counts.columns = ["split", "count"]
    fig = px.bar(
        counts,
        x="split",
        y="count",
        text="count",
        title="<b>Train / Val / Test split</b>",
        template="plotly_white",
        color="split",
        color_discrete_sequence=["#90caf9", "#a5d6a7", "#fff59d"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_image_size_hist(image_df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        image_df,
        x="area",
        nbins=40,
        color="split",
        opacity=0.75,
        title="<b>Image Area Distribution</b>",
        template="plotly_white",
    )
    fig.update_layout(height=460, margin=dict(l=40, r=20, t=70, b=50))
    return fig


# -----------------------------------------------------------------------------
# Part 1 figures (Core)
# -----------------------------------------------------------------------------


def fig_p1_size_scatter(image_df: pd.DataFrame) -> go.Figure:
    # Similar intent as notebook matplotlib scatter (width vs height, colored by top breeds)
    top_breeds = image_df["breed"].value_counts().head(8).index.tolist()
    df = image_df.copy()
    df["breed_group"] = df["breed"].where(df["breed"].isin(top_breeds), other="Other breeds")
    fig = px.scatter(
        df,
        x="width",
        y="height",
        color="breed_group",
        title="<b>Image Size Distribution (Width × Height)</b>",
        template="plotly_white",
        opacity=0.65,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p1_file_size_hist(image_df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        image_df,
        x="file_size_kb",
        nbins=40,
        color="split",
        opacity=0.75,
        title="<b>File Size Distribution</b>",
        template="plotly_white",
    )
    fig.update_layout(height=460, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p1_aspect_ratio_hist(image_df: pd.DataFrame) -> go.Figure:
    # Match notebook intent: Top breeds + Others overlay histogram
    df = image_df.dropna(subset=["aspect_ratio", "breed"]).copy()
    top_breeds = df["breed"].value_counts().head(8).index.tolist()
    df["breed_group"] = df["breed"].where(df["breed"].isin(top_breeds), other="Other breeds")

    # Use consistent, readable palette similar to matplotlib tab10
    fig = px.histogram(
        df,
        x="aspect_ratio",
        color="breed_group",
        nbins=35,
        opacity=0.55,
        title="<b>Aspect Ratio Distribution (Top Breeds + Others)</b>",
        template="plotly_white",
    )

    # Common ratio guide lines + labels (as in notebook)
    common_ratios = [0.75, 1.0, 1.33, 1.5, 2.0]
    for r in common_ratios:
        fig.add_vline(x=r, line_width=1, line_dash="dash", line_color="red", opacity=0.6)

    # Place text slightly below top of plot area
    fig.update_layout(
        height=480,
        margin=dict(l=40, r=20, t=70, b=70),
        legend_title_text="",
        barmode="overlay",
    )
    fig.update_yaxes(title="Count")
    fig.update_xaxes(title="Aspect Ratio (Width/Height)")

    # Add annotations for each guideline
    # (yref='paper' keeps labels visible regardless of scale)
    for r in common_ratios:
        fig.add_annotation(
            x=r,
            y=0.98,
            yref="paper",
            text=f"{r}:1",
            showarrow=False,
            font=dict(color="red", size=10),
            xanchor="center",
        )

    return fig


def fig_p1_rgb_3d(image_df: pd.DataFrame) -> go.Figure:
    df = image_df.dropna(subset=["mean_r", "mean_g", "mean_b", "brightness"]).copy()
    df_small = df.sample(min(4000, len(df)), random_state=42)
    fig = px.scatter_3d(
        df_small,
        x="mean_r",
        y="mean_g",
        z="mean_b",
        color="brightness",
        color_continuous_scale="Viridis",
        title="<b>Color Space Distribution (RGB)</b>",
    )
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=70, b=10))
    return fig


def fig_p1_sharp_vs_contrast(image_df: pd.DataFrame) -> go.Figure:
    df = image_df.dropna(subset=["sharpness", "contrast"]).copy()
    df_small = df.sample(min(6000, len(df)), random_state=42)
    fig = px.scatter(
        df_small,
        x="sharpness",
        y="contrast",
        title="<b>Image Quality Metrics (Sharpness vs Contrast)</b>",
        template="plotly_white",
        opacity=0.55,
    )
    fig.update_traces(marker=dict(size=5, color="#3b82f6"))
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p1_rgb_box(image_df: pd.DataFrame) -> go.Figure:
    df = image_df.dropna(subset=["mean_r", "mean_g", "mean_b"]).copy()
    long_df = df[["mean_r", "mean_g", "mean_b"]].melt(var_name="channel", value_name="value")
    channel_map = {"mean_r": "Red", "mean_g": "Green", "mean_b": "Blue"}
    long_df["channel"] = long_df["channel"].map(channel_map)
    fig = px.box(
        long_df,
        x="channel",
        y="value",
        points=False,
        title="<b>RGB Channel Value Distribution</b>",
        template="plotly_white",
        color="channel",
        color_discrete_map={"Red": "#ef4444", "Green": "#22c55e", "Blue": "#3b82f6"},
    )
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=50), showlegend=False)
    fig.update_yaxes(range=[0, 255])
    return fig


def fig_p1_brightness_hist(image_df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        image_df,
        x="brightness",
        nbins=35,
        title="<b>Brightness Distribution</b>",
        template="plotly_white",
    )
    fig.update_traces(marker_color="#f59e0b")
    fig.update_layout(height=460, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p1_contrast_hist(image_df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        image_df,
        x="contrast",
        nbins=35,
        title="<b>Contrast Distribution</b>",
        template="plotly_white",
    )
    fig.update_traces(marker_color="#3b82f6")
    fig.update_layout(height=460, margin=dict(l=40, r=20, t=70, b=50))
    return fig


# -----------------------------------------------------------------------------
# Part 2 figures (Detection)
# -----------------------------------------------------------------------------


def fig_p2_bbox_mean_median_bars(bbox_df: pd.DataFrame) -> go.Figure:
    width_mean = float(bbox_df["width"].mean())
    width_med = float(bbox_df["width"].median())
    height_mean = float(bbox_df["height"].mean())
    height_med = float(bbox_df["height"].median())

    fig = go.Figure()
    fig.add_bar(name="Width (px)", x=["Mean", "Median"], y=[width_mean, width_med], marker_color="#3b82f6")
    fig.add_bar(name="Height (px)", x=["Mean", "Median"], y=[height_mean, height_med], marker_color="#10b981")
    fig.update_layout(
        barmode="group",
        title="<b>Bounding Box Size Statistics</b>",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=70, b=50),
    )
    return fig


def fig_p2_bbox_ar_donut(bbox_df: pd.DataFrame) -> go.Figure:
    ar = (bbox_df["width"] / bbox_df["height"].replace(0, pd.NA)).astype(float)

    def cat(v: float) -> str:
        if v < 0.9:
            return "portrait"
        if v > 1.1:
            return "landscape"
        return "square"

    cats = ar.dropna().apply(cat)
    counts = cats.value_counts().reindex(["landscape", "square", "portrait"]).dropna()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=[c.capitalize() for c in counts.index.tolist()],
                values=counts.values.tolist(),
                hole=0.45,
                marker=dict(colors=["#f59e0b", "#10b981", "#ef4444"][: len(counts)]),
            )
        ]
    )
    fig.update_layout(
        title="<b>Aspect Ratio Distribution</b>",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=70, b=50),
    )
    return fig


def fig_p2_bbox_area_hist(bbox_df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        bbox_df,
        x="area",
        nbins=60,
        title="<b>Bounding Box Area Distribution</b>",
        template="plotly_white",
    )
    fig.update_traces(marker_color="#3b82f6")
    fig.update_layout(height=460, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p2_bbox_wh_scatter(bbox_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        bbox_df.sample(min(8000, len(bbox_df)), random_state=42),
        x="width",
        y="height",
        color="size_category",
        title="<b>Width vs Height Distribution (by size_category)</b>",
        template="plotly_white",
        opacity=0.65,
        color_discrete_map={"small": "#f59e0b", "medium": "#10b981", "large": "#3b82f6"},
    )
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p2_spatial_heatmap(spatial_df: pd.DataFrame) -> go.Figure:
    df = spatial_df.dropna(subset=["center_x", "center_y"]).copy()
    fig = px.density_heatmap(
        df,
        x="center_x",
        y="center_y",
        nbinsx=30,
        nbinsy=30,
        color_continuous_scale="Hot",
        title="<b>BBox Center Position Heatmap</b>",
        template="plotly_white",
    )
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=50))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
    return fig


def fig_p2_spatial_dist_hist(spatial_df: pd.DataFrame) -> go.Figure:
    df = spatial_df.dropna(subset=["center_x", "center_y"]).copy()
    df = df[df["center_x"].between(0, 1) & df["center_y"].between(0, 1)]
    df["dist_center"] = ((df["center_x"] - 0.5) ** 2 + (df["center_y"] - 0.5) ** 2) ** 0.5
    mean_d = float(df["dist_center"].mean())
    fig = px.histogram(
        df,
        x="dist_center",
        nbins=50,
        title="<b>Center Bias Distribution</b>",
        template="plotly_white",
    )
    fig.update_traces(marker_color="#3b82f6")
    fig.add_vline(x=mean_d, line_dash="dash", line_color="red", opacity=0.8)
    fig.update_layout(height=460, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p2_quality_area_cv_barh(quality_df: pd.DataFrame) -> go.Figure:
    top = quality_df.nlargest(15, "count").sort_values("count", ascending=True)
    fig = px.bar(
        top,
        x="area_cv",
        y="breed",
        orientation="h",
        title="<b>Size Consistency (Area CV)</b>",
        template="plotly_white",
    )
    fig.update_traces(marker_color="#3b82f6")
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p2_quality_coverage_barh(quality_df: pd.DataFrame) -> go.Figure:
    top = quality_df.nlargest(15, "count").sort_values("count", ascending=True)
    fig = px.bar(
        top,
        x="avg_coverage",
        y="breed",
        orientation="h",
        title="<b>Image Coverage</b>",
        template="plotly_white",
    )
    fig.update_traces(marker_color="#10b981")
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=50))
    return fig


# -----------------------------------------------------------------------------
# Part 3 figures (Segmentation)
# -----------------------------------------------------------------------------


def fig_p3_pixel_donut(pixel_df: pd.DataFrame) -> go.Figure:
    fg = float(pixel_df["fg_percentage_mean"].mean())
    bd = float(pixel_df["boundary_percentage_mean"].mean())
    bg = float(pixel_df["bg_percentage_mean"].mean())
    total = fg + bd + bg
    vals = [bg, fg, bd]
    if total > 0:
        vals = [v * 100.0 / total for v in vals]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Background", "Foreground", "Boundary"],
                values=vals,
                hole=0.55,
                marker=dict(colors=["#3b82f6", "#ef4444", "#f59e0b"]),
            )
        ]
    )
    fig.update_layout(
        title="<b>Pixel Class Distribution</b>",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=70, b=50),
    )
    return fig


def fig_p3_pixel_mean_std_bars(pixel_df: pd.DataFrame) -> go.Figure:
    means = [
        float(pixel_df["fg_percentage_mean"].mean()),
        float(pixel_df["boundary_percentage_mean"].mean()),
        float(pixel_df["bg_percentage_mean"].mean()),
    ]
    stds = [
        float(pixel_df["fg_percentage_mean"].std(ddof=0)),
        float(pixel_df["boundary_percentage_mean"].std(ddof=0)),
        float(pixel_df["bg_percentage_mean"].std(ddof=0)),
    ]
    classes = ["Foreground", "Boundary", "Background"]
    fig = go.Figure()
    fig.add_bar(name="Mean (%)", x=classes, y=means, marker_color="#3b82f6")
    fig.add_bar(name="Std Dev (%)", x=classes, y=stds, marker_color="#10b981")
    fig.update_layout(
        barmode="group",
        title="<b>Class Distribution Statistics</b>",
        template="plotly_white",
        height=460,
        margin=dict(l=40, r=20, t=70, b=50),
        yaxis_title="Percentage (%)",
    )
    return fig


def fig_p3_shape_metrics_bars(shape_df: pd.DataFrame) -> go.Figure:
    means = {
        "Convexity": float(shape_df["convexity_mean"].mean()),
        "Compactness": float(shape_df["compactness_mean"].mean()),
        "Eccentricity": float(shape_df["eccentricity_mean"].mean()),
    }
    stds = {
        "Convexity": float(shape_df["convexity_mean"].std(ddof=0)),
        "Compactness": float(shape_df["compactness_mean"].std(ddof=0)),
        "Eccentricity": float(shape_df["eccentricity_mean"].std(ddof=0)),
    }
    x = list(means.keys())
    fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=[means[k] for k in x],
                error_y=dict(type="data", array=[stds[k] for k in x], visible=True),
                marker_color=["#8b5cf6", "#7c3aed", "#a855f7"],
            )
        ]
    )
    fig.update_layout(
        title="<b>Shape Metrics</b>",
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=20, t=70, b=50),
        yaxis_range=[0, min(1.0, max(means.values()) + max(stds.values()) + 0.15)],
    )
    return fig


def _boundary_features(mask_df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    img_perimeter = 2.0 * (mask_df["mask_height"] + mask_df["mask_width"])
    out = pd.DataFrame(index=mask_df.index)
    out["thickness"] = mask_df["boundary_pixels"] / img_perimeter.clip(lower=eps)
    out["smoothness"] = 1.0 - (
        mask_df["boundary_pixels"]
        / (mask_df["fg_pixels"] + mask_df["boundary_pixels"]).clip(lower=eps)
    )
    out["smoothness"] = out["smoothness"].clip(0.0, 1.0)
    out["complexity"] = mask_df["boundary_pixels"] / (mask_df["fg_pixels"].clip(lower=1.0) ** 0.5)
    return out


def fig_p3_boundary_thickness_bar(mask_df: pd.DataFrame) -> go.Figure:
    f = _boundary_features(mask_df)
    bins = [0, 1, 2, 3, 4, 5, 1e18]
    labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5+"]
    counts = pd.cut(f["thickness"], bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
    fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        title="<b>Boundary Thickness</b>",
        template="plotly_white",
        labels={"x": "Thickness Range (pixels)", "y": "Count"},
    )
    fig.update_traces(marker_color="#ef4444")
    fig.update_layout(height=460, margin=dict(l=40, r=20, t=70, b=80))
    return fig


def fig_p3_boundary_smoothness_bar(mask_df: pd.DataFrame) -> go.Figure:
    f = _boundary_features(mask_df)
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    counts = pd.cut(f["smoothness"], bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
    fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        title="<b>Boundary Smoothness</b>",
        template="plotly_white",
        labels={"x": "Smoothness Range", "y": "Count"},
    )
    fig.update_traces(marker_color="#10b981")
    fig.update_layout(height=460, margin=dict(l=40, r=20, t=70, b=80))
    return fig


def fig_p3_boundary_complexity_bar(mask_df: pd.DataFrame) -> go.Figure:
    f = _boundary_features(mask_df)
    bins = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 1e18]
    labels = ["0-1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0+"]
    counts = pd.cut(f["complexity"], bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
    fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        title="<b>Boundary Complexity</b>",
        template="plotly_white",
        labels={"x": "Complexity Range", "y": "Count"},
    )
    fig.update_traces(marker_color="#f59e0b")
    fig.update_layout(height=460, margin=dict(l=40, r=20, t=70, b=80))
    return fig


def fig_p3_segq_coverage_barh(segq_df: pd.DataFrame) -> go.Figure:
    top = segq_df.nlargest(15, "count").sort_values("count", ascending=True)
    fig = px.bar(
        top,
        x="avg_coverage",
        y="breed",
        orientation="h",
        title="<b>Mask Coverage (avg_coverage)</b>",
        template="plotly_white",
    )
    fig.update_traces(marker_color="#3b82f6")
    fig.update_layout(height=560, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p3_segq_coverage_cv_barh(segq_df: pd.DataFrame) -> go.Figure:
    top = segq_df.nlargest(15, "count").sort_values("count", ascending=True)
    fig = px.bar(
        top,
        x="coverage_cv",
        y="breed",
        orientation="h",
        title="<b>Coverage Consistency (coverage_cv)</b>",
        template="plotly_white",
    )
    fig.update_traces(marker_color="#10b981")
    fig.update_layout(height=560, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p3_segq_fg_cv_barh(segq_df: pd.DataFrame) -> go.Figure:
    top = segq_df.nlargest(15, "count").sort_values("count", ascending=True)
    fig = px.bar(
        top,
        x="fg_cv",
        y="breed",
        orientation="h",
        title="<b>Foreground Consistency (fg_cv)</b>",
        template="plotly_white",
    )
    fig.update_traces(marker_color="#f59e0b")
    fig.update_layout(height=560, margin=dict(l=40, r=20, t=70, b=50))
    return fig


# -----------------------------------------------------------------------------
# Part 4 figures (Classification)
# -----------------------------------------------------------------------------


def fig_p4_class_hist(labels_df: pd.DataFrame) -> go.Figure:
    class_counts = labels_df["class_idx"].value_counts().sort_index()
    fig = px.histogram(
        x=class_counts.values,
        nbins=20,
        title="<b>Samples per Class (Histogram)</b>",
        template="plotly_white",
        labels={"x": "Images per class", "y": "Number of classes"},
    )
    fig.update_traces(marker_color="#3b82f6")
    fig.add_vline(x=float(class_counts.mean()), line_dash="dash", line_color="red")
    fig.update_layout(height=460, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p4_top10_classes_barh(labels_df: pd.DataFrame, classes_df: pd.DataFrame) -> go.Figure:
    class_counts = labels_df["class_idx"].value_counts().sort_index()
    top = class_counts.sort_values(ascending=False).head(10)
    mapper = classes_df.set_index("class_idx")["class_name"].astype(str)
    names = [mapper.get(i, str(i)) for i in top.index.tolist()]
    fig = px.bar(
        x=top.values[::-1],
        y=names[::-1],
        orientation="h",
        title="<b>Top-10 Classes by Count</b>",
        template="plotly_white",
        labels={"x": "Images", "y": "Class"},
    )
    fig.update_traces(marker_color="#10b981")
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p4_split_counts_bar(split_df: pd.DataFrame) -> go.Figure:
    df = split_df.copy()
    df["split"] = df["split"].astype(str).str.lower()
    counts = df["split"].value_counts().reset_index()
    counts.columns = ["split", "count"]
    fig = px.bar(
        counts,
        x="split",
        y="count",
        text="count",
        title="<b>Split Counts</b>",
        template="plotly_white",
        color="split",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_p4_split_ratio_pie(split_df: pd.DataFrame) -> go.Figure:
    df = split_df.copy()
    df["split"] = df["split"].astype(str).str.lower()
    counts = df["split"].value_counts()
    ratio = counts / counts.sum()
    labels = [f"{k} ({ratio[k]*100:.1f}%)" for k in counts.index]
    fig = go.Figure(data=[go.Pie(labels=labels, values=counts.values.tolist())])
    fig.update_layout(
        title="<b>Split Ratio</b>",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=70, b=50),
    )
    return fig


def fig_p4_tsne_scatter(tsne_df: pd.DataFrame) -> go.Figure:
    df = tsne_df.dropna(subset=["tsne_x", "tsne_y", "breed"]).copy()
    df_small = df.sample(min(6000, len(df)), random_state=42)
    fig = px.scatter(
        df_small,
        x="tsne_x",
        y="tsne_y",
        color="breed",
        title="<b>t-SNE: Breed Features in 2D</b>",
        template="plotly_white",
        opacity=0.75,
    )
    fig.update_layout(height=650, margin=dict(l=40, r=20, t=70, b=50), showlegend=False)
    return fig


def fig_p4_umap_scatter(tsne_df: pd.DataFrame) -> go.Figure:
    # Notebook computes UMAP from t-SNE coords + one-hot. We'll do a light PCA fallback always.
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df = tsne_df.dropna(subset=["tsne_x", "tsne_y", "breed"]).copy()
    df["breed"] = df["breed"].astype(str)
    breed_ohe = pd.get_dummies(df["breed"], dtype=float)
    X = pd.concat([df[["tsne_x", "tsne_y"]].astype(float), breed_ohe], axis=1).to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    xy = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    df["umap_x"] = xy[:, 0]
    df["umap_y"] = xy[:, 1]
    df_small = df.sample(min(6000, len(df)), random_state=42)
    fig = px.scatter(
        df_small,
        x="umap_x",
        y="umap_y",
        color="breed",
        title="<b>UMAP unavailable - PCA 2D fallback</b>",
        template="plotly_white",
        opacity=0.75,
    )
    fig.update_layout(height=650, margin=dict(l=40, r=20, t=70, b=50), showlegend=False)
    return fig


def fig_p4_pca_cumvar(tsne_df: pd.DataFrame) -> go.Figure:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df = tsne_df.dropna(subset=["tsne_x", "tsne_y", "breed"]).copy()
    df["breed"] = df["breed"].astype(str)
    breed_ohe = pd.get_dummies(df["breed"], dtype=float)
    X = pd.concat([df[["tsne_x", "tsne_y"]].astype(float), breed_ohe], axis=1).to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    max_components = min(50, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(X_scaled)
    cum_var = (pca.explained_variance_ratio_.cumsum() * 100.0)
    comp = list(range(1, max_components + 1))
    fig = px.line(
        x=comp,
        y=cum_var,
        markers=True,
        title="<b>PCA Explained Variance</b>",
        template="plotly_white",
        labels={"x": "Number of Components", "y": "Cumulative Variance (%)"},
    )
    fig.update_traces(line_color="#3b82f6")
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=50))
    fig.update_yaxes(range=[0, 100])
    return fig


def fig_p4_similarity_heatmap(sim_df: pd.DataFrame) -> go.Figure:
    return fig_similarity_heatmap(sim_df)


def fig_aspect_ratio_hist(image_df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        image_df,
        x="aspect_ratio",
        nbins=40,
        color="split",
        opacity=0.75,
        title="<b>Aspect Ratio Distribution</b>",
        template="plotly_white",
    )
    fig.update_layout(height=460, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_quality_scatter(image_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        image_df.sample(min(2500, len(image_df)), random_state=42),
        x="brightness",
        y="sharpness",
        color="split",
        title="<b>Brightness vs Sharpness (sampled)</b>",
        template="plotly_white",
        opacity=0.7,
    )
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_bbox_coverage_hist(bbox_df: pd.DataFrame) -> go.Figure:
    # coverage = bbox_area / image_area
    coverage = (bbox_df["area"] / (bbox_df["image_width"] * bbox_df["image_height"]))
    tmp = bbox_df.copy()
    tmp["coverage"] = coverage
    fig = px.histogram(
        tmp,
        x="coverage",
        nbins=40,
        color="split",
        opacity=0.75,
        title="<b>Bounding Box Coverage (bbox_area / image_area)</b>",
        template="plotly_white",
    )
    fig.update_layout(height=460, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_mask_pixel_distribution(pixel_df: pd.DataFrame) -> go.Figure:
    long_df = pixel_df.melt(
        id_vars=["breed", "count"],
        value_vars=["fg_percentage_mean", "boundary_percentage_mean", "bg_percentage_mean"],
        var_name="region",
        value_name="percentage",
    )
    region_map = {
        "fg_percentage_mean": "Foreground",
        "boundary_percentage_mean": "Boundary",
        "bg_percentage_mean": "Background",
    }
    long_df["region"] = long_df["region"].map(region_map)

    # show global distribution across breeds
    fig = px.box(
        long_df,
        x="region",
        y="percentage",
        points="all",
        title="<b>Mask pixel distribution across breeds</b>",
        template="plotly_white",
    )
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_tsne_scatter(tsne_df: pd.DataFrame) -> go.Figure:
    # limit points for performance
    df_small = tsne_df.sample(min(4000, len(tsne_df)), random_state=42)
    fig = px.scatter(
        df_small,
        x="tsne_x",
        y="tsne_y",
        color="breed",
        title="<b>t-SNE embeddings (sampled)</b>",
        template="plotly_white",
        opacity=0.75,
    )
    fig.update_layout(height=650, margin=dict(l=40, r=20, t=70, b=50))
    return fig


def fig_similarity_heatmap(sim_df: pd.DataFrame) -> go.Figure:
    # Expect first column maybe unnamed index; handle both
    if sim_df.columns[0].lower() in {"breed", "class", "label", "index"}:
        labels = sim_df.iloc[:, 0].astype(str).tolist()
        mat = sim_df.iloc[:, 1:].to_numpy()
        cols = sim_df.columns[1:].astype(str).tolist()
        if len(cols) == len(labels):
            xlabels = cols
        else:
            xlabels = labels
    else:
        labels = sim_df.columns.astype(str).tolist()
        mat = sim_df.to_numpy()
        xlabels = labels

    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=xlabels,
            y=labels,
            colorscale="Viridis",
            colorbar=dict(title="Similarity"),
        )
    )
    fig.update_layout(
        title="<b>Breed similarity matrix</b>",
        template="plotly_white",
        height=800,
        margin=dict(l=40, r=20, t=70, b=50),
    )
    return fig
