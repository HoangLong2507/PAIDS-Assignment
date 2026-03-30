from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _resolve_data_path() -> Path:
    """Resolve `netflix_titles.csv` robustly.
    """

    here = Path(__file__).resolve().parent
    candidates = [
        here / "netflix_titles.csv",
        Path(__file__).resolve().parents[5] / "netflix_titles.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


DATA_PATH = _resolve_data_path()


@dataclass(frozen=True)
class DatasetOverview:
    total_samples: int
    total_features: int
    numerical_features: int
    categorical_features: int


@lru_cache(maxsize=2)
def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Match notebook behavior (kaggle dataset has same columns).
    # Parse date_added for optional analysis; keep original too.
    if "date_added" in df.columns:
        df["date_added_parsed"] = pd.to_datetime(df["date_added"], errors="coerce")

    return df


def get_overview(df: pd.DataFrame) -> DatasetOverview:
    total_samples = int(len(df))
    total_features = int(len(df.columns))
    numerical_features = int(len(df.select_dtypes(include=["number"]).columns))
    categorical_features = int(len(df.select_dtypes(exclude=["number"]).columns))

    return DatasetOverview(
        total_samples=total_samples,
        total_features=total_features,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.concat([missing, missing_pct], axis=1)
    missing_df.columns = ["Count", "Percent (%)"]

    missing_data = (
        missing_df[missing_df["Count"] > 0]
        .sort_values(by="Count", ascending=False)
        .reset_index()
        .rename(columns={"index": "Feature"})
    )
    return missing_data


def fig_missing_bar(missing_data: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        missing_data,
        x="Feature",
        y="Percent (%)",
        text="Percent (%)",
        title="<b>Missing Values Analysis (%)</b>",
        template="plotly_white",
        color_discrete_sequence=["#ff8a80"],
    )
    fig.update_traces(
        texttemplate="%{text}%",
        textposition="outside",
        marker_line_color="#d32f2f",
        marker_line_width=1.5,
    )
    fig.update_layout(
        title_font=dict(size=24, color="#d32f2f"),
        xaxis_title="<b>Features</b>",
        yaxis_title="<b>Missing Percentage (%)</b>",
        height=500,
        margin=dict(l=40, r=20, t=70, b=60),
    )
    return fig


def fig_release_year_distribution(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, float]]:
    feature = "release_year"
    df_clean = df[feature].dropna()

    counts, bin_edges = np.histogram(df_clean, bins=25)
    bin_labels = [
        f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}"
        for i in range(len(bin_edges) - 1)
    ]

    # notebook reverses display order
    counts = counts[::-1]
    bin_labels = bin_labels[::-1]

    fig = go.Figure(
        data=[
            go.Bar(
                x=bin_labels,
                y=counts,
                marker_color="#81d4fa",
                marker_line_color="#0288d1",
                marker_line_width=1.5,
                name="Count",
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="<b>Release Year Distribution</b>",
            font=dict(size=24, color="#0288d1"),
            x=0,
            y=0.95,
        ),
        xaxis=dict(title="Release Year", tickangle=-45),
        yaxis=dict(title="Count"),
        template="plotly_white",
        width=1000,
        height=500,
        margin=dict(l=40, r=20, t=70, b=80),
    )

    stats = {
        "mean": float(df[feature].mean()),
        "median": float(df[feature].median()),
        "std": float(df[feature].std()),
        "min": float(df[feature].min()),
        "max": float(df[feature].max()),
    }

    return fig, stats


def _value_counts_top15(
    df: pd.DataFrame, feature: str, split_comma_space: bool
) -> pd.Series:
    data_series = df[feature].dropna().astype(str)
    if split_comma_space:
        data_series = data_series.str.split(", ").explode()
    return data_series.value_counts().head(15)


def fig_categorical_top15(df: pd.DataFrame) -> Dict[str, Tuple[go.Figure, Dict[str, object]]]:
    """Return figures + per-feature stats for categorical EDA.

    Notebook logic:
    - choose object columns excluding show_id,title,description,type
    - for country/cast/director/listed_in -> split and explode
    - plot top15 bar
    - print missing + top categories
    """

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cols_to_plot = [c for c in cat_cols if c not in ["show_id", "title", "description", "type"]]

    pastel_colors = [
        "#90caf9",
        "#a5d6a7",
        "#fff59d",
        "#f48fb1",
        "#ce93d8",
        "#81deea",
        "#e6ee9c",
    ]
    border_colors = [
        "#1976d2",
        "#388e3c",
        "#fbc02d",
        "#c2185b",
        "#7b1fa2",
        "#0097a7",
        "#afb42b",
    ]

    results: Dict[str, Tuple[go.Figure, Dict[str, object]]] = {}

    for i, feature in enumerate(cols_to_plot):
        split_needed = feature in {"country", "cast", "director", "listed_in"}
        value_counts = _value_counts_top15(df, feature, split_needed)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=value_counts.index.tolist(),
                    y=value_counts.values.tolist(),
                    marker_color=pastel_colors[i % len(pastel_colors)],
                    marker_line_color=border_colors[i % len(border_colors)],
                    marker_line_width=1.5,
                    text=value_counts.values.tolist(),
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text=f"<b>{feature.replace('_', ' ').title()} Distribution</b>",
                font=dict(size=20, color="#333"),
            ),
            template="plotly_white",
            xaxis_title=feature.replace("_", " ").title(),
            yaxis_title="Count",
            width=1000,
            height=600,
            xaxis={"tickangle": 45},
            margin=dict(l=40, r=20, t=70, b=120),
        )

        missing_count = int(df[feature].isnull().sum())
        missing_pct = float(missing_count / len(df) * 100)

        top10 = [
            {
                "category": str(cat),
                "count": int(count),
                "pct": float(count / len(df) * 100),
            }
            for cat, count in value_counts.head(10).items()
        ]

        results[feature] = (
            fig,
            {
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "top10": top10,
            },
        )

    return results


def fig_target_pie(df: pd.DataFrame) -> Tuple[go.Figure, List[Dict[str, object]]]:
    target_counts = df["type"].value_counts()

    fig = px.pie(
        df,
        names="type",
        title="<b>Netflix Content Type Distribution</b>",
        hole=0.4,
        color_discrete_sequence=["#ff8a80", "#80cebe"],
    )
    fig.update_traces(
        textinfo="percent+label",
        marker=dict(line=dict(color="white", width=2)),
    )
    fig.update_layout(
        title_font=dict(size=24, color="#d32f2f"),
        annotations=[
            dict(
                text="Netflix",
                x=0.5,
                y=0.5,
                font_size=20,
                showarrow=False,
                font_color="#d32f2f",
            )
        ],
        height=520,
        margin=dict(l=40, r=20, t=70, b=40),
    )

    dist = [
        {
            "label": str(label),
            "count": int(count),
            "pct": float(count / len(df) * 100),
        }
        for label, count in target_counts.items()
    ]

    return fig, dist


def fig_corr_matrix(df: pd.DataFrame) -> Tuple[go.Figure | None, List[str], List[Dict[str, object]]]:
    numerical_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    if len(numerical_cols) <= 1:
        return None, numerical_cols, []

    corr_matrix = df[numerical_cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu_r",
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 12},
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title="Correlation Matrix - Numerical Features",
        width=800,
        height=700,
        xaxis={"tickangle": 45},
        margin=dict(l=60, r=20, t=70, b=60),
    )

    high_corrs: List[Dict[str, object]] = []
    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            corr = float(corr_matrix.iloc[i, j])
            if abs(corr) > 0.5:
                high_corrs.append(
                    {
                        "a": numerical_cols[i],
                        "b": numerical_cols[j],
                        "r": corr,
                    }
                )

    return fig, numerical_cols, high_corrs


def outlier_iqr_release_year(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, object]]:
    feature = "release_year"

    fig = go.Figure(
        data=[
            go.Box(
                y=df[feature],
                name=feature.replace("_", " ").title(),
                marker_color="rgb(102, 197, 204)",
                boxmean="sd",
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="<b>Release Year Box Plot (Outliers)</b>",
            font=dict(size=22, color="#667eea"),
            x=0,
            y=0.95,
        ),
        yaxis_title="Value",
        width=600,
        height=600,
        plot_bgcolor="white",
        margin=dict(t=80, l=40, r=20, b=40),
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.3)")

    Q1 = float(df[feature].quantile(0.25))
    Q3 = float(df[feature].quantile(0.75))
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

    stats = {
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": int(len(outliers)),
        "outlier_pct": float(len(outliers) / len(df) * 100),
        "min_outlier": float(outliers[feature].min()) if len(outliers) else None,
        "max_outlier": float(outliers[feature].max()) if len(outliers) else None,
    }

    return fig, stats


def fig_target_relationships(df: pd.DataFrame) -> Dict[str, Tuple[go.Figure, Dict[str, object]]]:
    target_col = "type"
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cols_to_plot = [c for c in cat_cols if c not in ["show_id", "title", "description", "type"]]

    figs: Dict[str, Tuple[go.Figure, Dict[str, object]]] = {}

    for feature in cols_to_plot:
        if feature in ["country", "cast", "director", "listed_in"]:
            temp_df = df[[feature, target_col]].dropna().copy()
            temp_df[feature] = temp_df[feature].astype(str).str.split(", ")
            temp_df = temp_df.explode(feature)
        else:
            temp_df = df[[feature, target_col]].dropna().copy()

        top_categories = temp_df[feature].value_counts().nlargest(15).index
        filtered_df = temp_df[temp_df[feature].isin(top_categories)]
        crosstab = pd.crosstab(filtered_df[feature], filtered_df[target_col], normalize="index") * 100
        sort_col = "Movie" if "Movie" in crosstab.columns else crosstab.columns[0]
        crosstab = crosstab.sort_values(sort_col, ascending=False)

        fig = go.Figure()
        if "TV Show" in crosstab.columns:
            fig.add_trace(
                go.Bar(
                    x=crosstab.index,
                    y=crosstab["TV Show"],
                    name="TV Show",
                    marker_color="#f48fb1",
                    marker_line_color="#ad1457",
                    marker_line_width=1.5,
                )
            )
        if "Movie" in crosstab.columns:
            fig.add_trace(
                go.Bar(
                    x=crosstab.index,
                    y=crosstab["Movie"],
                    name="Movie",
                    marker_color="#90caf9",
                    marker_line_color="#1565c0",
                    marker_line_width=1.5,
                )
            )

        fig.update_layout(
            title=dict(
                text=f"<b>{target_col.title()} by {feature.replace('_', ' ').title()}</b>",
                font=dict(size=22, color="#1565c0"),
                x=0,
            ),
            xaxis_title=feature.replace("_", " ").title(),
            yaxis_title="Percentage (%)",
            barmode="stack",
            template="plotly_white",
            width=1000,
            height=600,
            margin=dict(l=40, r=20, t=80, b=140),
        )

        # Build a compact table showing the top-5 categories by the sort_col
        top_n = 5
        try:
            top_df = crosstab.nlargest(top_n, sort_col)
        except Exception:
            # Fallback: take first top_n rows if nlargest fails
            top_df = crosstab.head(top_n)

        top_rows: List[Dict[str, object]] = []
        for idx in top_df.index.tolist():
            movie_pct = float(top_df.loc[idx, "Movie"]) if "Movie" in top_df.columns else 0.0
            tv_pct = float(top_df.loc[idx, "TV Show"]) if "TV Show" in top_df.columns else 0.0
            top_rows.append({
                "category": str(idx),
                "movie_pct": movie_pct,
                "tv_pct": tv_pct,
                "total_pct": float(movie_pct + tv_pct),
            })

        figs[feature] = (fig, {"top_rows": top_rows, "sort_col": sort_col})

    return figs


def sample_rows(df: pd.DataFrame, n: int = 3, seed: int = 42) -> Dict[str, List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    out: Dict[str, List[Dict[str, object]]] = {}

    for t in ["Movie", "TV Show"]:
        subset = df[df["type"] == t]
        if len(subset) == 0:
            out[t] = []
            continue
        take = min(n, len(subset))
        idx = rng.choice(subset.index.to_numpy(), size=take, replace=False)
        out[t] = subset.loc[idx].to_dict(orient="records")

    return out
