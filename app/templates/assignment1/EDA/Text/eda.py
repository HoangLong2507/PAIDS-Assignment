from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _resolve_data_path() -> Path:
    """Resolve `tripadvisor_hotel_reviews.csv` robustly.

    This page is rendered from a FastAPI app, so we avoid cwd-based paths.
    """

    here = Path(__file__).resolve().parent
    candidates = [
        here / "tripadvisor_hotel_reviews.csv",
        Path(__file__).resolve().parents[5] / "tripadvisor_hotel_reviews.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


DATA_PATH = _resolve_data_path()


RATINGS: List[int] = [1, 2, 3, 4, 5]
COLORS_RATING: Dict[int, str] = {
    1: "#d62728",
    2: "#ff7f0e",
    3: "#bcbd22",
    4: "#2ca02c",
    5: "#1f77b4",
}


@dataclass(frozen=True)
class DatasetOverview:
    total_reviews: int
    total_features: int
    rating_min: int
    rating_max: int
    missing_review: int
    missing_rating: int
    duplicates: int
    avg_words: float
    avg_chars: float


@lru_cache(maxsize=2)
def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # normalize columns (dataset uses Review,Rating)
    if "Review" not in df.columns or "Rating" not in df.columns:
        raise ValueError("Expected columns: Review, Rating")

    # drop duplicates and fill missing
    df = df.drop_duplicates().reset_index(drop=True)
    df["Review"] = df["Review"].fillna("")

    # basic lengths used throughout
    df["word_count"] = df["Review"].astype(str).apply(lambda x: len(x.split()))
    df["char_count"] = df["Review"].astype(str).apply(len)
    return df


def get_overview(df: pd.DataFrame) -> DatasetOverview:
    raw = pd.read_csv(DATA_PATH)
    duplicates = int(raw.duplicated().sum())

    return DatasetOverview(
        total_reviews=int(len(df)),
        total_features=int(len(df.columns)),
        rating_min=int(df["Rating"].min()),
        rating_max=int(df["Rating"].max()),
        missing_review=int(raw["Review"].isnull().sum()) if "Review" in raw.columns else 0,
        missing_rating=int(raw["Rating"].isnull().sum()) if "Rating" in raw.columns else 0,
        duplicates=duplicates,
        avg_words=float(df["word_count"].mean()),
        avg_chars=float(df["char_count"].mean()),
    )


def per_rating_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Rating")[["word_count", "char_count"]]
        .mean()
        .reindex(RATINGS)
        .round(2)
        .reset_index()
    )


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["Rating"].value_counts().reindex(RATINGS).fillna(0).astype(int)
    pct = (vc / len(df) * 100).round(2)
    out = pd.DataFrame({"Rating": vc.index, "Count": vc.values, "Percent": pct.values})
    return out


def fig_rating_pie(df: pd.DataFrame) -> go.Figure:
    dist = rating_distribution(df)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[f"{int(r)}★" for r in dist["Rating"].tolist()],
                values=dist["Count"].tolist(),
                hole=0.3,
                marker=dict(colors=[COLORS_RATING[int(r)] for r in dist["Rating"].tolist()]),
                textinfo="label+percent",
                textfont_size=13,
            )
        ]
    )
    fig.update_layout(
        title="Rating Distribution — Proportion (1★ through 5★)",
        showlegend=True,
        width=600,
        height=450,
        template="plotly_white",
    )
    return fig


def fig_stopwords_raw(df: pd.DataFrame) -> Tuple[go.Figure, int, int]:
    stop_words = set(ENGLISH_STOP_WORDS)
    words = " ".join(df["Review"].astype(str)).lower().split()
    total_words = int(len(words))
    stop_counts: Dict[str, int] = {}
    for w in words:
        if w in stop_words:
            stop_counts[w] = stop_counts.get(w, 0) + 1
    top = sorted(stop_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    if top:
        x, y = zip(*top)
        x = list(x)
        y = list(y)
    else:
        x, y = [], []

    fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=y,
                marker_color="#9467bd",
                text=y,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Top 20 Stop Words in Raw Hotel Reviews",
        xaxis_title="Stop Words",
        yaxis_title="Frequency",
        template="plotly_white",
        height=500,
    )

    total_stop = int(sum(stop_counts.values()))
    return fig, total_words, total_stop


def clean_text_series(df: pd.DataFrame) -> pd.Series:
    stop_words = set(ENGLISH_STOP_WORDS)
    # close to notebook pipeline: lowercase, remove punctuation, keep alpha tokens, remove stopwords
    import string as _string

    translator = str.maketrans("", "", _string.punctuation)

    def _clean(text: str) -> str:
        txt = str(text).lower().translate(translator)
        toks = txt.split()
        toks = [w for w in toks if w.isalpha() and w not in stop_words]
        return " ".join(toks)

    return df["Review"].astype(str).apply(_clean)


def vocab_sizes(df: pd.DataFrame, cleaned_text: pd.Series) -> Tuple[int, int]:
    vocab_before = len(set(" ".join(df["Review"].astype(str).str.lower()).split()))
    vocab_after = len(set(" ".join(cleaned_text.astype(str)).split()))
    return int(vocab_before), int(vocab_after)


def vocab_richness(df: pd.DataFrame, cleaned_text: pd.Series) -> pd.DataFrame:
    tmp = df.copy()
    tmp["cleaned_text"] = cleaned_text

    rows: List[Dict[str, object]] = []
    for r in RATINGS:
        subset = tmp[tmp["Rating"] == r]["cleaned_text"].astype(str)
        all_words = " ".join(subset.tolist()).split()
        rows.append(
            {
                "Rating": f"{r}★",
                "Total Words": int(len(all_words)),
                "Unique Words": int(len(set(all_words))),
            }
        )
    return pd.DataFrame(rows)


def fig_vocab_richness(stats_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Total Words", "Unique Vocabulary Size"))

    for _, row in stats_df.iterrows():
        r_int = int(str(row["Rating"])[0])
        color = COLORS_RATING.get(r_int, "#333")
        fig.add_trace(
            go.Bar(
                name=row["Rating"],
                x=[row["Rating"]],
                y=[row["Total Words"]],
                marker_color=color,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name=row["Rating"],
                x=[row["Rating"]],
                y=[row["Unique Words"]],
                marker_color=color,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title_text="Vocabulary Richness by Rating (1–5★)",
        template="plotly_white",
        barmode="group",
        height=520,
    )
    return fig


def fig_top50_words_by_rating(df: pd.DataFrame, cleaned_text: pd.Series) -> Dict[int, go.Figure]:
    tmp = df.copy()
    tmp["cleaned_text"] = cleaned_text
    out: Dict[int, go.Figure] = {}

    for rating in RATINGS:
        subset_words = " ".join(tmp[tmp["Rating"] == rating]["cleaned_text"].astype(str)).split()
        if not subset_words:
            continue
        counts: Dict[str, int] = {}
        for w in subset_words:
            counts[w] = counts.get(w, 0) + 1
        top_50 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:50]
        words = [w for w, _ in top_50]
        freqs = [c for _, c in top_50]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=words,
                    y=freqs,
                    marker_color=COLORS_RATING[rating],
                )
            ]
        )
        fig.update_layout(
            title=f"Top 50 Frequent Words — Rating {rating}★ Reviews",
            xaxis_title="Words",
            yaxis_title="Frequency",
            template="plotly_white",
            xaxis_tickangle=-45,
            height=520,
        )
        out[rating] = fig

    return out


def fig_tfidf_top_terms(
    df: pd.DataFrame, cleaned_text: pd.Series, max_features: int = 5000
) -> Dict[int, go.Figure]:
    vec = TfidfVectorizer(max_features=max_features)
    tfidf = vec.fit_transform(cleaned_text.astype(str))
    feature_names = vec.get_feature_names_out()

    out: Dict[int, go.Figure] = {}
    for rating in RATINGS:
        idx = df[df["Rating"] == rating].index
        if len(idx) == 0:
            continue
        cat = tfidf[idx]
        mean_scores = np.array(cat.mean(axis=0)).flatten()
        top_idx = mean_scores.argsort()[-20:][::-1]
        top_words = [str(feature_names[i]) for i in top_idx]
        top_scores = [float(mean_scores[i]) for i in top_idx]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=top_scores,
                    y=top_words,
                    orientation="h",
                    marker_color=COLORS_RATING[rating],
                    text=[f"{s:.3f}" for s in top_scores],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            title=f"Top 20 TF-IDF Terms — Rating {rating}★ Reviews",
            xaxis_title="Mean TF-IDF Score",
            yaxis={"autorange": "reversed"},
            template="plotly_white",
            width=860,
            height=650,
        )
        out[rating] = fig

    return out


def fig_bigrams(
    df: pd.DataFrame, cleaned_text: pd.Series, max_features: int = 5000
) -> Dict[int, go.Figure]:
    tmp = df.copy()
    tmp["cleaned_text"] = cleaned_text

    out: Dict[int, go.Figure] = {}
    for rating in RATINGS:
        subset = tmp[tmp["Rating"] == rating]["cleaned_text"].astype(str).tolist()
        if not subset:
            continue

        vec = CountVectorizer(ngram_range=(2, 2), max_features=max_features)
        mat = vec.fit_transform(subset)
        freqs = list(zip(vec.get_feature_names_out(), mat.sum(axis=0).A1))
        freqs = sorted(freqs, key=lambda x: x[1], reverse=True)[:20]
        if not freqs:
            continue

        phrases = [p for p, _ in freqs]
        counts = [int(c) for _, c in freqs]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts,
                    y=phrases,
                    orientation="h",
                    marker_color=COLORS_RATING[rating],
                    text=[str(c) for c in counts],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            title=f"Top 20 Bigrams — Rating {rating}★ Reviews",
            xaxis_title="Frequency",
            yaxis={"autorange": "reversed"},
            template="plotly_white",
            width=860,
            height=650,
        )
        out[rating] = fig

    return out


def fig_rating_similarity(df: pd.DataFrame, max_features: int = 5000) -> Tuple[go.Figure, List[str], List[List[float]]]:
    cos_vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(ENGLISH_STOP_WORDS),
    )
    tfidf = cos_vectorizer.fit_transform(df["Review"].astype(str))

    n = len(RATINGS)
    sim_mat = np.zeros((n, n), dtype=float)
    for i, r1 in enumerate(RATINGS):
        v1 = tfidf[df[df["Rating"] == r1].index]
        for j, r2 in enumerate(RATINGS):
            v2 = tfidf[df[df["Rating"] == r2].index]
            pairwise = cosine_similarity(v1, v2)
            sim_mat[i, j] = float(np.mean(pairwise))

    labels = [f"{r}★" for r in RATINGS]
    fig = go.Figure(
        data=go.Heatmap(
            z=sim_mat,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=np.round(sim_mat, 4),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Similarity"),
        )
    )
    fig.update_layout(
        title="Rating Similarity Matrix — Mean Cosine Similarity (5×5)",
        xaxis_title="Rating",
        yaxis_title="Rating",
        width=650,
        height=600,
        template="plotly_white",
        yaxis={"autorange": "reversed"},
    )

    sim_list = sim_mat.round(6).tolist()
    return fig, labels, sim_list
