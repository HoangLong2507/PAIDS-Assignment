from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _resolve_data_path() -> Path:
    """Resolve `email_spam.csv` robustly.

    Preferred location is next to this module:
    `app/templates/assignment1/EDA/Text/email_spam.csv`.

    Fallback keeps compatibility with earlier assumptions where the CSV lived in
    the repository root.
    """

    here = Path(__file__).resolve().parent
    candidates = [
        here / "email_spam.csv",
        # repository root (..../Assignment/email_spam.csv)
        Path(__file__).resolve().parents[5] / "email_spam.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # If nothing exists, return the "expected" local path so the error is informative.
    return candidates[0]


DATA_PATH = _resolve_data_path()


COLORS = {
    "spam": "#ef553b",
    "ham": "#636efa",
    "not spam": "#636efa",
}


@dataclass(frozen=True)
class DatasetOverview:
    total_emails: int
    categories: List[str]
    missing_values: Dict[str, int]
    duplicates: int
    avg_words: float
    avg_chars: float


def _normalize_type(s: str) -> str:
    s = str(s).strip().lower()
    # normalize "not spam" -> "ham" for plotting colors/labels consistency
    if s in {"not spam", "not_spam", "notspam", "ham"}:
        return "ham" if s != "not spam" else "not spam"
    if s == "spam":
        return "spam"
    return s


@lru_cache(maxsize=2)
def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # basic cleaning (match notebook intent)
    df = df.drop_duplicates().reset_index(drop=True)
    if "text" in df.columns:
        df["text"] = df["text"].fillna("")

    df["type"] = df["type"].astype(str)

    # Derived stats used in multiple charts
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df["char_count"] = df["text"].apply(lambda x: len(str(x)))

    return df


def get_overview(df: pd.DataFrame) -> DatasetOverview:
    duplicates = int(df.duplicated().sum())
    missing = {c: int(df[c].isnull().sum()) for c in df.columns}

    return DatasetOverview(
        total_emails=int(len(df)),
        categories=len(sorted(df["type"].unique().tolist())),
        missing_values=missing,
        duplicates=duplicates,
        avg_words=float(df["word_count"].mean()),
        avg_chars=float(df["char_count"].mean()),
    )


def fig_category_bar(df: pd.DataFrame) -> go.Figure:
    category_counts = df["type"].value_counts()
    percentages = (category_counts / len(df) * 100).round(2)

    fig = go.Figure(
        data=[
            go.Bar(
                x=category_counts.index,
                y=category_counts.values,
                text=[
                    f"{val} ({pct}%)"
                    for val, pct in zip(category_counts.values, percentages.values)
                ],
                textposition="auto",
                marker_color=[
                    COLORS.get(_normalize_type(t), "#333") for t in category_counts.index
                ],
            )
        ]
    )

    fig.update_layout(
        title="Category Distribution (Spam vs. Ham)",
        xaxis_title="Email Type",
        yaxis_title="Count",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def fig_category_pie(df: pd.DataFrame) -> go.Figure:
    category_counts = df["type"].value_counts()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=0.5,
                marker=dict(colors=[
                    "#667eea",
                    "#764ba2",
                    "#f093fb",
                    "#4facfe",
                    "#43e97b",
                ]),
            )
        ]
    )
    fig.update_layout(
        title="Category Distribution (Spam vs. Ham)",
        showlegend=True,
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def fig_length_distributions(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Word Count Distribution", "Character Count Distribution"),
    )

    types = df["type"].unique()
    for t in types:
        subset = df[df["type"] == t]
        color = COLORS.get(_normalize_type(t), "#333333")

        fig.add_trace(
            go.Histogram(
                x=subset["word_count"],
                name=f"{t} Words",
                marker_color=color,
                opacity=0.7,
                nbinsx=50,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=subset["char_count"],
                name=f"{t} Chars",
                marker_color=color,
                opacity=0.7,
                nbinsx=50,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        barmode="overlay",
        title_text="Length Distributions by Category",
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    max_words = float(df["word_count"].quantile(0.95))
    max_chars = float(df["char_count"].quantile(0.95))
    fig.update_xaxes(range=[0, max_words], row=1, col=1)
    fig.update_xaxes(range=[0, max_chars], row=1, col=2)

    return fig


def fig_stopwords_raw(df: pd.DataFrame) -> Tuple[go.Figure, int, int]:
    stop_words = set(ENGLISH_STOP_WORDS)
    all_words_raw = " ".join(df["text"].astype(str)).lower().split()

    from collections import Counter

    stop_word_counts = Counter([w for w in all_words_raw if w in stop_words])
    top_stopwords = stop_word_counts.most_common(20)

    words, counts = zip(*top_stopwords) if top_stopwords else ([], [])

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(words),
                y=list(counts),
                marker_color="#9467bd",
                text=list(counts),
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Top 20 Stop Words in Raw Data",
        xaxis_title="Stop Words",
        yaxis_title="Frequency",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    total_words = len(all_words_raw)
    total_stopwords_found = int(sum(stop_word_counts.values()))
    return fig, total_words, total_stopwords_found


def clean_text_series(df: pd.DataFrame) -> pd.Series:
    stop_words = set(ENGLISH_STOP_WORDS)

    import string as _string

    translator = str.maketrans("", "", _string.punctuation)

    def clean_text(text: str) -> str:
        text = str(text).lower()
        text = text.translate(translator)
        tokens = text.split()
        cleaned = [w for w in tokens if w not in stop_words and w.isalpha()]
        return " ".join(cleaned)

    return df["text"].apply(clean_text)


def vocab_richness(df: pd.DataFrame, cleaned_text: pd.Series) -> pd.DataFrame:
    types = df["type"].unique()
    rows = []
    for t in types:
        subset = cleaned_text[df["type"] == t]
        all_words = " ".join(subset.astype(str)).split()
        rows.append(
            {
                "Category": t,
                "Total Words": int(len(all_words)),
                "Unique Words": int(len(set(all_words))),
            }
        )
    return pd.DataFrame(rows)


def fig_vocab_richness(stats_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Total Words", "Unique Vocabulary Size"),
    )

    for _, row in stats_df.iterrows():
        color = COLORS.get(_normalize_type(row["Category"]), "#333333")
        fig.add_trace(
            go.Bar(
                name=row["Category"],
                x=[row["Category"]],
                y=[row["Total Words"]],
                marker_color=color,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name=row["Category"],
                x=[row["Category"]],
                y=[row["Unique Words"]],
                marker_color=color,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title_text="Vocabulary Richness Comparison",
        template="plotly_white",
        barmode="group",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def fig_top50_words_by_type(df: pd.DataFrame, cleaned_text: pd.Series) -> Dict[str, go.Figure]:
    from collections import Counter

    types = df["type"].unique().tolist()
    figures: Dict[str, go.Figure] = {}

    for t in types:
        words = " ".join(cleaned_text[df["type"] == t]).split()
        top_50 = Counter(words).most_common(50)
        if not top_50:
            continue

        x_words, counts = zip(*top_50)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(x_words),
                    y=list(counts),
                    marker_color=COLORS.get(_normalize_type(t), "#333333"),
                )
            ]
        )
        fig.update_layout(
            title=f"Top 50 Frequent Words in Class: {t}",
            xaxis_title="Words",
            yaxis_title="Frequency",
            template="plotly_white",
            xaxis_tickangle=-45,
            height=520,
            margin=dict(l=40, r=20, t=60, b=80),
        )
        figures[str(t)] = fig

    return figures


def fig_tfidf_top_terms(df: pd.DataFrame, cleaned_text: pd.Series) -> Dict[str, go.Figure]:
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(cleaned_text)
    feature_names = vectorizer.get_feature_names_out()

    figures: Dict[str, go.Figure] = {}
    types = df["type"].unique().tolist()

    for t in types:
        cat_indices = df[df["type"] == t].index
        if len(cat_indices) == 0:
            continue

        cat_tfidf = tfidf_matrix[cat_indices]
        mean_scores = np.array(cat_tfidf.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[-20:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = [float(mean_scores[i]) for i in top_indices]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=top_scores,
                    y=top_words,
                    orientation="h",
                    marker_color=COLORS.get(_normalize_type(t), "#333333"),
                    text=[f"{s:.3f}" for s in top_scores],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            title=f"Top 20 TF-IDF Characteristic Terms in Class: {t}",
            xaxis_title="Mean TF-IDF Score",
            yaxis={"autorange": "reversed"},
            template="plotly_white",
            height=640,
            margin=dict(l=80, r=20, t=60, b=40),
        )

        figures[str(t)] = fig

    return figures


def fig_bigrams(df: pd.DataFrame, cleaned_text: pd.Series) -> Dict[str, go.Figure]:
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=2000)
    figures: Dict[str, go.Figure] = {}

    types = df["type"].unique().tolist()

    for t in types:
        texts = cleaned_text[df["type"] == t]
        if len(texts) == 0:
            continue

        try:
            bigram_matrix = bigram_vectorizer.fit_transform(texts)
        except ValueError:
            continue

        bigram_freqs = np.array(bigram_matrix.sum(axis=0)).flatten()
        bigram_names = bigram_vectorizer.get_feature_names_out()

        top_indices = bigram_freqs.argsort()[-20:][::-1]
        top_bigrams = [bigram_names[i] for i in top_indices]
        top_counts = [int(bigram_freqs[i]) for i in top_indices]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=top_counts,
                    y=top_bigrams,
                    orientation="h",
                    marker_color=COLORS.get(_normalize_type(t), "#333333"),
                    text=top_counts,
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            title=f"Top 20 Bigrams in Class: {t}",
            xaxis_title="Frequency",
            yaxis={"autorange": "reversed"},
            template="plotly_white",
            height=640,
            margin=dict(l=80, r=20, t=60, b=40),
        )

        figures[str(t)] = fig

    return figures


def fig_category_similarity(df: pd.DataFrame) -> Tuple[go.Figure, List[str], np.ndarray]:
    categories = sorted(df["type"].unique().tolist())
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words=list(ENGLISH_STOP_WORDS)
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"].astype(str))

    n_cats = len(categories)
    sim = np.zeros((n_cats, n_cats), dtype=float)

    for i, cat1 in enumerate(categories):
        cat1_idx = df[df["type"] == cat1].index
        v1 = tfidf_matrix[cat1_idx]

        for j, cat2 in enumerate(categories):
            cat2_idx = df[df["type"] == cat2].index
            v2 = tfidf_matrix[cat2_idx]

            pairwise = cosine_similarity(v1, v2)
            sim[i, j] = float(np.mean(pairwise))

    fig = go.Figure(
        data=go.Heatmap(
            z=sim,
            x=categories,
            y=categories,
            colorscale="Purples",
            text=np.round(sim, 4),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Similarity"),
        )
    )

    fig.update_layout(
        title="Category Similarity Matrix (Mean Cosine Similarity)",
        xaxis_title="Category",
        yaxis_title="Category",
        template="plotly_white",
        height=520,
        margin=dict(l=60, r=20, t=60, b=40),
        yaxis={"autorange": "reversed"},
    )

    return fig, categories, sim
