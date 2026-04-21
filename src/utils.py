"""
Shared helper functions used across modules.
"""

import json
import pandas as pd
from pathlib import Path


def ensure_dirs():
    """Create required directories if they don't exist."""
    dirs = [
        "data/raw",
        "data/processed",
        "notebooks",
        "dashboard",
        "report",
        "ppt",
        "tests"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_processed(path: str = "data/processed/reviews_final.csv") -> pd.DataFrame:
    """Load final processed CSV. Deserialize token lists."""
    df = pd.read_csv(path)
    if "tokens" in df.columns:
        df["tokens"] = df["tokens"].apply(
            lambda x: x.split(" | ") if isinstance(x, str) else []
        )
    return df


def get_rating_distribution(df: pd.DataFrame,
                              rating_col: str = "overall") -> pd.DataFrame:
    if rating_col not in df.columns:
        return pd.DataFrame()
    dist = df[rating_col].value_counts().sort_index().reset_index()
    dist.columns = ["rating", "count"]
    dist["rating"] = dist["rating"].astype(int)
    return dist


def get_sentiment_counts(df: pd.DataFrame,
                          col: str = "predicted_sentiment") -> dict:
    counts = df[col].value_counts().to_dict()
    return {
        "positive": counts.get("positive", 0),
        "negative": counts.get("negative", 0),
        "neutral": counts.get("neutral", 0),
    }


def get_summary_stats(df: pd.DataFrame) -> dict:
    stats = {
        "total_reviews": len(df),
        "avg_rating": round(df["overall"].mean(), 2) if "overall" in df.columns else None,
        "avg_vader_compound": round(df["vader_compound"].mean(), 4) if "vader_compound" in df.columns else None,
    }
    if "predicted_sentiment" in df.columns:
        counts = get_sentiment_counts(df)
        total = len(df)
        stats["pct_positive"] = round(counts["positive"] / total * 100, 1)
        stats["pct_negative"] = round(counts["negative"] / total * 100, 1)
        stats["pct_neutral"] = round(counts["neutral"] / total * 100, 1)
    return stats


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[Utils] Saved JSON: {path}")


def load_json(path: str):
    with open(path) as f:
        return json.load(f)
