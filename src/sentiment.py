"""
Uses VADER for fast rule-based sentiment scoring.
Includes aspect-based sentiment analysis (ASBA) using keyword matching + VADER.
"""

import re
import pandas as pd
from typing import Any, cast
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Aspect Keyword map
ASPECT_KEYWORDS = {
    "battery": ["battery", "charge", "charging", "power", "drain", "mah"],
    "screen": ["screen", "display", "resolution", "brightness", "pixel", "lcd", "oled"],
    "performance": ["fast", "slow", "speed", "lag", "performance", "smooth", "processor", "cpu"],
    "build_quality":["build", "quality", "material", "plastic", "metal", "sturdy", "cheap", "durable"],
    "camera": ["camera", "photo", "picture", "image", "lens", "zoom", "megapixel", "selfie"],
    "price": ["price", "cost", "worth", "value", "expensive", "cheap", "affordable", "money"],
    "software": ["software", "app", "update", "bug", "interface", "ui", "ux", "os", "android"],
    "sound": ["sound", "audio", "speaker", "volume", "bass", "noise", "mic", "microphone"],
    "delivery": ["delivery", "shipping", "package", "arrived", "damaged", "box", "packaging"],
    "customer_service": ["support", "service", "refund", "return", "response", "help", "customer"],
}


def get_vader_scores(text: str) -> dict:
    """Return VADER compound, positive, negative, neutral scores."""
    if not isinstance(text, str) or text.strip() == "":
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    return analyzer.polarity_scores(text)


def classify_sentiment(compound: float) -> str:
    """Convert compound score to label."""
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


# ML-based general sentiment (non-review text)
_transformer_pipeline = None  # lazy singleton

def _load_transformer():
    """Load distilbert SST-2 pipeline once and cache it."""
    global _transformer_pipeline
    if _transformer_pipeline is not None:
        return _transformer_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        _transformer_pipeline = hf_pipeline(
            task=cast(Any, "sentiment-analysis"),
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512
        )
        return _transformer_pipeline
    except Exception:
        return None


def analyze_general_text(text: str) -> dict:
    """
    Analyse sentiment of any free-form text (not necessarily a product review).
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "vader": {"compound": 0.0, "label": "neutral"},
            "textblob": {"polarity": 0.0, "subjectivity": 0.0, "label": "neutral"},
            "ml": {"label": "neutral", "score": 0.5, "model": "none"},
            "consensus": "neutral",
        }

    # VADER
    vader_scores = analyzer.polarity_scores(text)
    vader_label = classify_sentiment(vader_scores["compound"])

    # TextBlob
    try:
        from textblob import TextBlob
        tb = TextBlob(text)
        sentiment: Any = tb.sentiment
        tb_polarity = round(sentiment.polarity, 4)
        tb_subjectivity = round(sentiment.subjectivity, 4)
        if tb_polarity > 0.1:
            tb_label = "positive"
        elif tb_polarity < -0.1:
            tb_label = "negative"
        else:
            tb_label = "neutral"
    except ImportError:
        tb_polarity, tb_subjectivity, tb_label = 0.0, 0.0, vader_label

    # — Transformer (distilbert SST-2)
    pipe = _load_transformer()
    if pipe is not None:
        try:
            result = pipe(text[:512])[0]
            raw_label = result["label"].lower()   # "positive" | "negative"
            ml_score = round(result["score"], 4)
            # SST-2 has no "neutral" — map low-confidence predictions to neutral
            if ml_score < 0.65:
                ml_label = "neutral"
            else:
                ml_label = raw_label
            ml_model = "distilbert-sst2"
        except Exception:
            ml_label, ml_score, ml_model = tb_label, abs(tb_polarity), "textblob-fallback"
    else:
        ml_label = tb_label
        ml_score = round((tb_polarity + 1) / 2, 4)   # normalise −1..1 → 0..1
        ml_model = "textblob-fallback"

    # Consensus: majority vote across three signals
    votes = [vader_label, tb_label, ml_label]
    consensus = max(set(votes), key=votes.count)

    return {
        "vader": {"compound": round(vader_scores["compound"], 4), "label": vader_label},
        "textblob": {"polarity": tb_polarity, "subjectivity": tb_subjectivity, "label": tb_label},
        "ml": {"label": ml_label, "score": ml_score, "model": ml_model},
        "consensus": consensus,
    }


def run_sentiment(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Add VADER sentiment scores and label to the DataFrame.
    Returns the DataFrame with new columns:
        vader_compound, vader_pos, vader_neg, vader_neu, predicted_sentiment
    """
    print("[Sentiment] Running VADER on all reviews...")
    scores = df[text_col].apply(get_vader_scores)
    df["vader_compound"] = scores.apply(lambda s: s["compound"])
    df["vader_pos"] = scores.apply(lambda s: s["pos"])
    df["vader_neg"] = scores.apply(lambda s: s["neg"])
    df["vader_neu"] = scores.apply(lambda s: s["neu"])
    df["predicted_sentiment"] = df["vader_compound"].apply(classify_sentiment)
    print("[Sentiment] Done.")
    return df


def extract_aspect_sentences(text: str) -> dict[str, list[str]]:
    """
    Split review into sentences and bucket them by aspect keyword.
    Returns {aspect: [sentence, ...]}
    """
    sentences = re.split(r'[.!?]', text)
    aspect_sentences = {aspect: [] for aspect in ASPECT_KEYWORDS}
    for sentence in sentences:
        sentence = sentence.strip().lower()
        if not sentence:
            continue
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(kw in sentence for kw in keywords):
                aspect_sentences[aspect].append(sentence)
    return aspect_sentences


def run_absa(df: pd.DataFrame, text_col: str = "clean_text",
             min_mentions: int = 5) -> pd.DataFrame:
    """
    Aspect-Based Sentiment Analysis.
    For each aspect, collect all mentioning sentences, score with VADER.
    Returns a summary DataFrame: aspect | avg_score | label | mention_count
    """
    print("[ABSA] Extracting aspect-level sentiment...")
    aspect_scores = {aspect: [] for aspect in ASPECT_KEYWORDS}

    for text in df[text_col]:
        aspect_sents = extract_aspect_sentences(str(text))
        for aspect, sents in aspect_sents.items():
            for sent in sents:
                score = analyzer.polarity_scores(sent)["compound"]
                aspect_scores[aspect].append(score)

    rows = []
    for aspect, scores in aspect_scores.items():
        if len(scores) >= min_mentions:
            avg = sum(scores) / len(scores)
            rows.append({
                "aspect": aspect.replace("_", " ").title(),
                "avg_score": round(avg, 4),
                "label": classify_sentiment(avg),
                "mention_count": len(scores)
            })

    absa_df = pd.DataFrame(rows).sort_values("avg_score", ascending=False).reset_index(drop=True)
    print(f"[ABSA] Found {len(absa_df)} aspects with enough mentions.")
    return absa_df


def sentiment_over_time(df: pd.DataFrame,
                         date_col: str = "review_date",
                         score_col: str = "vader_compound",
                         freq: str = "ME") -> pd.DataFrame:
    """
    Aggregate average sentiment score by time period.
    freq: 'ME' = month end, 'W' = weekly, 'QE' = quarterly
    """
    if date_col not in df.columns:
        return pd.DataFrame()
    temp = df[[date_col, score_col]].dropna()
    temp[date_col] = pd.to_datetime(temp[date_col])
    return temp.set_index(date_col).resample(freq)[score_col].mean().reset_index()


if __name__ == "__main__":
    # Quick test
    df = pd.read_csv("data/processed/reviews_clean.csv")
    df = run_sentiment(df)
    absa = run_absa(df)
    print("\nOverall Sentiment Distribution")
    print(df["predicted_sentiment"].value_counts())
    print("\nAspect Sentiment")
    print(absa)
    df.to_csv("data/processed/reviews_sentiment.csv", index=False)
