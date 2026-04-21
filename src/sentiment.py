"""
Uses VADER for fast rule-based sentiment scoring.
Includes aspect-based sentiment analysis (ASBA) using keyword matching + VADER.
"""

import re
import pandas as pd
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
