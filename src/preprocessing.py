"""
Loads raw Amazon review CSV, cleans text, tokenizes, lemmatizes,
and exports a processed CSV for downstream modules.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download required NLTK data (run once)
def download_nltk_data():
    packages = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'punkt_tab'
    ]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {pkg}: {e}")

download_nltk_data()

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Negation words - these matter for sentiment
NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "hardly", "barely", "scarcely"}
STOPWORDS -= NEGATION_WORDS


def clean_text(text: str) -> str:
    """Remove HTML tags, URLs, special characters, lowercase."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)  # HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)  # URLs
    text = re.sub(r"[^a-zA-Z\s']", " ", text)  # Keep letters, spaces, apostrophes
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    return text.strip().lower()


def tokenize_and_lemmatize(text: str) -> list[str]:
    """Tokenize, remove stopwords, lemmatize."""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS and len(t) - 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


def load_and_preprocess(filepath: str, text_col: str = "reviewText",
                        rating_col: str = "overall",
                        sample_size: int | None = None) -> pd.DataFrame:
    """
    Load a raw review CSV and return a cleaned DataFrame.
    """
    print(f"Preprocessing loading: {filepath}")
    df = pd.read_csv(filepath)

    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    # Drop rows with missing review text
    df = df.dropna(subset=[text_col]).reset_index(drop=True)

    print(f"[Preprocessing] {len(df)} reviews loaded. Cleaning...")

    df["clean_text"] = df[text_col].apply(clean_text)
    df["tokens"] = df["clean_text"].apply(tokenize_and_lemmatize)
    df["token_str"] = df["tokens"].apply(lambda t: " ".join(t))

    # Derive sentiment label from star rating
    if rating_col in df.columns:
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
        df["sentiment_label"] = df[rating_col].apply(lambda r: "positive" if r >= 4 else ("negative" if r <= 2 else "neutral"))

    if "reviewTime" in df.columns:
        df["review_date"] = pd.to_datetime(df["reviewTime"], errors="coerce")

    print(f"[Preprocessing] Done. Shape: {df.shape}")
    return df


def save_processed(df: pd.DataFrame, output_path: str = "data/processed?reviews_clean.csv"):
    df_save = df.copy()
    df_save["tokens"] = df_save["tokens"].apply(lambda t: " | ".join(t))  # Serialize list
    df_save.to_csv(output_path, index=False)
    print(f"[Preprocessing] Saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="data/raw/reviews.csv")
    parser.add_argument("--sample", type=int, default=20000)
    args = parser.parse_args()
    df = load_and_preprocess(args.input, sample_size=args.sample)
    save_processed(df, "data/processed/reviews_clean.csv")
    print(df[["clean_text", "tokens", "sentiment_label"]].head())
