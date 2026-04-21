"""
Uses LDA (Latent Dirichlet Allocation) via gensim.
Keyword extraction via TF-IDF.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# LDA Topic Modeling

def train_lda(df: pd.DataFrame,
              token_col: str = "token_str",
              n_topics: int = 8,
              n_words: int = 10,
              max_features: int = 5000) -> tuple:
    """
    Train an LDA model on preprocessed token strings.
    """
    print(f"[LDA] Training with {n_topics} topics...")
    texts = df[token_col].fillna("").tolist()

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=5,  # word must appear in at least 5 docs
        max_df=0.90,  # ignore words in >90% of docs
        ngram_range=(1, 2)
    )
    dtm = vectorizer.fit_transform(texts)

    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
        max_iter=20,
        doc_topic_prior=0.1,
        topic_word_prior=0.01
    )
    doc_topics = lda_model.fit_transform(dtm)

    # Extract top words per topic
    feature_names = vectorizer.get_feature_names_out()
    topic_words = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        words = [feature_names[i] for i in top_indices]
        weights = [round(topic[i], 4) for i in top_indices]
        topic_words.append({
            "topic_id": topic_idx,
            "label": f"Topic {topic_idx + 1}",
            "words": words,
            "weights": weights,
            "top_words_str": ", ".join(map(str, words[:5]))
        })

    print(f"[LDA] Done. Perplexity: {lda_model.perplexity(dtm):.2f}")
    return lda_model, vectorizer, doc_topics, topic_words


def assign_topics(df: pd.DataFrame, doc_topics: np.ndarray) -> pd.DataFrame:
    """Add dominant topic index and confidence to DataFrame."""
    df["dominant_topic"] = doc_topics.argmax(axis=1)
    df["topic_confidence"] = doc_topics.max(axis=1).round(4)
    return df


def get_topic_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return count of reviews per dominant topic."""
    return df.groupby("dominant_topic").size().reset_index(name="count")


def extract_keywords(df: pd.DataFrame,
                     token_col: str = "token_str",
                     top_n: int = 30) -> pd.DataFrame:
    """
    Extract top keywords from the entire corpus using TF-IDF.
    """
    print("[Keywords] Extracting via TF-IDF...")
    texts = df[token_col].fillna("").tolist()

    tfidf = TfidfVectorizer(
        max_features=1000,
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2)
    )
    matrix = csr_matrix(tfidf.fit_transform(texts))
    scores = np.asarray(matrix.mean(axis=0)).flatten()
    feature_names = tfidf.get_feature_names_out()

    kw_df = pd.DataFrame({"keyword": feature_names, "score": scores})
    kw_df = kw_df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    print(f"[Keywords] Top keyword: '{kw_df.iloc[0]['keyword']}'")
    return kw_df


def keywords_by_sentiment(df: pd.DataFrame,
                           token_col: str = "token_str",
                           sentiment_col: str = "predicted_sentiment",
                           top_n: int = 20) -> dict[str, pd.DataFrame]:
    """
    Extract top keywords separately for positive and negative reviews.
    Returns {"positive": df, "negative": df}
    """
    results = {}
    for label in ["positive", "negative"]:
        subset = df[df[sentiment_col] == label]
        if len(subset) < 10:
            continue
        results[label] = extract_keywords(subset, token_col=token_col, top_n=top_n)
    return results


def get_wordcloud_data(df: pd.DataFrame,
                        token_col: str = "token_str",
                        top_n: int = 100) -> dict[str, float]:
    """
    Return a dict {word: frequency} for wordcloud rendering.
    Uses raw term frequency (not TF-IDF).
    """
    from collections import Counter
    all_tokens = []
    for token_str in df[token_col].fillna(""):
        all_tokens.extend(token_str.split())
    counter = Counter(all_tokens)
    return dict(counter.most_common(top_n))


if __name__ == "__main__":
    df = pd.read_csv("data/processed/reviews_sentiment.csv")
    df["token_str"] = df["token_str"].fillna("")

    lda_model, vectorizer, doc_topics, topic_words = train_lda(df, n_topics=8)
    df = assign_topics(df, doc_topics)

    print("\nTopics")
    for t in topic_words:
        print(f"  {t['label']}: {t['top_words_str']}")

    kw_df = extract_keywords(df)
    print("\nTop Keywords")
    print(kw_df.head(10))

    df.to_csv("data/processed/reviews_final.csv", index=False)
