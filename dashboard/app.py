"""
Product Review Analysis Dashboard
Built with Streamlit + Plotly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

from src.utils import load_processed, get_summary_stats, get_rating_distribution, get_sentiment_counts
from src.sentiment import run_absa, sentiment_over_time
from src.topic_modeling import train_lda, assign_topics, extract_keywords, get_wordcloud_data

# Page config
st.set_page_config(
    page_title="Review Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme detection + color palette
def get_theme_colors():
    """Return chart colors based on current Streamlit theme."""
    # Detect via query params or default to dark
    theme = st.get_option("theme.base") or "dark"
    is_dark = (theme == "dark")
    return {
        "is_dark": is_dark,
        "bg": "#0e1117" if is_dark else "#ffffff",
        "surface": "#1a1f2e" if is_dark else "#f7f7f7",
        "border": "#2d3347" if is_dark else "#e0e0e0",
        "text": "#e8eaf0" if is_dark else "#1a1a1a",
        "text_muted": "#7a8299" if is_dark else "#666666",
        "bar_fill": "#ffffff" if is_dark else "#111111",
        "bar_pos": "#ffffff" if is_dark else "#111111",
        "bar_neg": "#888888" if is_dark else "#555555",
        "bar_neu": "#555555" if is_dark else "#999999",
        "line": "#ffffff" if is_dark else "#000000",
        "line2": "#aaaaaa" if is_dark else "#555555",
        "grid": "#2a2f3d" if is_dark else "#eeeeee",
        "wc_bg": "black" if is_dark else "white",
        "wc_color": "white" if is_dark else "black",
        "plotly_theme":"plotly_dark" if is_dark else "plotly_white",
    }

C = get_theme_colors()

# Global Plotly layout defaults
LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="monospace", color=C["text"]),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor=C["grid"], linecolor=C["border"], tickcolor=C["text_muted"]),
    yaxis=dict(gridcolor=C["grid"], linecolor=C["border"], tickcolor=C["text_muted"]),
)

def apply_layout(fig):
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


# CSS
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] {{
      font-family: 'JetBrains Mono', monospace;
      background-color: {C['bg']};
      color: {C['text']};
  }}

  /* Header */
  .dash-header {{
      border-bottom: 1px solid {C['border']};
      padding-bottom: 1.2rem;
      margin-bottom: 2rem;
  }}
  .dash-title {{
      font-family: 'Syne', sans-serif;
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      color: {C['text']};
      margin: 0;
  }}
  .dash-subtitle {{
      font-size: 0.75rem;
      color: {C['text_muted']};
      letter-spacing: 0.15em;
      text-transform: uppercase;
      margin-top: 0.3rem;
  }}

  /* Metric cards */
  .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1rem;
      margin-bottom: 2rem;
  }}
  .metric-card {{
      background: {C['surface']};
      border: 1px solid {C['border']};
      padding: 1.2rem 1.4rem;
      border-radius: 4px;
  }}
  .metric-label {{
      font-size: 0.65rem;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: {C['text_muted']};
      margin-bottom: 0.4rem;
  }}
  .metric-value {{
      font-family: 'Syne', sans-serif;
      font-size: 1.8rem;
      font-weight: 700;
      color: {C['text']};
      line-height: 1;
  }}
  .metric-sub {{
      font-size: 0.7rem;
      color: {C['text_muted']};
      margin-top: 0.3rem;
  }}

  /* Section heading */
  .section-label {{
      font-size: 0.65rem;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      color: {C['text_muted']};
      border-bottom: 1px solid {C['border']};
      padding-bottom: 0.4rem;
      margin-bottom: 1rem;
  }}

  /* Aspect table */
  .aspect-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.5rem 0;
      border-bottom: 1px solid {C['border']};
      font-size: 0.82rem;
  }}
  .aspect-bar-wrap {{
      width: 120px;
      height: 6px;
      background: {C['border']};
      border-radius: 3px;
      overflow: hidden;
  }}
  .aspect-bar-fill {{
      height: 100%;
      border-radius: 3px;
  }}
  .badge {{
      font-size: 0.6rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 2px 8px;
      border-radius: 2px;
  }}
  .badge-pos {{ background: {C['bar_fill']}22; color: {C['text']}; border: 1px solid {C['bar_fill']}55; }}
  .badge-neg {{ background: #88888822; color: {C['text_muted']}; border: 1px solid #88888855; }}
  .badge-neu {{ background: #55555522; color: {C['text_muted']}; border: 1px solid #55555555; }}

  /* Hide Streamlit chrome */
  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
  /* header {{ visibility: hidden; }} */
  .stDeployButton {{ display: none; }}

  div[data-testid="stSidebar"] {{
      background: {C['surface']};
      border-right: 1px solid {C['border']};
  }}
</style>
""", unsafe_allow_html=True)


# Data loading
@st.cache_data(show_spinner=False)
def load_data(path):
    df = load_processed(path)
    return df

@st.cache_data(show_spinner=False)
def compute_absa(_df):
    return run_absa(_df)

@st.cache_data(show_spinner=False)
def compute_lda(_df, n_topics):
    doc_topics, topic_words = train_lda(_df, n_topics=n_topics)
    df_with_topics = assign_topics(_df.copy(), doc_topics)
    return df_with_topics, topic_words

@st.cache_data(show_spinner=False)
def compute_keywords(_df):
    return extract_keywords(_df)

@st.cache_data(show_spinner=False)
def compute_wordcloud_data(_df, sentiment_filter):
    sub = _df if sentiment_filter == "all" else _df[_df["predicted_sentiment"] == sentiment_filter]
    return get_wordcloud_data(sub)


# Sidebar
with st.sidebar:
    st.markdown(f"<div class='section-label'>Configuration</div>", unsafe_allow_html=True)

    data_path = st.text_input(
        "Dataset path",
        value="data/processed/reviews_final.csv",
        help="Path to your processed CSV"
    )

    st.markdown("---")
    st.markdown(f"<div class='section-label'>Filters</div>", unsafe_allow_html=True)

    sentiment_filter = st.selectbox(
        "Sentiment filter",
        ["all", "positive", "negative", "neutral"]
    )

    rating_filter = st.multiselect(
        "Star ratings",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5]
    )

    n_topics = st.slider("LDA topics", min_value=3, max_value=15, value=8)

    st.markdown("---")
    st.markdown(f"<div class='section-label'>Wordcloud</div>", unsafe_allow_html=True)
    wc_sentiment = st.selectbox(
        "Wordcloud sentiment",
        ["all", "positive", "negative"]
    )

    # st.markdown("---")
    # st.markdown(f"""
    # <div style='font-size:0.65rem; color:{C['text_muted']}; line-height:1.8;'>
    # NLP Review Dashboard<br>
    # VTU · 6th Sem · NLP Project<br>
    # May 2025
    # </div>
    # """, unsafe_allow_html=True)


# Load + filter data
try:
    df_raw = load_data(data_path)
except FileNotFoundError:
    st.error(f"Dataset not found at `{data_path}`. Run the preprocessing pipeline first.")
    st.info("Quick start: `python src/preprocessing.py data/raw/reviews.csv`")
    st.stop()

# Apply filters
df = df_raw.copy()
if sentiment_filter != "all" and "predicted_sentiment" in df.columns:
    df = df[df["predicted_sentiment"] == sentiment_filter]
if "overall" in df.columns and rating_filter:
    df = df[df["overall"].isin(rating_filter)]

if df.empty:
    st.warning("No data matches current filters. Adjust the sidebar.")
    st.stop()


# Header
st.markdown(f"""
<div class='dash-header'>
  <p class='dash-title'>Product Review Analysis</p>
  <p class='dash-subtitle'>NLP · Sentiment · Topic Modeling · Aspect Analysis</p>
</div>
""", unsafe_allow_html=True)


# KPI Cards
stats = get_summary_stats(df)
sentiment_counts = get_sentiment_counts(df) if "predicted_sentiment" in df.columns else {}
total = len(df)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Total Reviews</div>
      <div class='metric-value'>{total:,}</div>
      <div class='metric-sub'>after filters</div>
    </div>""", unsafe_allow_html=True)

with col2:
    avg_r = stats.get("avg_rating", "—")
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Avg. Rating</div>
      <div class='metric-value'>{avg_r if avg_r else '—'}</div>
      <div class='metric-sub'>out of 5.0</div>
    </div>""", unsafe_allow_html=True)

with col3:
    pct_pos = stats.get("pct_positive", 0)
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Positive</div>
      <div class='metric-value'>{pct_pos}%</div>
      <div class='metric-sub'>{sentiment_counts.get('positive', 0):,} reviews</div>
    </div>""", unsafe_allow_html=True)

with col4:
    avg_v = stats.get("avg_vader_compound", "—")
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>VADER Score</div>
      <div class='metric-value'>{avg_v if avg_v else '—'}</div>
      <div class='metric-sub'>avg compound (−1 to +1)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# Row 1: Sentiment Distribution + Rating Distribution
st.markdown(f"<div class='section-label'>Sentiment &amp; Rating Overview</div>", unsafe_allow_html=True)
r1c1, r1c2 = st.columns(2)

with r1c1:
    # Sentiment pie / donut
    sent_data = {
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count": [
            sentiment_counts.get("positive", 0),
            sentiment_counts.get("negative", 0),
            sentiment_counts.get("neutral", 0),
        ]
    }
    fig_sent = go.Figure(data=[go.Pie(
        labels=sent_data["Sentiment"],
        values=sent_data["Count"],
        hole=0.6,
        marker=dict(colors=["#ffffff", "#555555", "#999999"] if C["is_dark"] else ["#111111", "#888888", "#cccccc"]),
        textfont=dict(color=C["text"], family="monospace"),
        showlegend=True,
    )])
    layout_updates = {
        "title": dict(text="Sentiment Distribution", font=dict(color=C["text"], size=13)),
        "legend": dict(font=dict(color=C["text"], size=10))
    }
    layout_updates.update(
        {k: v for k, v in LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis")}
    )
    fig_sent.update_layout(**layout_updates)
    st.plotly_chart(fig_sent, use_container_width=True, key="sent_pie")

with r1c2:
    # Rating bar chart
    rating_df = get_rating_distribution(df)
    if not rating_df.empty:
        fig_rating = px.bar(
            rating_df, x="rating", y="count",
            labels={"rating": "Stars", "count": "Reviews"},
            title="Rating Distribution",
            color_discrete_sequence=[C["bar_fill"]]
        )
        apply_layout(fig_rating)
        fig_rating.update_layout(title_font_color=C["text"], title_font_size=13)
        st.plotly_chart(fig_rating, use_container_width=True, key="rating_bar")


# Row 2: Sentiment over time (weekly, monthly, quarterly, annually)
if "review_date" in df.columns:
    st.markdown(f"<div class='section-label'>Sentiment Over Time</div>", unsafe_allow_html=True)
    freq_map = {"Weekly": "W", "Monthly": "ME", "Quarterly": "QE", "Annually": "YE"}
    freq_choice = st.radio("Aggregation", list(freq_map.keys()), horizontal=True)
    sot = sentiment_over_time(df, freq=freq_map[freq_choice])
    if not sot.empty:
        fig_time = px.line(
            sot, x="review_date", y="vader_compound",
            labels={"review_date": "Date", "vader_compound": "Avg VADER Score"},
            title="Average Sentiment Score Over Time",
            color_discrete_sequence=[C["line"]]
        )
        fig_time.add_hline(y=0, line_dash="dot", line_color=C["text_muted"], opacity=0.5)
        apply_layout(fig_time)
        fig_time.update_layout(title_font_color=C["text"], title_font_size=13)
        st.plotly_chart(fig_time, use_container_width=True, key="time_line")


# Row 3: ABSA
st.markdown(f"<div class='section-label'>Aspect-Based Sentiment Analysis</div>", unsafe_allow_html=True)

with st.spinner("Analyzing aspects..."):
    absa_df = compute_absa(df)

if absa_df.empty:
    st.info("Not enough aspect mentions found with current filters.")
else:
    absa_c1, absa_c2 = st.columns([1, 1])

    with absa_c1:
        # Horizontal bar chart
        fig_absa = px.bar(
            absa_df.sort_values("avg_score"),
            x="avg_score", y="aspect",
            orientation="h",
            color="avg_score",
            color_continuous_scale=["#555555", "#ffffff"] if C["is_dark"] else ["#cccccc", "#000000"],
            labels={"avg_score": "VADER Score", "aspect": ""},
            title="Aspect Sentiment Scores"
        )
        apply_layout(fig_absa)
        fig_absa.update_layout(
            title_font_color=C["text"], title_font_size=13,
            coloraxis_showscale=False,
            yaxis=dict(tickfont=dict(size=11), gridcolor=C["grid"])
        )
        st.plotly_chart(fig_absa, use_container_width=True, key="absa_bar")

    with absa_c2:
        st.markdown(f"<div style='font-size:0.7rem; color:{C['text_muted']}; margin-bottom:0.5rem;'>Aspect breakdown - sorted by score</div>", unsafe_allow_html=True)
        for _, row in absa_df.iterrows():
            score = row["avg_score"]
            norm = (score + 1) / 2  # normalize −1..1 → 0..1
            bar_color = C["bar_fill"] if score >= 0.05 else (C["bar_neg"] if score <= -0.05 else C["bar_neu"])
            badge_cls = "badge-pos" if row["label"] == "positive" else ("badge-neg" if row["label"] == "negative" else "badge-neu")
            st.markdown(f"""
            <div class='aspect-row'>
              <span>{row['aspect']}</span>
              <div style='display:flex; align-items:center; gap:0.6rem;'>
                <div class='aspect-bar-wrap'>
                  <div class='aspect-bar-fill' style='width:{int(norm*100)}%; background:{bar_color};'></div>
                </div>
                <span style='font-size:0.7rem; color:{C['text_muted']}; width:40px; text-align:right;'>{score:+.3f}</span>
                <span class='badge {badge_cls}'>{row['label']}</span>
                <span style='font-size:0.65rem; color:{C['text_muted']};'>{row['mention_count']}×</span>
              </div>
            </div>""", unsafe_allow_html=True)


# Row 4: LDA Topic Modeling
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"<div class='section-label'>LDA Topic Modeling</div>", unsafe_allow_html=True)

with st.spinner(f"Training LDA with {n_topics} topics..."):
    df_topics, topic_words = compute_lda(df, n_topics)

t_c1, t_c2 = st.columns([1.2, 1])

with t_c1:
    # Topic distribution bar
    topic_dist = df_topics.groupby("dominant_topic").size().reset_index(name="count")
    topic_labels = {t["topic_id"]: t["label"] for t in topic_words}
    topic_dist["topic_label"] = topic_dist["dominant_topic"].map(topic_labels)
    topic_dist["top_words"] = topic_dist["dominant_topic"].map(
        {t["topic_id"]: t["top_words_str"] for t in topic_words}
    )

    fig_topics = px.bar(
        topic_dist.sort_values("count", ascending=True),
        x="count", y="topic_label", orientation="h",
        color_discrete_sequence=[C["bar_fill"]],
        labels={"count": "Reviews", "topic_label": ""},
        title="Review Distribution by Topic",
        hover_data=["top_words"]
    )
    apply_layout(fig_topics)
    fig_topics.update_layout(title_font_color=C["text"], title_font_size=13)
    st.plotly_chart(fig_topics, use_container_width=True, key="topic_bar")

with t_c2:
    st.markdown(f"<div style='font-size:0.7rem; color:{C['text_muted']}; margin-bottom:0.8rem;'>Topic word composition</div>", unsafe_allow_html=True)
    selected_topic_label = st.selectbox(
        "Inspect topic",
        options=[t["label"] for t in topic_words],
        label_visibility="collapsed"
    )
    selected_topic = next(t for t in topic_words if t["label"] == selected_topic_label)

    # Mini bar chart of word weights
    fig_words = px.bar(
        x=selected_topic["weights"][::-1],
        y=selected_topic["words"][::-1],
        orientation="h",
        color_discrete_sequence=[C["bar_fill"]],
        labels={"x": "Weight", "y": ""},
    )
    apply_layout(fig_words)
    fig_words.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(tickfont=dict(size=10), gridcolor=C["grid"])
    )
    st.plotly_chart(fig_words, use_container_width=True, key="topic_words_bar")


# Row 5: Keywords + Wordcloud
st.markdown(f"<div class='section-label'>Keywords &amp; Wordcloud</div>", unsafe_allow_html=True)

kw_c1, kw_c2 = st.columns(2)

with kw_c1:
    with st.spinner("Extracting keywords..."):
        kw_df = compute_keywords(df)

    fig_kw = px.bar(
        kw_df.head(20).sort_values("score"),
        x="score", y="keyword", orientation="h",
        color_discrete_sequence=[C["bar_fill"]],
        labels={"score": "TF-IDF Score", "keyword": ""},
        title="Top Keywords (TF-IDF)"
    )
    apply_layout(fig_kw)
    fig_kw.update_layout(title_font_color=C["text"], title_font_size=13)
    st.plotly_chart(fig_kw, use_container_width=True, key="kw_bar")

with kw_c2:
    st.markdown(f"<div style='font-size:0.7rem; color:{C['text_muted']}; margin-bottom:0.5rem;'>Wordcloud · {wc_sentiment} reviews</div>", unsafe_allow_html=True)
    with st.spinner("Generating wordcloud..."):
        wc_data = compute_wordcloud_data(df, wc_sentiment)

    if wc_data:
        wc = WordCloud(
            width=700, height=380,
            background_color=C["wc_bg"],
            colormap="gray" if C["is_dark"] else "Greys",
            prefer_horizontal=0.9,
            max_words=80,
            min_font_size=10,
        ).generate_from_frequencies(wc_data)

        fig_wc, ax = plt.subplots(figsize=(7, 3.8))
        fig_wc.patch.set_facecolor(C["wc_bg"])
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=C["wc_bg"], dpi=150)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close()


# Row 6: VADER Score Distribution
st.markdown(f"<div class='section-label'>VADER Score Distribution</div>", unsafe_allow_html=True)

if "vader_compound" in df.columns:
    fig_hist = px.histogram(
        df, x="vader_compound", nbins=60,
        color_discrete_sequence=[C["bar_fill"]],
        labels={"vader_compound": "Compound Score", "count": "Reviews"},
        title="Distribution of VADER Compound Scores"
    )
    fig_hist.add_vline(x=0.05, line_dash="dot", line_color=C["text_muted"], opacity=0.6)
    fig_hist.add_vline(x=-0.05, line_dash="dot", line_color=C["text_muted"], opacity=0.6)
    apply_layout(fig_hist)
    fig_hist.update_layout(title_font_color=C["text"], title_font_size=13)
    st.plotly_chart(fig_hist, use_container_width=True, key="vader_hist")


# Row 7: Raw data explorer
with st.expander("Raw Data Explorer"):
    display_cols = [c for c in ["clean_text", "overall", "vader_compound", "predicted_sentiment", "dominant_topic"] if c in df.columns]
    st.dataframe(
        df[display_cols].head(500),
        use_container_width=True,
        height=300
    )
    st.caption(f"Showing first 500 of {len(df):,} filtered rows.")
