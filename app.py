import streamlit as st
import pandas as pd
import joblib
import re
from transformers import pipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FeedbackAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
/* Hide Streamlit's default top header */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0px !important;
}

header[data-testid="stHeader"] > div {
    display: none !important;
}

/* Remove empty top space */
.block-container {
    padding-top: 1.5rem !important;
}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99, 102, 241, 0.14),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(168, 85, 247, 0.12),
            transparent 30%
        ),
        #080b14;
    color: #f8fafc;
}


/* Remove default top spacing */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0b1020 0%,
            #11162a 100%
        );
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}


/* =========================================================
   BRAND
   ========================================================= */

.brand {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 4px;
}

.brand span {
    background: linear-gradient(
        90deg,
        #8b5cf6,
        #22d3ee
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-sub {
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 30px;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    padding: 38px;
    border-radius: 26px;
    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.20),
            rgba(168,85,247,0.10)
        );
    border: 1px solid rgba(255,255,255,0.09);
    box-shadow:
        0 20px 60px rgba(0,0,0,0.25);
    margin-bottom: 28px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(34,211,238,0.10);
    border: 1px solid rgba(34,211,238,0.25);
    color: #67e8f9;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 16px;
}

.hero-title {
    font-size: 42px;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 12px;
}

.hero-title span {
    background: linear-gradient(
        90deg,
        #a78bfa,
        #22d3ee
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    color: #aeb8ca;
    font-size: 16px;
    max-width: 760px;
    line-height: 1.7;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

.metric {
    padding: 22px;
    border-radius: 20px;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow:
        0 12px 30px rgba(0,0,0,0.16);
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #f8fafc;
}

.metric-icon {
    font-size: 23px;
    margin-bottom: 12px;
}


/* =========================================================
   SECTION
   ========================================================= */

.section-title {
    font-size: 24px;
    font-weight: 750;
    margin-top: 35px;
    margin-bottom: 18px;
    color: #f8fafc;
}

.section-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 20px;
}


/* =========================================================
   ANALYZER CARD
   ========================================================= */

.analyzer-card {
    padding: 28px;
    border-radius: 24px;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    box-shadow: 0 15px 45px rgba(0,0,0,0.18);
}


/* =========================================================
   RESULT CARDS
   ========================================================= */

.result-card {
    padding: 22px;
    border-radius: 18px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
}

.result-label {
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 8px;
}

.result-value {
    font-size: 25px;
    font-weight: 750;
}


/* =========================================================
   STATUS
   ========================================================= */

.online {
    color: #4ade80;
    font-weight: 600;
}

.online-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #4ade80;
    border-radius: 50%;
    margin-right: 7px;
    box-shadow: 0 0 10px #4ade80;
}


/* =========================================================
   INFO BOX
   ========================================================= */

.info-box {
    padding: 20px;
    border-radius: 18px;
    background: rgba(99,102,241,0.09);
    border: 1px solid rgba(99,102,241,0.18);
    color: #c7d2fe;
    line-height: 1.7;
}


/* =========================================================
   STREAMLIT INPUT
   ========================================================= */

textarea {
    background: #111827 !important;
    color: #f8fafc !important;
    border: 1px solid #263247 !important;
    border-radius: 14px !important;
}

textarea:focus {
    border: 1px solid #8b5cf6 !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(139,92,246,0.4);
    background:
        linear-gradient(
            90deg,
            #7c3aed,
            #4f46e5
        );
    color: white;
    font-weight: 700;
    padding: 12px;
    transition: 0.2s;
}

.stButton > button:hover {
    border-color: #a78bfa;
    transform: translateY(-1px);
}


/* =========================================================
   DATAFRAME
   ========================================================= */

[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.07);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_sentiment_model():

    model = joblib.load(
        "data/processed/sentiment_model.pkl"
    )

    vectorizer = joblib.load(
        "data/processed/tfidf_vectorizer.pkl"
    )

    return model, vectorizer


@st.cache_resource
def load_emotion_model():

    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )


@st.cache_data
def load_reviews():

    return pd.DataFrame({
        "sentiment": (
            ["Positive"] * 443777
            + ["Neutral"] * 42640
            + ["Negative"] * 82037
        )
    })

@st.cache_data
def load_intelligence():

    try:
        return pd.read_csv(
            "data/processed/customer_intelligence_summary.csv"
        )

    except FileNotFoundError:

        try:
            return pd.read_csv(
                "data/processed/aspect_priority_results.csv"
            )

        except FileNotFoundError:

            return pd.DataFrame()


model, vectorizer = load_sentiment_model()

emotion_classifier = load_emotion_model()

df_reviews = load_reviews()

df_intelligence = load_intelligence()


# ============================================================
# ASPECTS
# ============================================================

aspects = {

    "Taste": [
        "taste",
        "flavor",
        "flavour",
        "delicious",
        "tasty",
        "sweet",
        "bitter",
        "spicy"
    ],

    "Quality": [
        "quality",
        "fresh",
        "stale",
        "ingredient",
        "ingredients",
        "organic"
    ],

    "Price": [
        "price",
        "cost",
        "expensive",
        "cheap",
        "value",
        "worth"
    ],

    "Packaging": [
        "package",
        "packaging",
        "box",
        "packed",
        "container",
        "seal"
    ],

    "Delivery": [
        "delivery",
        "delivered",
        "shipping",
        "arrived",
        "shipment"
    ],

    "Service": [
        "service",
        "seller",
        "customer support",
        "support",
        "refund"
    ],

    "Appearance": [
        "color",
        "colour",
        "design",
        "appearance",
        "look",
        "looks"
    ]
}


positive_words = [
    "amazing",
    "excellent",
    "great",
    "good",
    "perfect",
    "delicious",
    "tasty",
    "love",
    "loved",
    "best",
    "fresh",
    "wonderful",
    "fantastic",
    "awesome",
    "beautiful"
]


negative_words = [
    "terrible",
    "bad",
    "poor",
    "awful",
    "worst",
    "hate",
    "hated",
    "disappointing",
    "stale",
    "expensive",
    "slow",
    "useless",
    "broken"
]


# ============================================================
# FUNCTIONS
# ============================================================

def extract_aspects(review):

    text = review.lower()

    found = []

    for aspect, keywords in aspects.items():

        for keyword in keywords:

            if keyword in text:

                found.append(aspect)

                break

    return found


def aspect_sentiment(review, aspect):

    text = review.lower()

    clauses = re.split(
        r'\bbut\b|\band\b|\bhowever\b',
        text
    )

    for clause in clauses:

        if aspect.lower() in clause:

            positive_count = sum(
                word in clause
                for word in positive_words
            )

            negative_count = sum(
                word in clause
                for word in negative_words
            )

            if positive_count > negative_count:
                return "Positive"

            if negative_count > positive_count:
                return "Negative"

            return "Neutral"

    return "Unknown"


def detect_emotion(review):

    result = emotion_classifier(
        review,
        truncation=True,
        max_length=512
    )[0]

    return result["label"], result["score"]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">Feedback<span>AI</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-sub">CONSUMER INTELLIGENCE PLATFORM</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<span class="online-dot"></span>'
        '<span class="online">AI Engine Online</span>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "⌂  Overview",
            "⌕  Review Analyzer",
            "◈  Aspect Intelligence",
            "✦  Emotion Intelligence",
            "ⓘ  About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.caption(
        "NLP • Machine Learning • Transformers"
    )

    st.caption(
        "FeedbackAI v1.0"
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "⌂  Overview":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-badge">
                ✦ AI-POWERED CUSTOMER INTELLIGENCE
            </div>

            <div class="hero-title">
                Turn customer voice into
                <span>business intelligence.</span>
            </div>

            <div class="hero-description">
                Analyze thousands of customer reviews using
                machine learning, NLP, aspect analysis and
                transformer-based emotion detection.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    total = len(df_reviews)

    positive = (
        df_reviews["sentiment"] == "Positive"
    ).sum()

    negative = (
        df_reviews["sentiment"] == "Negative"
    ).sum()

    neutral = (
        df_reviews["sentiment"] == "Neutral"
    ).sum()

    positive_pct = positive / total * 100

    negative_pct = negative / total * 100

    neutral_pct = neutral / total * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-icon">🗂️</div>
                <div class="metric-label">
                    REVIEWS ANALYZED
                </div>
                <div class="metric-value">
                    {total:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-icon">😊</div>
                <div class="metric-label">
                    POSITIVE FEEDBACK
                </div>
                <div class="metric-value">
                    {positive_pct:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-icon">😐</div>
                <div class="metric-label">
                    NEUTRAL FEEDBACK
                </div>
                <div class="metric-value">
                    {neutral_pct:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-icon">⚠️</div>
                <div class="metric-label">
                    NEGATIVE FEEDBACK
                </div>
                <div class="metric-value">
                    {negative_pct:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Customer Sentiment Overview</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        chart = pd.DataFrame(
            {
                "Reviews": [
                    positive,
                    neutral,
                    negative
                ]
            },
            index=[
                "Positive",
                "Neutral",
                "Negative"
            ]
        )

        st.bar_chart(chart)

    with right:

        st.markdown(
            """
            <div class="info-box">

            <b>How FeedbackAI works</b><br><br>

            01 — Customer review is received<br>
            02 — Sentiment is predicted<br>
            03 — Important aspects are detected<br>
            04 — Aspect-level sentiment is calculated<br>
            05 — Emotional response is identified<br>
            06 — Business insight is generated

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# REVIEW ANALYZER
# ============================================================

elif page == "⌕  Review Analyzer":

    st.markdown(
        '<div class="section-title">Review Intelligence Engine</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Enter a customer review and let the AI analyze what the customer is saying.'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="analyzer-card">',
        unsafe_allow_html=True
    )


    review = st.text_area(
        "Customer Review",
        placeholder=(
            "Example: The taste is amazing but "
            "the delivery was very slow and packaging was terrible."
        ),
        height=160,
        label_visibility="visible"
    )


    analyze = st.button(
        "✦  RUN AI ANALYSIS"
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    if analyze:

        if not review.strip():

            st.warning(
                "Please enter a customer review."
            )

        else:

            with st.spinner(
                "Running sentiment, aspect and emotion models..."
            ):

                vector = vectorizer.transform(
                    [review]
                )

                sentiment = model.predict(
                    vector
                )[0]

                detected = extract_aspects(
                    review
                )

                aspect_results = []

                for aspect in detected:

                    aspect_results.append(
                        {
                            "Aspect": aspect,
                            "Sentiment": aspect_sentiment(
                                review,
                                aspect
                            )
                        }
                    )

                emotion, confidence = detect_emotion(
                    review
                )


            st.markdown(
                '<div class="section-title">AI Analysis Result</div>',
                unsafe_allow_html=True
            )


            r1, r2, r3 = st.columns(3)


            with r1:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            OVERALL SENTIMENT
                        </div>
                        <div class="result-value">
                            {'🟢' if sentiment == 'Positive'
                             else '🔴' if sentiment == 'Negative'
                             else '🟡'}
                            {sentiment}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with r2:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            DETECTED EMOTION
                        </div>
                        <div class="result-value">
                            🧠 {emotion.capitalize()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with r3:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            MODEL CONFIDENCE
                        </div>
                        <div class="result-value">
                            {confidence:.1%}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.markdown(
                '<div class="section-title">Detected Customer Aspects</div>',
                unsafe_allow_html=True
            )


            if aspect_results:

                st.dataframe(
                    pd.DataFrame(
                        aspect_results
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No predefined customer aspect was detected."
                )


            negative = [
                x["Aspect"]
                for x in aspect_results
                if x["Sentiment"] == "Negative"
            ]

            positive = [
                x["Aspect"]
                for x in aspect_results
                if x["Sentiment"] == "Positive"
            ]


            st.markdown(
                '<div class="section-title">Business Interpretation</div>',
                unsafe_allow_html=True
            )


            if negative:

                st.error(
                    "⚠️ Customer concern detected in: "
                    + ", ".join(negative)
                )

            elif positive:

                st.success(
                    "✓ Positive customer feedback detected for: "
                    + ", ".join(positive)
                )

            else:

                st.info(
                    "No strong aspect-level concern detected."
                )


# ============================================================
# ASPECT INTELLIGENCE
# ============================================================

elif page == "◈  Aspect Intelligence":

    st.markdown(
        '<div class="section-title">Aspect Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Understand which parts of the customer experience require attention.'
        '</div>',
        unsafe_allow_html=True
    )


    if not df_intelligence.empty:

        st.dataframe(
            df_intelligence,
            use_container_width=True,
            hide_index=True
        )


        if "negative_percentage" in df_intelligence.columns:

            st.markdown(
                '<div class="section-title">Negative Feedback by Aspect</div>',
                unsafe_allow_html=True
            )


            chart = (
                df_intelligence[
                    [
                        "aspect",
                        "negative_percentage"
                    ]
                ]
                .set_index("aspect")
            )


            st.bar_chart(chart)


        if "priority" in df_intelligence.columns:

            st.markdown(
                '<div class="section-title">Priority Areas</div>',
                unsafe_allow_html=True
            )


            priority = (
                df_intelligence[
                    "priority"
                ]
                .value_counts()
            )


            st.bar_chart(priority)

    else:

        st.warning(
            "Aspect intelligence data is unavailable."
        )


# ============================================================
# EMOTION INTELLIGENCE
# ============================================================

elif page == "✦  Emotion Intelligence":

    st.markdown(
        '<div class="section-title">Emotion Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Go beyond positive and negative sentiment to understand emotional response.'
        '</div>',
        unsafe_allow_html=True
    )


    emotions = pd.DataFrame(
        {
            "Emotion": [
                "Anger",
                "Disgust",
                "Fear",
                "Joy",
                "Neutral",
                "Sadness",
                "Surprise"
            ],

            "Meaning": [
                "Strong frustration",
                "Strong negative reaction",
                "Concern or uncertainty",
                "Happiness and satisfaction",
                "Limited emotional signal",
                "Disappointment or unhappiness",
                "Unexpected reaction"
            ]
        }
    )


    st.dataframe(
        emotions,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        '<div class="section-title">Try the Emotion Engine</div>',
        unsafe_allow_html=True
    )


    text = st.text_area(
        "Enter text",
        placeholder="I am extremely happy with this product!",
        height=130
    )


    if st.button(
        "✦  DETECT EMOTION"
    ):

        if text.strip():

            with st.spinner(
                "Analyzing emotional response..."
            ):

                emotion, score = detect_emotion(
                    text
                )


            st.success(
                f"Detected emotion: {emotion.capitalize()}"
            )


            st.progress(
                score
            )


            st.write(
                f"Model confidence: **{score:.2%}**"
            )

        else:

            st.warning(
                "Please enter some text."
            )


# ============================================================
# ABOUT
# ============================================================

else:

    st.markdown(
        '<div class="section-title">About FeedbackAI</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="hero">

        <div class="hero-badge">
        AI • NLP • MACHINE LEARNING
        </div>

        <div class="hero-title">
        Consumer Feedback <span>Intelligence Platform</span>
        </div>

        <div class="hero-description">

        FeedbackAI transforms large-scale customer reviews into
        structured intelligence using Natural Language Processing,
        machine learning and transformer-based emotion detection.

        <br><br>

        The system identifies sentiment, discovers important
        customer-facing aspects, analyzes aspect-level sentiment,
        detects emotional responses and generates business-oriented
        insights.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.markdown(
            """
            <div class="metric">
                <div class="metric-icon">🤖</div>
                <div class="metric-label">ML MODEL</div>
                <div class="metric-value">
                    Logistic Regression
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            """
            <div class="metric">
                <div class="metric-icon">🧠</div>
                <div class="metric-label">NLP MODEL</div>
                <div class="metric-value">
                    Transformer
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            """
            <div class="metric">
                <div class="metric-icon">📊</div>
                <div class="metric-label">DATASET</div>
                <div class="metric-value">
                    568K+ Reviews
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="section-title">Technology Stack</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info-box">

        🐍 Python &nbsp; • &nbsp;
        🐼 Pandas &nbsp; • &nbsp;
        ⚙️ Scikit-learn &nbsp; • &nbsp;
        🔤 TF-IDF &nbsp; • &nbsp;
        🤗 Transformers &nbsp; • &nbsp;
        🔥 PyTorch &nbsp; • &nbsp;
        🎈 Streamlit

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        FeedbackAI • AI Consumer Feedback Intelligence
        <br>
        Turning customer voice into actionable intelligence.
    </div>
    """,
    unsafe_allow_html=True
)