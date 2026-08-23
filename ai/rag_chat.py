import os
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector

from google import genai
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"

NEW_REVIEWS = 50
TOP_K = 5

CACHE_FILE = "review_embeddings.parquet"


# ============================================================
# GEMINI CLIENT
# ============================================================

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing from your .env file.")
    st.stop()

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Zomato Review Intelligence",
    page_icon="🍽️",
    layout="wide",
)



# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🍽️ Zomato Review Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about customer reviews using Gemini-powered '
    'semantic search and RAG.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    review_count = st.slider(
        "Reviews to load",
        min_value=100,
        max_value=2000,
        value=NEW_REVIEWS,
        step=100,
    )

    top_k = st.slider(
        "Reviews used for answer",
        min_value=3,
        max_value=10,
        value=TOP_K,
    )

    st.divider()

    st.subheader("🤖 Models")

    st.write(
        f"**Embedding:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**Chat:** `{CHAT_MODEL}`"
    )

    st.divider()

    st.caption(
        "Reviews are retrieved using vector similarity "
        "and analyzed by Gemini."
    )


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================

def get_snowflake_connection():
    """
    Create a connection to Snowflake.
    """

    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )


# ============================================================
# READ REVIEWS FROM SNOWFLAKE
# ============================================================

def read_reviews_from_snowflake(review_count):
    """
    Read reviews from Snowflake and return
    them as a Pandas DataFrame.
    """

    conn = get_snowflake_connection()

    query = f"""
        SELECT
            REVIEW_ID,
            CITY,
            RATING,
            COMMENT
        FROM ZOMATO.STAGING.STAGE_REVIEWS
        SAMPLE ({review_count} ROWS)
    """

    try:

        cursor = conn.cursor()

        df = cursor.execute(
            query
        ).fetch_pandas_all()

        cursor.close()

    finally:

        conn.close()

    # Normalize column names
    df.columns = [
        column.lower()
        for column in df.columns
    ]

    # Remove missing comments
    df = df.dropna(
        subset=["comment"]
    )

    # Convert comments to strings
    df["comment"] = df["comment"].astype(str)

    return df


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

def embed(texts):
    """
    Convert text into Gemini embedding vectors.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )

    return [
        embedding.values
        for embedding in response.embeddings
    ]


# ============================================================
# LOAD REVIEWS + CACHE
# ============================================================

@st.cache_data(show_spinner=False)
def load_reviews(review_count):
    """
    Load reviews and embeddings.

    If the Parquet cache exists and contains
    enough reviews, reuse it.

    Otherwise fetch fresh reviews from Snowflake
    and generate Gemini embeddings.
    """

    # --------------------------------------------------------
    # Check persistent cache
    # --------------------------------------------------------

    if os.path.exists(CACHE_FILE):

        try:

            cached_df = pd.read_parquet(
                CACHE_FILE
            )

            # Check whether cache contains enough reviews
            if len(cached_df) >= review_count:

                return cached_df.head(
                    review_count
                )

        except Exception:

            # If cache is corrupted or incompatible,
            # rebuild it.
            pass

    # --------------------------------------------------------
    # Fetch fresh data from Snowflake
    # --------------------------------------------------------

    df = read_reviews_from_snowflake(
        review_count
    )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = []

    batch_size = 20

    for start in range(
        0,
        len(df),
        batch_size,
    ):

        batch = (
            df["comment"]
            .iloc[start:start + batch_size]
            .tolist()
        )

        batch_embeddings = embed(
            batch
        )

        embeddings.extend(
            batch_embeddings
        )

    # Add embeddings to DataFrame
    df["embedding"] = embeddings

    # --------------------------------------------------------
    # Save cache
    # --------------------------------------------------------

    df.to_parquet(
        CACHE_FILE,
        index=False,
    )

    return df


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    vector_a,
    vector_b,
):
    """
    Calculate cosine similarity between
    two embedding vectors.
    """

    vector_a = np.array(
        vector_a
    )

    vector_b = np.array(
        vector_b
    )

    denominator = (
        np.linalg.norm(vector_a)
        *
        np.linalg.norm(vector_b)
    )

    if denominator == 0:

        return 0.0

    return float(
        np.dot(
            vector_a,
            vector_b,
        )
        / denominator
    )


# ============================================================
# FIND SIMILAR REVIEWS
# ============================================================

def find_similar_reviews(
    question,
    df,
    top_k,
):
    """
    Embed the user's question and compare
    it against every review embedding.

    Return the top-K most similar reviews.
    """

    # --------------------------------------------------------
    # Embed question
    # --------------------------------------------------------

    question_vector = embed(
        [question]
    )[0]

    # --------------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------------

    scores = []

    for review_vector in df[
        "embedding"
    ]:

        score = cosine_similarity(
            question_vector,
            review_vector,
        )

        scores.append(
            score
        )

    # --------------------------------------------------------
    # Add scores
    # --------------------------------------------------------

    results = df.copy()

    results["similarity"] = scores

    # --------------------------------------------------------
    # Return top K
    # --------------------------------------------------------

    return results.nlargest(
        top_k,
        "similarity",
    )


# ============================================================
# BUILD CONTEXT FOR GEMINI
# ============================================================

def build_context(top_reviews):
    """
    Convert retrieved reviews into
    text context for Gemini.
    """

    context = ""

    for _, row in top_reviews.iterrows():

        context += (
            f"City: {row['city']}\n"
            f"Rating: {row['rating']} stars\n"
            f"Review: {row['comment']}\n"
            f"Similarity: "
            f"{row['similarity']:.3f}\n"
            f"---\n"
        )

    return context


# ============================================================
# ASK GEMINI
# ============================================================

def ask_gemini(
    question,
    top_reviews,
):
    """
    Ask Gemini to answer the question
    using only the retrieved reviews.
    """

    context = build_context(
        top_reviews
    )

    system_instruction = """
You are a customer review analyst
for a food delivery application.

Answer the user's question ONLY using
the customer reviews provided.

Do not use outside knowledge.

Do not invent facts.

If the provided reviews do not contain
enough information to answer the question,
say that the available reviews do not
provide enough information.

Be concise and clear.

When appropriate, mention:
- common complaints
- positive patterns
- recurring themes
- rating patterns
- cities associated with feedback
"""

    prompt = f"""
Customer Question:

{question}

Retrieved Customer Reviews:

{context}

Answer the question using ONLY
the retrieved reviews.
"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config={
            "system_instruction":
                system_instruction,

            "temperature": 0.2,
        },
    )

    return response.text


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner(
    "🔄 Loading review intelligence..."
):

    try:

        review_df = load_reviews(
            review_count
        )

    except Exception as e:

        st.error(
            f"Failed to load reviews: {e}"
        )

        st.stop()


# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Reviews",
        f"{len(review_df):,}",
    )

with col2:

    st.metric(
        "Cities",
        f"{review_df['city'].nunique():,}",
    )

with col3:

    st.metric(
        "Avg Rating",
        f"{review_df['rating'].mean():.2f} ⭐",
    )

with col4:

    st.metric(
        "Top-K",
        top_k,
    )


st.divider()


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader(
    "💬 Ask your reviews"
)

question = st.text_input(
    "Ask a question",
    placeholder=(
        "e.g. What are the most common "
        "complaints about delivery?"
    ),
    label_visibility="collapsed",
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.caption(
    "💡 Example questions"
)

example_1, example_2, example_3 = st.columns(3)

examples = [
    (
        example_1,
        "What are the common complaints about delivery?"
    ),
    (
        example_2,
        "What do customers like about the food?"
    ),
    (
        example_3,
        "Why are customers unhappy with pricing?"
    ),
]

for column, example in examples:

    with column:

        if st.button(
            example,
            use_container_width=True,
        ):

            question = example


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    st.divider()

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "🔎 Searching relevant reviews..."
    ):

        try:

            top_reviews = find_similar_reviews(
                question,
                review_df,
                top_k,
            )

        except Exception as e:

            st.error(
                f"Search failed: {e}"
            )

            st.stop()

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Gemini is analyzing the reviews..."
    ):

        try:

            answer = ask_gemini(
                question,
                top_reviews,
            )

        except Exception as e:

            st.error(
                f"Gemini request failed: {e}"
            )

            st.stop()

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    st.subheader(
        "💡 Answer"
    )

    st.markdown(
        f"""
        <div class="answer-box">
        {answer}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # RETRIEVED REVIEWS
    # --------------------------------------------------------

    st.subheader(
        "🔎 Evidence"
    )

    st.caption(
        f"Gemini used the top "
        f"{len(top_reviews)} semantically "
        f"similar reviews."
    )

    # --------------------------------------------------------
    # REVIEW CARDS
    # --------------------------------------------------------

    for index, (_, row) in enumerate(
        top_reviews.iterrows(),
        start=1,
    ):

        with st.expander(
            f"#{index}  "
            f"{row['city']} • "
            f"{row['rating']} ⭐ • "
            f"Similarity "
            f"{row['similarity']:.3f}"
        ):

            st.write(
                row["comment"]
            )

            st.caption(
                f"Review ID: "
                f"{row['review_id']}"
            )

    # --------------------------------------------------------
    # DATA TABLE
    # --------------------------------------------------------

    with st.expander(
        "📊 View retrieved review data"
    ):

        display_df = top_reviews[
            [
                "review_id",
                "city",
                "rating",
                "comment",
                "similarity",
            ]
        ].copy()

        display_df[
            "similarity"
        ] = display_df[
            "similarity"
        ].round(3)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "👆 Ask a question above to search "
        "your Zomato reviews."
    )

    st.subheader(
        "🚀 Things you can ask"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            ### 📦 Delivery

            - What are the common delivery complaints?
            - Do customers mention late deliveries?
            - What do customers say about delivery partners?
            """
        )

    with col2:

        st.markdown(
            """
            ### 🍛 Food

            - What do customers like about the food?
            - What food quality problems appear?
            - What do customers say about taste?
            """
        )

    with col3:

        st.markdown(
            """
            ### 💰 Pricing

            - Why are customers unhappy with pricing?
            - Do customers think restaurants are expensive?
            - What do customers say about value?
            """
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Powered by Snowflake • Gemini Embeddings • "
    "Gemini Flash • Streamlit"
)