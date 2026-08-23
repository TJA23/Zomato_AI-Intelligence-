import os
import json
import re
import pandas as pd
import streamlit as st
import snowflake.connector

from google import genai
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gemini-3.6-flash"

EXAMPLE_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisine has the most orders?",
    "Average delivery time by city, worst first",
    "Cancel rate by payment method",
]


# ============================================================
# GEMINI CLIENT
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing from your .env file.")
    st.stop()


client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# DATABASE SCHEMA
# ============================================================

SCHEMA = """
Tables available in Snowflake.

Use bare table names only.
Do not use database or schema prefixes.

FACT_ORDERS(
    order_id,
    order_date,
    customer_id,
    restaurant_id,
    city,
    cuisine,
    payment_method,
    order_status,
    is_delivered,
    sales_amount,
    discount,
    delivery_fee,
    gst,
    customer_rating,
    delivery_time_min
)

DIM_RESTAURANT(
    restaurant_id,
    restaurant_name,
    city,
    cuisine,
    rating,
    cost_for_two
)

DIM_CUSTOMER(
    customer_id,
    customer_name,
    age,
    age_segment,
    gender,
    city
)

MART_DAILY_CITY_REVENUE(
    order_date,
    city,
    orders,
    cancel_rate,
    gmv,
    aov
)

MART_RESTAURANT_PERFORMANCE(
    restaurant_id,
    restaurant_name,
    city,
    cuisine,
    orders,
    revenue,
    avg_customer_rating,
    cancel_rate
)

MART_DELIVERY_SLA(
    city,
    order_hour,
    delivered_orders,
    p50_delivery_min,
    late_rate
)

Business definition:

gmv means delivered revenue.

Prefer MART_ tables when they fit the question.
"""


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are an expert Snowflake SQL analyst.

Your job is to convert a user's natural-language
business question into ONE SQL query.

Rules:

1. Generate ONLY one SELECT or WITH query.

2. Never generate:
   - INSERT
   - UPDATE
   - DELETE
   - DROP
   - ALTER
   - TRUNCATE
   - CREATE
   - REPLACE
   - GRANT
   - REVOKE

3. Use only the tables and columns provided in the schema.

4. Use bare table names.

   Correct:
   FROM FACT_ORDERS

   Incorrect:
   FROM ZOMATO.MARTS.FACT_ORDERS

5. Prefer MART_ tables when they contain the
   information needed to answer the question.

6. Add LIMIT 100 or less for multi-row results.

7. If the user asks for a single total or single
   aggregate value, a LIMIT is not required.

8. Use appropriate aggregation such as:
   SUM, AVG, COUNT, COUNT DISTINCT.

9. Use GROUP BY when aggregation is required.

10. Use ORDER BY when the user asks for rankings,
    highest, lowest, best, worst, top, etc.

11. For delivery metrics, use delivered orders
    when appropriate.

12. Return ONLY valid JSON in this format:

{{
    "sql": "your SQL query here"
}}

Database schema:

{SCHEMA}
"""


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================

@st.cache_resource
def get_connection():
    """
    Create and cache a Snowflake connection.

    Streamlit will reuse this connection
    across application reruns.
    """

    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema="MARTS",
        role="DBT_ROLE",
    )


# ============================================================
# GENERATE SQL
# ============================================================

def generate_sql(question):
    """
    Convert natural-language question
    into a Snowflake SQL query using Gemini.
    """

    SQL_RESPONSE_SCHEMA = {
        "type": "OBJECT",
        "properties": {
            "sql": {
                "type": "STRING"
            }
        },
        "required": ["sql"],
    }

    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_schema": SQL_RESPONSE_SCHEMA,
        },
    )

    # Gemini response is now expected to be JSON
    answer = response.text.strip()

    try:

        parsed = json.loads(answer)

    except json.JSONDecodeError as e:

        raise ValueError(
            "Gemini returned invalid JSON.\n\n"
            f"Response:\n{answer}"
        ) from e

    sql = parsed.get("sql")

    if not sql:

        raise ValueError(
            "Gemini response did not contain a SQL query."
        )

    # Remove accidental database/schema prefixes
    sql = sql.replace(
        "ZOMATO.MARTS.",
        ""
    )

    sql = sql.replace(
        "ZOMATO.",
        ""
    )

    # Remove whitespace
    sql = sql.strip()

    # Remove ONLY trailing semicolon
    sql = sql.rstrip(";")

    return sql.strip()


# ============================================================
# SQL SAFETY CHECK
# ============================================================

FORBIDDEN_WORDS = [
    "drop",
    "delete",
    "truncate",
    "alter",
    "update",
    "insert",
    "create",
    "replace",
    "grant",
    "revoke",
]


def is_safe(sql):
    """
    Basic safety validation.

    Returns:
        True  -> SQL is allowed
        False -> SQL should not execute
    """

    # --------------------------------------------------------
    # Create lowercase COPY for checking
    # --------------------------------------------------------

    lowered = sql.lower().strip()

    # --------------------------------------------------------
    # Must start with SELECT or WITH
    # --------------------------------------------------------

    if not (
        lowered.startswith("select")
        or lowered.startswith("with")
    ):
        return False

    # --------------------------------------------------------
    # Remove SQL comments
    # --------------------------------------------------------

    without_comments = re.sub(
        r"--.*?$",
        "",
        lowered,
        flags=re.MULTILINE,
    )

    without_comments = re.sub(
        r"/\*.*?\*/",
        "",
        without_comments,
        flags=re.DOTALL,
    )

    # --------------------------------------------------------
    # Check forbidden SQL operations
    # --------------------------------------------------------

    for word in FORBIDDEN_WORDS:

        pattern = rf"\b{word}\b"

        if re.search(
            pattern,
            without_comments,
        ):
            return False

    # --------------------------------------------------------
    # Prevent multiple SQL statements
    # --------------------------------------------------------

    if ";" in without_comments:

        return False

    return True


# ============================================================
# RUN QUERY
# ============================================================

def run_query(sql):
    """
    Execute validated SQL in Snowflake
    and return a Pandas DataFrame.
    """

    conn = get_connection()

    cursor = conn.cursor()

    try:

        # Optional query timeout: 60 seconds
        cursor.execute(
            "ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 60"
        )

        df = cursor.execute(
            sql
        ).fetch_pandas_all()

        return df

    finally:

        cursor.close()


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Zomato Data Chat",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 Chat with your Zomato Data")

st.caption(
    f"Ask questions in English → "
    f"{MODEL} generates SQL → "
    f"Snowflake executes it"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("💡 Example Questions")

    for question in EXAMPLE_QUESTIONS:

        st.markdown(
            f"- {question}"
        )

    st.divider()

    st.subheader("🤖 AI Model")

    st.write(
        f"`{MODEL}`"
    )

    st.divider()

    st.subheader("🔐 Security")

    st.write(
        "Only SELECT/WITH queries are allowed."
    )

    st.write(
        "Destructive SQL operations are blocked."
    )


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_input(
    "Ask a question about your Zomato data",
    placeholder=(
        "Example: Top 10 restaurants by revenue in Bangalore"
    ),
)


# ============================================================
# MAIN PROCESSING
# ============================================================

if question:

    # --------------------------------------------------------
    # GENERATE SQL
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Gemini is generating SQL..."
    ):

        try:

            sql = generate_sql(
                question
            )

        except Exception as e:

            st.error(
                f"Failed to generate SQL: {e}"
            )

            st.stop()

    # --------------------------------------------------------
    # SHOW GENERATED SQL
    # --------------------------------------------------------

    st.subheader(
        "🧠 Generated SQL"
    )

    st.code(
        sql,
        language="sql",
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if not is_safe(sql):

        st.error(
            "🚫 The generated SQL failed the safety check."
        )

        st.warning(
            "Only SELECT/WITH queries are allowed."
        )

        st.stop()

    st.success(
        "✅ SQL passed the safety check."
    )

    # --------------------------------------------------------
    # EXECUTE QUERY
    # --------------------------------------------------------

    with st.spinner(
        "❄️ Snowflake is executing the query..."
    ):

        try:

            df = run_query(
                sql
            )

        except Exception as e:

            st.error(
                f"Snowflake query failed:\n\n{e}"
            )

            st.stop()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.subheader(
        "📈 Query Results"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows Returned",
            len(df),
        )

    with col2:

        st.metric(
            "Columns",
            len(df.columns),
        )

    with col3:

        st.metric(
            "Query Status",
            "Success",
        )

    # --------------------------------------------------------
    # DATA TABLE
    # --------------------------------------------------------

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # AUTOMATIC CHART
    # --------------------------------------------------------

    if (
        len(df.columns) == 2
        and len(df) > 0
        and pd.api.types.is_numeric_dtype(
            df.iloc[:, 1]
        )
    ):

        st.subheader(
            "📊 Visualization"
        )

        st.bar_chart(
            df,
            x=df.columns[0],
            y=df.columns[1],
        )

    # --------------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------------

    with st.expander(
        "🔍 Query Details"
    ):

        st.write(
            "User question:"
        )

        st.code(
            question
        )

        st.write(
            "SQL generated by Gemini:"
        )

        st.code(
            sql,
            language="sql",
        )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "👆 Enter a question above to start exploring your Zomato data."
    )

    st.subheader(
        "Try asking:"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            **Revenue & Business**

            - Top 10 cities by GMV
            - Which restaurants generate the most revenue?
            - What is the average order value by city?
            - Which cuisine generates the highest revenue?
            """
        )

    with col2:

        st.markdown(
            """
            **Operations**

            - Which city has the worst delivery time?
            - What is the cancellation rate by payment method?
            - Which restaurants have the highest cancellation rate?
            - What is the average delivery time by city?
            """
        )