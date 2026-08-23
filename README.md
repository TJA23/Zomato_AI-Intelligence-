<img width="1626" height="919" alt="Screenshot 2026-08-23 at 9 30 49 PM" src="https://github.com/user-attachments/assets/78046eaa-b798-48cc-9d81-55e31aab1fb3" />
<img width="1674" height="915" alt="Screenshot 2026-08-23 at 9 29 28 PM" src="https://github.com/user-attachments/assets/2dc4ce18-bd6c-4788-aeba-cb130340218d" />
# Zomato AI Intelligence

> **An end-to-end data engineering and AI analytics platform built on Snowflake, dbt, Apache Airflow, and Gemini.**

Zomato AI Intelligence transforms raw Zomato-style operational data into analytics-ready marts and adds an AI layer for **review enrichment, Retrieval-Augmented Generation (RAG), and natural-language-to-SQL analytics**.

The project is designed as a production-style data platform: data is ingested into a raw layer, transformed through dbt into staging and mart models, orchestrated with Airflow, and exposed through Streamlit applications and AI capabilities.

---

## 🚀 Project Highlights

- **End-to-end data pipeline** from raw CSV data to analytics-ready Snowflake marts
- **Amazon S3 → Snowflake RAW** ingestion
- **dbt transformations** from RAW → STAGING → MARTS
- **Apache Airflow orchestration** of ingestion, transformation, AI enrichment, and downstream builds
- **Gemini-powered review enrichment** for sentiment, topics, scores, and key issues
- **RAG chatbot** for grounded questions about customer reviews
- **Text-to-SQL assistant** that converts business questions into Snowflake SQL
- **SELECT-only SQL guardrails** for safer natural-language analytics
- **Streamlit interfaces** for self-service analytics
- **dbt lineage** across the analytical data model
- **Snowflake role-based execution** using a restricted `DBT_ROLE`

---

# 🏗️ Architecture

![Zomato AI Intelligence Architecture](docs/architecture.png)

### High-level flow

```text
                    ZOMATO AI INTELLIGENCE
                           │
                           ▼
              ┌─────────────────────────┐
              │     SOURCE DATA         │
              │  Zomato CSV datasets    │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │       AMAZON S3         │
              │      Raw CSV files      │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │    SNOWFLAKE RAW        │
              │      Bronze layer       │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │      dbt STAGING        │
              │ clean • type • join     │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │       dbt MARTS         │
              │ facts • dimensions      │
              │ business marts          │
              └────────────┬────────────┘
                           │
                ┌──────────┴───────────┐
                ▼                      ▼
        ┌───────────────┐      ┌────────────────┐
        │   Streamlit   │      │   AI Layer     │
        │   Analytics   │      │ Gemini-powered │
        └───────────────┘      └────────────────┘
```

---

# 🧱 Data Architecture

The platform follows a layered warehouse architecture:

```text
SOURCE
  │
  ▼
S3
  │
  ▼
RAW / BRONZE
  │
  ▼
STAGING / SILVER
  │
  ▼
MARTS / GOLD
  │
  ▼
SERVE
```

### RAW / Bronze

Contains source-level data loaded into Snowflake with minimal transformation.

Example entities:

- `raw.food`
- `raw.users`
- `raw.orders`
- `raw.order_items`
- `raw.restaurants`
- `raw.menu`
- `raw.reviews`

### STAGING / Silver

dbt models standardize and clean the raw data.

Examples:

- `stage_food`
- `stage_users`
- `stage_orders`
- `stage_order_items`
- `stage_restaurants`
- `stage_menu`
- `stage_reviews`

Typical transformations include:

- data type standardization
- column cleanup
- joins
- derived fields
- business logic
- validation

### MARTS / Gold

Business-ready models designed for analytics and downstream applications.

Examples:

- `dim_customer`
- `dim_restaurants`
- `dim_food`
- `dim_date`
- `fact_orders`
- `fact_order_items`
- `mart_daily_city_revenue`
- `mart_restaurant_performance`
- `mart_delivery_sla`
- `marts_reviews_insights`

---

# 🔄 Data Lineage

![dbt Data Lineage](docs/lineage.png)

The core lineage follows patterns such as:

```text
raw.users
    │
    ▼
stage_users
    │
    ▼
dim_customer
```

```text
raw.orders
    │
    ▼
stage_orders
    │
    ├───────────────► fact_orders
    │                     │
    │                     ├──► mart_daily_city_revenue
    │                     ├──► mart_delivery_sla
    │                     └──► mart_restaurant_performance
    │
    ▼
fact_order_items
```

```text
raw.restaurants
    │
    ▼
stage_restaurants
    │
    ▼
dim_restaurants
    │
    └──────────────► mart_restaurant_performance
```

Reviews have an additional AI enrichment path:

```text
raw.reviews
    │
    ▼
stage_reviews
    │
    ├──────────────► marts_reviews_insights
    │
    ▼
Gemini enrichment
    │
    ▼
ai.review_enriched
    │
    └──────────────► marts_reviews_insights
```

---

# 🤖 AI Layer

The project contains three AI capabilities.

## 1. LLM Review Enrichment

The review enrichment pipeline uses Gemini to convert unstructured customer reviews into structured analytical attributes.

### Input

```text
"Great eco-friendly packaging. Delivered right on time."
```

### Gemini output

```json
{
  "sentiment_label": "positive",
  "sentiment_score": 0.9,
  "topic": "packaging",
  "key_issue": null
}
```

### Stored in Snowflake

```text
AI.REVIEW_ENRICHED
```

Example columns:

| Column | Description |
|---|---|
| `REVIEW_ID` | Original review identifier |
| `SENTIMENT_LABEL` | Positive / negative / neutral |
| `SENTIMENT_SCORE` | Score from -1 to 1 |
| `TOPIC` | Food, delivery, pricing, service, packaging, other |
| `KEY_ISSUE` | Short description of the main issue |
| `MODEL` | Gemini model used |
| `ENRICHED_AT` | Enrichment timestamp |

This turns unstructured text into structured features that can be used in SQL analytics and dashboards.

---

# 🔎 2. RAG — Chat With Your Reviews

The RAG application allows users to ask questions about customer reviews.

Example:

```text
"What are customers complaining about regarding delivery?"
```

The application:

```text
User question
      │
      ▼
Gemini embedding
      │
      ▼
Vector similarity search
      │
      ▼
Top-K relevant reviews
      │
      ▼
Gemini
      │
      ▼
Grounded answer
```

The answer is generated using the retrieved reviews rather than asking the LLM to rely on general knowledge.

Example:

```text
Question:
What are the main delivery complaints?

Retrieved reviews:
- Delivery was extremely late.
- Driver arrived 40 minutes after the ETA.
- Food arrived cold because of the delay.

Answer:
The main delivery complaints are delays, missed ETAs,
and food arriving cold after long delivery times.
```

The Streamlit interface also exposes the reviews used to construct the answer.

---

# 🧠 3. Text-to-SQL

The Text-to-SQL application lets business users query the warehouse using natural language.

Example:

```text
Top 10 cities by GMV
```

Gemini generates SQL such as:

```sql
SELECT
    city,
    SUM(gmv) AS total_gmv
FROM MART_DAILY_CITY_REVENUE
GROUP BY city
ORDER BY total_gmv DESC
LIMIT 10;
```

The generated SQL is displayed before execution.

### Safety layer

The application checks that generated SQL is read-only.

Blocked operations include:

```text
DROP
DELETE
TRUNCATE
ALTER
UPDATE
INSERT
CREATE
REPLACE
GRANT
REVOKE
```

The intended execution path is:

```text
Natural language
      │
      ▼
Gemini
      │
      ▼
Generated SQL
      │
      ▼
SELECT / WITH safety check
      │
      ▼
Snowflake
      │
      ▼
Pandas DataFrame
      │
      ├──► Table
      └──► Chart
```

The application executes queries using the restricted `DBT_ROLE` rather than an unrestricted administrative role.

---

# ⚙️ Orchestration with Apache Airflow

Airflow coordinates the main pipeline.

The DAG contains tasks following the general flow:

```text
upload_raw
     │
     ▼
dbt_build_core
     │
     ▼
enrich_reviews
     │
     ▼
dbt_build_all
```

### Task responsibilities

| Task | Purpose |
|---|---|
| `upload_raw` | Load source data into Snowflake RAW |
| `dbt_build_core` | Build staging/core dbt models |
| `enrich_reviews` | Run Gemini-based review enrichment |
| `dbt_build_all` | Build downstream marts including AI-derived insights |

This provides a single orchestration layer for both traditional data engineering and AI workloads.

---

# 🛠️ Technology Stack

| Technology | Role |
|---|---|
| **Python** | Pipeline and AI application logic |
| **Pandas** | Data processing |
| **Amazon S3** | Raw data lake/storage |
| **Snowflake** | Cloud data warehouse |
| **dbt** | Data transformation and modeling |
| **Apache Airflow** | Workflow orchestration |
| **Gemini API** | LLM + embeddings |
| **Streamlit** | AI/self-service analytics UI |
| **Docker** | Local containerized Airflow environment |
| **Git/GitHub** | Version control |

---

# 📁 Project Structure

```text
Zomato_AI-Intelligence/
│
├── ai/
│   ├── enrich_reviews.py
│   ├── rag_chat.py
│   └── text_to_sql.py
│
├── airflow/
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   └── dags/
│       └── zomato_batch.py
│
├── dbt/
│   └── zomato/
│       ├── models/
│       │   ├── staging/
│       │   └── marts/
│       └── dbt_project.yml
│
├── .gitignore
└── README.md
```

> Local `.env` files, generated embeddings/cache files, Airflow logs, Python bytecode, and other secrets/generated artifacts are intentionally excluded from Git.

---

# 🔐 Security

The project is designed with several safety controls.

### Secrets

Credentials are stored in environment variables rather than source code.

Examples:

```text
SNOWFLAKE_USER
SNOWFLAKE_PASSWORD
SNOWFLAKE_ACCOUNT
GEMINI_API_KEY
```

`.env` files are excluded from Git using `.gitignore`.

### Warehouse access

AI-generated SQL is executed using:

```text
DBT_ROLE
```

rather than an unrestricted administrative role.

### Text-to-SQL guardrail

Only:

```sql
SELECT
```

and:

```sql
WITH
```

queries are allowed.

Mutation and DDL operations are rejected before execution.

---

# 🧪 Example Business Questions

The Text-to-SQL application can answer questions such as:

```text
Top 10 cities by GMV

Which cuisine has the most orders?

Average delivery time by city, worst first

Cancel rate by payment method

Which restaurants have the highest revenue?

Which cities have the highest delivery late rate?

What are the most common complaints in customer reviews?
```

---

# 🚀 Running the Project Locally

## 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Zomato_AI-Intelligence
```

## 2. Configure environment variables

Create the required `.env` files from the provided examples.

Never commit real credentials.

Example:

```env
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema

GEMINI_API_KEY=your_gemini_api_key
```

## 3. Start Airflow

From the Airflow directory:

```bash
cd airflow
docker compose up -d
```

Check services:

```bash
docker compose ps
```

Open:

```text
http://localhost:8080
```

## 4. Run the Streamlit applications

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the required application, for example:

```bash
streamlit run ai/rag_chat.py
```

or:

```bash
streamlit run ai/text_to_sql.py
```

---

# 📊 Analytics Layer

The Gold/MART layer is designed around common business questions.

### Revenue

```text
mart_daily_city_revenue
```

Supports:

- GMV
- order volume
- AOV
- city-level revenue analysis
- cancellation analysis

### Restaurant performance

```text
mart_restaurant_performance
```

Supports:

- restaurant revenue
- order volume
- average rating
- cancellation rate
- cuisine analysis

### Delivery

```text
mart_delivery_sla
```

Supports:

- P50 delivery time
- late rate
- delivery performance by city
- delivery performance by hour

### Review intelligence

```text
marts_reviews_insights
```

Combines customer reviews with Gemini-derived attributes for:

- sentiment analysis
- topic analysis
- issue identification
- customer experience trends

---

# 🎯 Engineering Goals

This project demonstrates the ability to build across both **data engineering and AI engineering layers**.

### Data Engineering

- Data lake ingestion
- Snowflake warehouse design
- Bronze/Silver/Gold architecture
- Dimensional modeling
- Fact and dimension tables
- dbt transformations
- Data lineage
- Airflow orchestration
- Dockerized development

### AI Engineering

- LLM-powered structured extraction
- Embeddings
- Vector similarity search
- Retrieval-Augmented Generation
- Natural-language-to-SQL
- Prompt engineering
- Structured model output
- AI safety/SQL guardrails

### Analytics Engineering

- Reusable marts
- Business metrics
- Self-service analytics
- Streamlit applications
- SQL-driven reporting

---

# 🔮 Future Improvements

Potential next steps include:

- Add automated dbt tests and data quality checks
- Add Airflow retries and alerting
- Add incremental ingestion from S3
- Replace local RAG cache with a production vector database
- Add hybrid keyword + vector retrieval
- Add review topic clustering
- Add evaluation datasets for RAG and Text-to-SQL
- Add query cost controls for AI-generated SQL
- Add authentication to Streamlit applications
- Add CI/CD with GitHub Actions
- Deploy the platform to a cloud environment
- Add observability and pipeline SLA monitoring

---

# 👤 Author

**Teeja**

Data Engineering • Analytics Engineering • AI/ML

This project demonstrates an end-to-end approach to building modern data platforms that combine reliable warehouse engineering with practical generative AI capabilities.
