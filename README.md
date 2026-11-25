# Smart Retail AI

A local end-to-end portfolio project combining retail demand forecasting, inventory reorder recommendations, Hugging Face NLP, Gemini tool calling, RAG, PostgreSQL/pgvector, FastAPI and Streamlit.

> **Important:** The included sales data is synthetic and generated for development/testing. Forecasts are educational estimates, not real purchasing advice.

## Features

- Seed 12 sample retail products and two years of daily synthetic sales.
- Train and compare XGBoost and LightGBM using a chronological holdout window.
- Forecast product demand recursively for 1–30 days.
- Recommend reorder quantities using predicted demand during lead time plus safety stock.
- Explain an individual next-day prediction with SHAP feature contributions.
- Upload PDF, DOCX, TXT and Markdown documents; extract, chunk and embed them with Sentence Transformers.
- Store 384-dimensional embeddings in PostgreSQL with pgvector and retrieve relevant chunks by cosine distance.
- Ask a Gemini-powered retail copilot questions. The model can call allow-listed tools for inventory, demand forecasts, top-selling products and policy-document retrieval.
- Analyze customer review sentiment with a Hugging Face DistilBERT model.
- Explore metrics and forecasts in a Streamlit dashboard.

## 1. Prerequisites

Install these before starting:

1. **Python 3.11** (recommended; add Python to PATH during installation).
2. **Docker Desktop** for Windows, with Docker Compose enabled.
3. **Git** (optional, only if you want to put the project in a repository).
4. A **Gemini API key** from Google AI Studio for the conversational agent. The forecasting, dashboard, document indexing and sentiment endpoints can be used without a Gemini key.

The first run downloads the Sentence Transformers embedding model and the DistilBERT sentiment model from Hugging Face. Internet access is needed for those downloads. CPU execution is supported; a GPU is not required.

## 2. Create the Python environment (Windows PowerShell)

Open PowerShell in the `smart-retail-ai` project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If you use Command Prompt instead:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure environment variables

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set `GEMINI_API_KEY` to your private key. Do not commit `.env` to GitHub. Keep `DATABASE_URL` as shown if you use the included Docker Compose database.

`GEMINI_MODEL` is configurable. The example uses `gemini-3.8-flash`; model availability can change, so update this one setting if your Google AI Studio project exposes a different supported model.

## 4. Start PostgreSQL with pgvector

Make sure Docker Desktop is running, then run:

```powershell
docker compose up -d db
docker compose ps
```

Wait until the database reports healthy. The database is available at `localhost:5432` with the credentials in `docker-compose.yml`.

## 5. Create tables and insert demo data

From the project root, with `.venv` activated:

```powershell
python -m backend.seed
```

This command enables the pgvector extension, creates database tables, and inserts 12 products plus approximately 8,760 daily sales records. It is safe to run again: it skips insertion if products already exist.

## 6. Train the ML models

```powershell
python -m backend.scripts.train_models
```

The script:

1. Loads daily sales from PostgreSQL.
2. Creates lag features (1, 7, 14 and 28 days), rolling means and calendar features.
3. Uses the most recent 30 calendar days as a chronological holdout set.
4. Trains XGBoost and LightGBM.
5. Reports MAE, RMSE and WAPE for both models.
6. Saves both models and selects the lower-MAE model for forecasting.

Model files are saved under `artifacts/`. The `artifacts/` folder is generated locally and is not committed by default.

### Optional: train a Hugging Face PatchTST time-series Transformer

After the core XGBoost/LightGBM baseline works, run:

```powershell
python -m backend.scripts.train_patchtst
```

This optional experiment trains `PatchTSTForPrediction` from scratch on 64-day context windows to predict the next 7 days. It saves the model and validation MAE/RMSE under `artifacts/patchtst_retail/`. It is deliberately not wired into the production forecast endpoint: the demo data is synthetic, and this Transformer must be benchmarked against the tabular models using the same real-data backtest before selecting it.

## 7. Start the FastAPI backend

In terminal 1:

```powershell
uvicorn backend.main:app --reload
```

Open:

- API root: http://127.0.0.1:8000
- Swagger API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

The backend creates/validates database tables at startup. Keep this terminal running.

## 8. Start the Streamlit dashboard

Open terminal 2 in the same project folder, activate the same virtual environment, and run:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

Streamlit normally opens http://localhost:8501 automatically.

## 9. Try the application

### Demand forecast

1. Open the **Demand Forecast** tab.
2. Select a product.
3. Choose 1–30 forecast days.
4. Click **Generate forecast**.

### Inventory optimization

Open the **Inventory** tab and click **Refresh recommendations**. The reorder quantity is calculated as:

`max(0, forecast demand during supplier lead time + safety stock - current stock)`

### RAG knowledge base

1. Open **Knowledge Base**.
2. Upload `data/demo_docs/inventory_policy.txt` (or a PDF/DOCX/TXT/MD file).
3. Click **Index document**.
4. Ask a semantic search question such as: `How should the store calculate its reorder point?`

The first embedding request downloads `sentence-transformers/all-MiniLM-L6-v2`.

### Gemini retail assistant

After adding a valid `GEMINI_API_KEY`, open **AI Assistant** and try:

- `Which products have the lowest stock?`
- `Forecast demand for Whole Milk 1L for the next 7 days.`
- `What were the top-selling products over the last 30 days?`
- `What does the inventory policy say about reorder planning?`

The assistant uses function calling to select from the backend's allow-listed read-only tools. The policy search tool retrieves relevant chunks from pgvector before the LLM writes its answer.

### Review sentiment

Open **Review Analysis** and enter a review, for example:

`The product quality is great, but delivery was very slow.`

The first call downloads the DistilBERT sentiment model. The base model is a general positive/negative classifier; it does not perform aspect-based sentiment extraction.

## 10. Useful API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend health |
| GET | `/api/products` | Product list |
| GET | `/api/dashboard/summary` | Dashboard KPIs |
| GET | `/api/sales/trend?days=30` | Daily sales trend |
| GET | `/api/forecast?days=7&product_id=1` | Demand forecast |
| GET | `/api/forecast/explain?product_id=1` | SHAP explanation for the next-day prediction |
| GET | `/api/inventory/recommendations` | Reorder recommendations |
| POST | `/api/documents/upload` | Index a PDF/DOCX/TXT/MD file |
| POST | `/api/documents/search` | Semantic document retrieval |
| POST | `/api/chat` | Gemini agent with tools |
| POST | `/api/reviews/analyze` | DistilBERT sentiment |

## 11. Troubleshooting

### `docker` is not recognized
Install/start Docker Desktop, close and reopen PowerShell, then run `docker --version`.

### Database connection refused
Run `docker compose ps`. If the database is stopped, run `docker compose up -d db`. Check that port 5432 is not already being used by another PostgreSQL service.

### `CREATE EXTENSION vector` fails
Confirm the Compose service uses `pgvector/pgvector:pg16`, not the standard `postgres` image. Then restart the database service.

### Forecast endpoint says model files are missing
Run `python -m backend.scripts.train_models` after seeding the database.

### Gemini returns an API/model error
Check that `GEMINI_API_KEY` is valid, billing/quota and API access are enabled as required, and `GEMINI_MODEL` names a model available to your API key. Restart FastAPI after editing `.env`.

### Hugging Face model download fails
Check your internet connection and retry. The first download can take time. You can pre-download the models while connected to the internet; later runs normally reuse the local Hugging Face cache.

### Port 5432 is already in use
Change the left side of the Compose port mapping to `5433:5432`, then update `DATABASE_URL` in `.env` to use port `5433`.

## 12. Technical notes and honest limitations

- The demo sales dataset is synthetic. It is designed to make the application runnable and to demonstrate the pipeline, not to prove real retail forecasting performance.
- The API forecaster is a global tabular model trained across the sample products. It uses recursive multi-step predictions, so uncertainty accumulates as the horizon grows. PatchTST is an optional separate experiment and is not used by the live API forecast endpoint.
- Promotions, holidays, product launches, stockouts, supplier constraints and external events are not fully modelled. Add these variables when using real data.
- Demand forecasting is regression. It should be evaluated using a time-based split, MAE/RMSE and a scale-aware metric such as WAPE. Do not use random train/test splitting for time series.
- The RAG collection stores document chunks and embeddings. It does not currently implement hybrid keyword/vector retrieval, reranking, access control or document-level deletion UI.
- DistilBERT's default sentiment model only predicts positive/negative sentiment; it is not an aspect-based review classifier.
- PatchTST is a possible advanced extension. It is intentionally not in the required startup path: first establish a reliable XGBoost/LightGBM baseline, then add and benchmark a time-series Transformer on real retail history.
- The agent tools in this demo are read-only. They do not place orders, modify stock, or contact suppliers.

## 13. Suggested improvements for a stronger portfolio version

1. Replace generated data with real historical sales and add stockout-aware demand labels.
2. Add holiday, price, promotion and weather features.
3. Add a backtesting routine across multiple rolling time windows.
4. Add SHAP explanations to compare why XGBoost/LightGBM forecast demand changes.
5. Add hybrid retrieval (BM25 + vectors) and a reranker for RAG.
6. Add role-based authentication, request logging, API tests and CI.
7. Fine-tune or benchmark PatchTST against the tabular ML models using the same chronological test windows.

## References

- Google Gen AI Python SDK: https://googleapis.github.io/python-genai/
- pgvector Python / SQLAlchemy: https://github.com/pgvector/pgvector-python
- Hugging Face Transformers pipelines: https://huggingface.co/docs/transformers/en/main_classes/pipelines
- Sentence Transformers: https://www.sbert.net/
