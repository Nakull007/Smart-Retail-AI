from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Smart Retail AI", page_icon="🛒", layout="wide")
st.title("🛒 Smart Retail AI")
st.caption("Demand forecasting · Inventory optimization · RAG · Gemini retail assistant")


def api_get(path: str, params: dict | None = None):
    response = requests.get(f"{API_BASE}{path}", params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict | None = None, files: dict | None = None):
    response = requests.post(
        f"{API_BASE}{path}",
        json=payload,
        files=files,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


try:
    health = requests.get(f"{API_BASE}/health", timeout=5)
    health.raise_for_status()
    st.sidebar.success("Backend connected")
except requests.RequestException:
    st.sidebar.error("Backend is not reachable. Start FastAPI first.")
    st.sidebar.code("uvicorn backend.main:app --reload")
    st.stop()

pages = st.tabs(["Overview", "Demand Forecast", "Inventory", "AI Assistant", "Knowledge Base", "Review Analysis"])

with pages[0]:
    st.subheader("Store overview")
    try:
        summary = api_get("/api/dashboard/summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Products", summary["total_products"])
        c2.metric("Units sold · 30 days", f"{summary['units_sold_last_30_days']:,}")
        c3.metric("Revenue · 30 days", f"₹{summary['revenue_last_30_days']:,.0f}")
        c4.metric("Low-stock products", summary["low_stock_products"])
        trend = api_get("/api/sales/trend", params={"days": 30})
        if trend:
            trend_df = pd.DataFrame(trend)
            fig = px.line(trend_df, x="date", y="units_sold", title="Daily units sold · Last 30 days", markers=True)
            st.plotly_chart(fig, use_container_width=True)
    except requests.RequestException as exc:
        st.error(f"Could not load dashboard: {exc}")

with pages[1]:
    st.subheader("Product demand forecasting")
    try:
        products = api_get("/api/products")
        if products:
            options = {f"{p['name']} ({p['sku']})": p["id"] for p in products}
            selected_label = st.selectbox("Select product", list(options.keys()))
            horizon = st.slider("Forecast horizon (days)", 1, 30, 7)
            if st.button("Generate forecast"):
                rows = api_get("/api/forecast", params={"days": horizon, "product_id": options[selected_label]})
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                if not df.empty:
                    fig = px.bar(df, x="forecast_date", y="predicted_units", title="Predicted units by day")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Model selected using validation MAE: {df['model'].iloc[0]}")
                    with st.expander("Explain the next-day prediction (SHAP)"):
                        explanation = api_get("/api/forecast/explain", params={"product_id": options[selected_label]})
                        st.write(explanation["note"])
                        st.metric("Next-day prediction", explanation["predicted_units"])
                        st.dataframe(pd.DataFrame(explanation["top_feature_contributions"]), use_container_width=True, hide_index=True)
        else:
            st.info("No products found. Run the seed script first.")
    except requests.RequestException as exc:
        st.error(f"Forecast request failed: {exc}")

with pages[2]:
    st.subheader("Inventory reorder recommendations")
    if st.button("Refresh recommendations"):
        st.session_state["inventory_refresh"] = True
    if st.session_state.get("inventory_refresh", True):
        try:
            rows = api_get("/api/inventory/recommendations")
            df = pd.DataFrame(rows)
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
                reorder_count = int((df["status"] == "REORDER").sum())
                st.metric("Products requiring reorder", reorder_count)
            else:
                st.info("No inventory rows found.")
        except requests.RequestException as exc:
            st.error(f"Inventory request failed: {exc}")

with pages[3]:
    st.subheader("Ask SmartRetail Copilot")
    st.write("Ask about stock, demand forecasts, top-selling products, or uploaded supplier and inventory policies.")
    question = st.text_area("Your question", placeholder="Which products should I reorder this week?", height=90)
    if st.button("Ask assistant", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Thinking and checking retail data..."):
                    result = api_post("/api/chat", payload={"question": question.strip()})
                st.markdown(result["answer"])
                if result.get("tools_used"):
                    with st.expander("Tools used by the agent"):
                        st.json(result["tools_used"])
            except requests.HTTPError as exc:
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                st.error(detail)
            except requests.RequestException as exc:
                st.error(f"Assistant request failed: {exc}")

with pages[4]:
    st.subheader("Retail knowledge base (RAG)")
    st.write("Upload supplier agreements, inventory policies, product manuals or store guidelines. The app chunks the documents, embeds them and stores vectors in PostgreSQL/pgvector.")
    uploaded = st.file_uploader("Upload a PDF, DOCX, TXT or MD file", type=["pdf", "docx", "txt", "md"])
    if uploaded is not None and st.button("Index document"):
        try:
            with st.spinner("Extracting text and creating embeddings..."):
                result = api_post(
                    "/api/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                )
            st.success(f"Indexed {result['chunks_stored']} chunks from {result['filename']}.")
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            st.error(detail)
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")
    query = st.text_input("Search indexed documents", placeholder="What is the recommended safety stock policy?")
    if st.button("Search knowledge base"):
        if query.strip():
            try:
                result = api_post("/api/documents/search", payload={"query": query.strip(), "top_k": 5})
                for index, item in enumerate(result["results"], start=1):
                    with st.expander(f"{index}. {item['source']} · chunk {item['chunk_index']}"):
                        st.write(item["content"])
                if not result["results"]:
                    st.info("No indexed document chunks found. Upload a document first.")
            except requests.RequestException as exc:
                st.error(f"Search failed: {exc}")

with pages[5]:
    st.subheader("Customer review sentiment")
    st.caption("The first analysis downloads the Hugging Face DistilBERT model; allow time for the initial download.")
    try:
        products = api_get("/api/products")
        product_options = {"General feedback": None}
        product_options.update({p["name"]: p["id"] for p in products})
        selected_product = st.selectbox("Related product (optional)", list(product_options.keys()))
        review_text = st.text_area("Customer review", placeholder="The product is good, but delivery was late.", height=100)
        if st.button("Analyze sentiment"):
            if not review_text.strip():
                st.warning("Enter a review first.")
            else:
                result = api_post(
                    "/api/reviews/analyze",
                    payload={"review_text": review_text.strip(), "product_id": product_options[selected_product]},
                )
                st.metric("Sentiment", result["label"], f"Confidence {result['score']:.1%}")
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(detail)
    except requests.RequestException as exc:
        st.error(f"Could not analyze review: {exc}")

st.sidebar.caption(f"API: {API_BASE}")
st.sidebar.caption("Local demo data is synthetic. Forecasts are estimates, not purchasing instructions.")
