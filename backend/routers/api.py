from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from backend.config import settings
from backend.database import get_db
from backend.models import Inventory, Product, Review, Sale
from backend.schemas import ChatRequest, ChatResponse, DocumentSearchRequest, ReviewRequest
from backend.services.forecasting import forecast_all
from backend.services.explainability import explain_next_day_forecast
from backend.services.inventory import get_reorder_recommendations
from backend.services.llm_agent import ask_retail_agent
from backend.services.rag import ingest_document, retrieve_chunks
from backend.services.sentiment import analyze_sentiment

router = APIRouter(prefix="/api")


@router.get("/products")
def list_products(db: Session = Depends(get_db)):
    products = db.scalars(select(Product).order_by(Product.name)).all()
    return [
        {"id": p.id, "sku": p.sku, "name": p.name, "category": p.category, "base_price": p.base_price}
        for p in products
    ]


@router.get("/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db)):
    start_date = date.today() - timedelta(days=30)
    total_products = db.scalar(select(func.count(Product.id))) or 0
    units_30d = db.scalar(select(func.coalesce(func.sum(Sale.units_sold), 0)).where(Sale.sale_date >= start_date)) or 0
    revenue_30d = db.scalar(
        select(func.coalesce(func.sum(Sale.units_sold * Sale.unit_price), 0)).where(Sale.sale_date >= start_date)
    ) or 0.0
    inventory_rows = db.scalars(select(Inventory).options(joinedload(Inventory.product))).all()
    low_stock = 0
    total_stock = 0
    for item in inventory_rows:
        total_stock += item.stock_qty
        recent_units = db.scalar(
            select(func.coalesce(func.avg(Sale.units_sold), 0))
            .where(Sale.product_id == item.product_id, Sale.sale_date >= date.today() - timedelta(days=30))
        ) or 0
        if item.stock_qty < float(recent_units) * item.lead_time_days + item.safety_stock:
            low_stock += 1
    return {
        "total_products": int(total_products),
        "units_sold_last_30_days": int(units_30d),
        "revenue_last_30_days": round(float(revenue_30d), 2),
        "total_units_in_stock": int(total_stock),
        "low_stock_products": low_stock,
    }


@router.get("/sales/trend")
def sales_trend(days: int = 30, db: Session = Depends(get_db)):
    days = max(7, min(days, 180))
    start_date = date.today() - timedelta(days=days - 1)
    rows = db.execute(
        select(Sale.sale_date, func.sum(Sale.units_sold).label("units"))
        .where(Sale.sale_date >= start_date)
        .group_by(Sale.sale_date)
        .order_by(Sale.sale_date)
    ).all()
    return [{"date": row.sale_date.isoformat(), "units_sold": int(row.units or 0)} for row in rows]


@router.get("/forecast")
def get_forecast(days: int = 7, product_id: int | None = None, db: Session = Depends(get_db)):
    try:
        return forecast_all(db, days=days, product_id=product_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/forecast/explain")
def forecast_explanation(product_id: int, db: Session = Depends(get_db)):
    try:
        return explain_next_day_forecast(db, product_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {exc}") from exc


@router.get("/inventory/recommendations")
def inventory_recommendations(db: Session = Depends(get_db)):
    try:
        return get_reorder_recommendations(db)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = Path(file.filename or "upload.txt").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Supported file types: PDF, DOCX, TXT and MD.")
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_MB} MB.")
    try:
        chunks = ingest_document(db, filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}") from exc
    return {"filename": filename, "chunks_stored": chunks}


@router.post("/documents/search")
def search_documents(payload: DocumentSearchRequest, db: Session = Depends(get_db)):
    try:
        results = retrieve_chunks(db, payload.query, payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document search failed: {exc}") from exc
    return {"query": payload.query, "results": results}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    try:
        answer, tools_used = ask_retail_agent(payload.question, db)
        return ChatResponse(answer=answer, tools_used=tools_used)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}") from exc


@router.post("/reviews/analyze")
def review_analyze(payload: ReviewRequest, db: Session = Depends(get_db)):
    if payload.product_id is not None and db.get(Product, payload.product_id) is None:
        raise HTTPException(status_code=404, detail="Product not found.")
    try:
        result = analyze_sentiment(payload.review_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sentiment model failed to load or predict: {exc}") from exc
    review = Review(
        product_id=payload.product_id,
        review_text=payload.review_text,
        sentiment_label=result["label"],
        sentiment_score=result["score"],
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return {"id": review.id, **result}
