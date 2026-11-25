from __future__ import annotations

import json
from datetime import timedelta
from functools import lru_cache

import joblib
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models import Product, Sale
from backend.services.feature_engineering import make_next_day_features, normalize_daily_history


@lru_cache(maxsize=1)
def load_model_bundle():
    metadata_path = settings.ARTIFACTS_DIR / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Trained models are missing. Run `python -m backend.scripts.train_models` first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_name = metadata["best_model"]
    model_path = settings.ARTIFACTS_DIR / f"{model_name}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path), metadata


def forecast_product(db: Session, product: Product, days: int = 7) -> list[dict]:
    if not 1 <= days <= 30:
        raise ValueError("days must be between 1 and 30")
    model, metadata = load_model_bundle()
    sales = db.scalars(
        select(Sale).where(Sale.product_id == product.id).order_by(Sale.sale_date)
    ).all()
    sales_df = pd.DataFrame(
        [
            {
                "sale_date": sale.sale_date,
                "units_sold": sale.units_sold,
                "unit_price": sale.unit_price,
                "discount_pct": sale.discount_pct,
            }
            for sale in sales
        ]
    )
    daily = normalize_daily_history(sales_df, product.base_price)
    history = daily["units_sold"].astype(float).tolist()
    latest_price = float(daily["unit_price"].iloc[-1])
    next_date = pd.Timestamp(daily.index[-1]) + pd.Timedelta(days=1)

    results: list[dict] = []
    for _ in range(days):
        features = make_next_day_features(
            product_id=product.id,
            history=history,
            target_date=next_date,
            unit_price=latest_price,
            discount_pct=0.0,
        )
        prediction = max(0.0, float(model.predict(features)[0]))
        rounded = round(prediction, 1)
        results.append(
            {
                "product_id": product.id,
                "product_name": product.name,
                "forecast_date": next_date.date().isoformat(),
                "predicted_units": rounded,
                "model": metadata["best_model"],
            }
        )
        history.append(prediction)
        next_date += pd.Timedelta(days=1)
    return results


def forecast_all(db: Session, days: int = 7, product_id: int | None = None) -> list[dict]:
    query = select(Product).order_by(Product.name)
    if product_id is not None:
        query = query.where(Product.id == product_id)
    products = db.scalars(query).all()
    output: list[dict] = []
    for product in products:
        output.extend(forecast_product(db, product, days=days))
    return output
