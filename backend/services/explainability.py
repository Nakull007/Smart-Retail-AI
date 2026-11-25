from __future__ import annotations

import pandas as pd
import shap
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import Product, Sale
from backend.services.feature_engineering import make_next_day_features, normalize_daily_history
from backend.services.forecasting import load_model_bundle


def explain_next_day_forecast(db: Session, product_id: int) -> dict:
    product = db.get(Product, product_id)
    if product is None:
        raise ValueError("Product not found.")

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
    features = make_next_day_features(
        product_id=product.id,
        history=daily["units_sold"].astype(float).tolist(),
        target_date=pd.Timestamp(daily.index[-1]) + pd.Timedelta(days=1),
        unit_price=float(daily["unit_price"].iloc[-1]),
        discount_pct=0.0,
    )
    model, metadata = load_model_bundle()
    prediction = max(0.0, float(model.predict(features)[0]))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    values = shap_values[0].tolist()
    contributions = [
        {
            "feature": feature,
            "input_value": float(features.iloc[0][feature]),
            "shap_value": float(value),
        }
        for feature, value in zip(features.columns, values)
    ]
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    base_value = explainer.expected_value
    if hasattr(base_value, "tolist"):
        base_value = base_value.tolist()
    if isinstance(base_value, list):
        base_value = base_value[0]
    return {
        "product_id": product.id,
        "product_name": product.name,
        "model": metadata["best_model"],
        "forecast_date": (pd.Timestamp(daily.index[-1]) + pd.Timedelta(days=1)).date().isoformat(),
        "predicted_units": round(prediction, 1),
        "baseline_value": float(base_value),
        "top_feature_contributions": contributions[:8],
        "note": "SHAP contributions explain this model prediction relative to the model baseline; they do not establish causation.",
    }
