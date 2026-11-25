from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import select
from xgboost import XGBRegressor
import joblib

from backend.config import settings
from backend.database import SessionLocal, init_db
from backend.models import Product, Sale
from backend.services.feature_engineering import FEATURE_COLUMNS, make_training_rows, normalize_daily_history


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.abs(y_true).sum())
    return float(np.abs(y_true - y_pred).sum() / denominator * 100) if denominator else 0.0


def main() -> None:
    init_db()
    with SessionLocal() as db:
        products = db.scalars(select(Product).order_by(Product.id)).all()
        rows: list[pd.DataFrame] = []
        for product in products:
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
            if sales_df.empty:
                continue
            daily = normalize_daily_history(sales_df, product.base_price)
            product_rows = make_training_rows(product.id, daily)
            if not product_rows.empty:
                rows.append(product_rows)

    if not rows:
        raise RuntimeError("No training rows found. Run `python -m backend.seed` first.")

    dataset = pd.concat(rows).sort_index()
    # Hold out the most recent 30 calendar days to avoid random-split leakage.
    all_dates = pd.to_datetime(dataset.index)
    cutoff = all_dates.max() - pd.Timedelta(days=30)
    train_mask = all_dates <= cutoff
    test_mask = all_dates > cutoff
    X_train = dataset.loc[train_mask, FEATURE_COLUMNS]
    y_train = dataset.loc[train_mask, "target"]
    X_test = dataset.loc[test_mask, FEATURE_COLUMNS]
    y_test = dataset.loc[test_mask, "target"]

    if X_test.empty:
        raise RuntimeError("Not enough historical data for the 30-day validation window.")

    models = {
        "xgboost": XGBRegressor(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=-1,
        ),
    }

    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions = np.clip(model.predict(X_test), 0, None)
        metrics[name] = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "wape_percent": wape(y_test.to_numpy(), predictions),
        }
        joblib.dump(model, settings.ARTIFACTS_DIR / f"{name}_model.joblib")
        print(f"{name}: {metrics[name]}")

    best_name = min(metrics, key=lambda key: metrics[key]["mae"])
    metadata = {
        "best_model": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "validation": "Most recent 30 calendar days; chronological split",
    }
    (settings.ARTIFACTS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Training complete. Best model by MAE: {best_name}")
    print(f"Saved artifacts to: {Path(settings.ARTIFACTS_DIR).resolve()}")


if __name__ == "__main__":
    main()
