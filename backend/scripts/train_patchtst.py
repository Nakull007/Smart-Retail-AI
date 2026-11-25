"""Optional Hugging Face PatchTST experiment.

This trains a small PatchTST model from scratch on the demo sales series.
It is intentionally separate from the required XGBoost/LightGBM pipeline so
that the core application remains faster to set up and easier to debug.
"""
from __future__ import annotations

from datetime import timedelta

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSTConfig, PatchTSTForPrediction
from sqlalchemy import select

from backend.config import settings
from backend.database import SessionLocal, init_db
from backend.models import Product, Sale
from backend.services.feature_engineering import normalize_daily_history

CONTEXT_LENGTH = 64
PREDICTION_LENGTH = 7
BATCH_SIZE = 64
EPOCHS = 5


def collect_windows():
    init_db()
    train_past, train_future = [], []
    test_past, test_future = [], []
    with SessionLocal() as db:
        products = db.scalars(select(Product).order_by(Product.id)).all()
        if not products:
            raise RuntimeError("No products found. Run `python -m backend.seed` first.")

        latest_date = db.scalar(select(Sale.sale_date).order_by(Sale.sale_date.desc()).limit(1))
        if latest_date is None:
            raise RuntimeError("No sales rows found. Run `python -m backend.seed` first.")
        validation_start = latest_date - timedelta(days=28)

        for product in products:
            sales = db.scalars(
                select(Sale).where(Sale.product_id == product.id).order_by(Sale.sale_date)
            ).all()
            sales_df = [
                {
                    "sale_date": row.sale_date,
                    "units_sold": row.units_sold,
                    "unit_price": row.unit_price,
                    "discount_pct": row.discount_pct,
                }
                for row in sales
            ]
            if not sales_df:
                continue
            daily = normalize_daily_history(pd.DataFrame(sales_df), product.base_price)
            values = daily["units_sold"].to_numpy(dtype=np.float32)
            dates = daily.index
            last_start = len(values) - CONTEXT_LENGTH - PREDICTION_LENGTH
            for start in range(max(0, last_start + 1)):
                target_start_index = start + CONTEXT_LENGTH
                past = values[start : start + CONTEXT_LENGTH]
                future = values[target_start_index : target_start_index + PREDICTION_LENGTH]
                target_start_date = dates[target_start_index].date()
                if target_start_date >= validation_start:
                    test_past.append(past)
                    test_future.append(future)
                else:
                    train_past.append(past)
                    train_future.append(future)

    if not train_past or not test_past:
        raise RuntimeError("Not enough sales history to build PatchTST training and validation windows.")

    def shape(values):
        return torch.tensor(np.asarray(values, dtype=np.float32)[:, :, None], dtype=torch.float32)

    return shape(train_past), shape(train_future), shape(test_past), shape(test_future)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    train_past, train_future, test_past, test_future = collect_windows()

    config = PatchTSTConfig(
        num_input_channels=1,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        patch_length=16,
        patch_stride=8,
        d_model=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        ffn_dim=128,
        ff_dropout=0.1,
        head_dropout=0.1,
        scaling="std",
        loss="mse",
    )
    model = PatchTSTForPrediction(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mask = torch.ones_like(train_past, dtype=torch.bool)
    dataset = TensorDataset(train_past, train_future, mask)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for past_batch, future_batch, mask_batch in loader:
            optimizer.zero_grad()
            output = model(
                past_values=past_batch,
                past_observed_mask=mask_batch,
                future_values=future_batch,
            )
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"Epoch {epoch + 1}/{EPOCHS} - training MSE: {np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(
            past_values=test_past,
            past_observed_mask=torch.ones_like(test_past, dtype=torch.bool),
        ).prediction_outputs
    predictions = predictions.squeeze(-1).cpu().numpy()
    actual = test_future.squeeze(-1).cpu().numpy()
    mae = float(np.mean(np.abs(actual - predictions)))
    rmse = float(np.sqrt(np.mean((actual - predictions) ** 2)))

    output_dir = settings.ARTIFACTS_DIR / "patchtst_retail"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model": "PatchTSTForPrediction (trained from scratch)",
                "context_length": CONTEXT_LENGTH,
                "prediction_length": PREDICTION_LENGTH,
                "epochs": EPOCHS,
                "validation_mae": mae,
                "validation_rmse": rmse,
                "note": "Experimental benchmark on synthetic data; compare on real data before drawing conclusions.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"PatchTST validation MAE: {mae:.3f}; RMSE: {rmse:.3f}")
    print(f"Saved PatchTST model to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
