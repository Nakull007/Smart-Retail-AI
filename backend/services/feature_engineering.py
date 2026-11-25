from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "product_id",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "unit_price",
    "discount_pct",
]


def normalize_daily_history(sales_df: pd.DataFrame, base_price: float) -> pd.DataFrame:
    """Return a continuous daily series, filling days with no sales as zero."""
    if sales_df.empty:
        raise ValueError("No sales history is available for this product.")

    frame = sales_df.copy()
    frame["sale_date"] = pd.to_datetime(frame["sale_date"])
    frame = frame.groupby("sale_date", as_index=True).agg(
        units_sold=("units_sold", "sum"),
        unit_price=("unit_price", "mean"),
        discount_pct=("discount_pct", "mean"),
    )
    full_dates = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    frame = frame.reindex(full_dates)
    frame.index.name = "sale_date"
    frame["units_sold"] = frame["units_sold"].fillna(0).astype(float)
    frame["unit_price"] = frame["unit_price"].ffill().bfill().fillna(base_price)
    frame["discount_pct"] = frame["discount_pct"].fillna(0.0)
    return frame


def make_training_rows(product_id: int, daily: pd.DataFrame) -> pd.DataFrame:
    """Create supervised lag/rolling features without leaking future sales."""
    frame = daily.copy()
    target = frame["units_sold"]
    frame["product_id"] = product_id
    frame["lag_1"] = target.shift(1)
    frame["lag_7"] = target.shift(7)
    frame["lag_14"] = target.shift(14)
    frame["lag_28"] = target.shift(28)
    frame["rolling_mean_7"] = target.shift(1).rolling(7).mean()
    frame["rolling_mean_28"] = target.shift(1).rolling(28).mean()
    frame["day_of_week"] = frame.index.dayofweek
    frame["day_of_month"] = frame.index.day
    frame["month"] = frame.index.month
    frame["is_weekend"] = (frame.index.dayofweek >= 5).astype(int)
    frame["target"] = target
    frame = frame.dropna(subset=FEATURE_COLUMNS + ["target"])
    return frame[FEATURE_COLUMNS + ["target"]]


def make_next_day_features(
    product_id: int,
    history: list[float],
    target_date: pd.Timestamp,
    unit_price: float,
    discount_pct: float = 0.0,
) -> pd.DataFrame:
    """Create the same features used at training time for one future day."""
    if len(history) < 28:
        raise ValueError("At least 28 days of sales history are required to forecast.")
    row = {
        "product_id": product_id,
        "lag_1": history[-1],
        "lag_7": history[-7],
        "lag_14": history[-14],
        "lag_28": history[-28],
        "rolling_mean_7": sum(history[-7:]) / 7,
        "rolling_mean_28": sum(history[-28:]) / 28,
        "day_of_week": target_date.dayofweek,
        "day_of_month": target_date.day,
        "month": target_date.month,
        "is_weekend": int(target_date.dayofweek >= 5),
        "unit_price": unit_price,
        "discount_pct": discount_pct,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
