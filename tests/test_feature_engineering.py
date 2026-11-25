from datetime import date, timedelta

import pandas as pd

from backend.services.feature_engineering import FEATURE_COLUMNS, make_next_day_features, make_training_rows, normalize_daily_history


def test_daily_history_fills_missing_dates():
    today = date(2025, 1, 1)
    frame = pd.DataFrame(
        [
            {"sale_date": today, "units_sold": 4, "unit_price": 10.0, "discount_pct": 0.0},
            {"sale_date": today + timedelta(days=2), "units_sold": 6, "unit_price": 9.0, "discount_pct": 0.1},
        ]
    )
    result = normalize_daily_history(frame, base_price=10.0)
    assert len(result) == 3
    assert result["units_sold"].tolist() == [4.0, 0.0, 6.0]


def test_training_and_forecast_feature_columns_match():
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    daily = pd.DataFrame(
        {
            "units_sold": [float((i % 10) + 1) for i in range(60)],
            "unit_price": [10.0] * 60,
            "discount_pct": [0.0] * 60,
        },
        index=dates,
    )
    rows = make_training_rows(product_id=1, daily=daily)
    forecast_features = make_next_day_features(1, daily["units_sold"].tolist(), pd.Timestamp("2025-03-02"), 10.0)
    assert list(rows[FEATURE_COLUMNS].columns) == list(forecast_features.columns)
    assert forecast_features.shape == (1, len(FEATURE_COLUMNS))
