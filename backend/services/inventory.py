from __future__ import annotations

import math

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from backend.models import Inventory
from backend.services.forecasting import forecast_product


def get_reorder_recommendations(db: Session) -> list[dict]:
    inventory_rows = db.scalars(
        select(Inventory).options(joinedload(Inventory.product)).order_by(Inventory.id)
    ).all()
    recommendations: list[dict] = []
    for item in inventory_rows:
        horizon = max(1, min(item.lead_time_days, 30))
        forecasts = forecast_product(db, item.product, days=horizon)
        expected_demand = sum(row["predicted_units"] for row in forecasts)
        reorder_point = expected_demand + item.safety_stock
        order_qty = max(0, math.ceil(reorder_point - item.stock_qty))
        recommendations.append(
            {
                "product_id": item.product_id,
                "product_name": item.product.name,
                "category": item.product.category,
                "stock_qty": item.stock_qty,
                "lead_time_days": item.lead_time_days,
                "safety_stock": item.safety_stock,
                "forecast_demand_during_lead_time": round(expected_demand, 1),
                "reorder_point": round(reorder_point, 1),
                "recommended_order_qty": order_qty,
                "status": "REORDER" if order_qty > 0 else "OK",
            }
        )
    return recommendations
