from __future__ import annotations

import math
import random
from datetime import date, timedelta

from sqlalchemy import select

from backend.database import SessionLocal, init_db
from backend.models import Inventory, Product, Sale

PRODUCTS = [
    {"sku": "GRC-001", "name": "Whole Milk 1L", "category": "Dairy", "price": 62.0, "cost": 43.0, "base": 42},
    {"sku": "GRC-002", "name": "Brown Bread", "category": "Bakery", "price": 48.0, "cost": 30.0, "base": 35},
    {"sku": "GRC-003", "name": "Basmati Rice 5kg", "category": "Staples", "price": 520.0, "cost": 420.0, "base": 12},
    {"sku": "GRC-004", "name": "Cooking Oil 1L", "category": "Staples", "price": 155.0, "cost": 125.0, "base": 18},
    {"sku": "GRC-005", "name": "Eggs 12 Pack", "category": "Dairy", "price": 96.0, "cost": 72.0, "base": 24},
    {"sku": "GRC-006", "name": "Bananas 1kg", "category": "Produce", "price": 55.0, "cost": 35.0, "base": 30},
    {"sku": "GRC-007", "name": "Apples 1kg", "category": "Produce", "price": 180.0, "cost": 135.0, "base": 16},
    {"sku": "GRC-008", "name": "Coffee 200g", "category": "Beverages", "price": 245.0, "cost": 190.0, "base": 10},
    {"sku": "GRC-009", "name": "Orange Juice 1L", "category": "Beverages", "price": 110.0, "cost": 78.0, "base": 14},
    {"sku": "GRC-010", "name": "Potato Chips", "category": "Snacks", "price": 30.0, "cost": 18.0, "base": 45},
    {"sku": "GRC-011", "name": "Chocolate Bar", "category": "Snacks", "price": 45.0, "cost": 27.0, "base": 28},
    {"sku": "GRC-012", "name": "Dishwashing Liquid", "category": "Household", "price": 125.0, "cost": 88.0, "base": 9},
]


def seed_database() -> None:
    init_db()
    random.seed(42)
    with SessionLocal() as db:
        existing = db.scalar(select(Product.id).limit(1))
        if existing:
            print("Seed data already exists. Skipping to avoid duplicate records.")
            return

        product_rows: list[Product] = []
        for item in PRODUCTS:
            product_rows.append(
                Product(
                    sku=item["sku"],
                    name=item["name"],
                    category=item["category"],
                    base_price=item["price"],
                    cost_price=item["cost"],
                )
            )
        db.add_all(product_rows)
        db.flush()

        today = date.today()
        start = today - timedelta(days=730)
        sales_rows: list[Sale] = []
        recent_units: dict[int, list[int]] = {}

        for product, spec in zip(product_rows, PRODUCTS):
            recent_units[product.id] = []
            for day_offset in range(730):
                current_day = start + timedelta(days=day_offset)
                weekday_factor = 1.15 if current_day.weekday() >= 5 else 1.0
                seasonal_factor = 1.0 + 0.10 * math.sin(2 * math.pi * current_day.timetuple().tm_yday / 365.25)
                trend_factor = 1.0 + 0.00015 * day_offset
                promo = random.random() < 0.09
                discount = round(random.uniform(0.05, 0.20), 3) if promo else 0.0
                demand_mean = spec["base"] * weekday_factor * seasonal_factor * trend_factor * (1 + discount * 0.35)
                units = max(0, int(round(random.gauss(demand_mean, max(1.0, demand_mean * 0.13)))))
                effective_price = round(spec["price"] * (1 - discount), 2)
                sales_rows.append(
                    Sale(
                        product_id=product.id,
                        sale_date=current_day,
                        units_sold=units,
                        unit_price=effective_price,
                        discount_pct=discount,
                    )
                )
                if day_offset >= 700:
                    recent_units[product.id].append(units)

        # Batch insertion keeps memory and transaction overhead reasonable.
        for start_index in range(0, len(sales_rows), 2000):
            db.add_all(sales_rows[start_index : start_index + 2000])
            db.flush()
        db.flush()

        for product, spec in zip(product_rows, PRODUCTS):
            avg_daily = sum(recent_units[product.id]) / max(1, len(recent_units[product.id]))
            lead_days = random.randint(3, 10)
            safety = max(5, int(avg_daily * 2))
            stock = max(0, int(avg_daily * (lead_days + 3)))
            db.add(
                Inventory(
                    product_id=product.id,
                    stock_qty=stock,
                    lead_time_days=lead_days,
                    safety_stock=safety,
                )
            )
        db.commit()
        print(f"Seeded {len(product_rows)} products and {len(sales_rows)} daily sales records.")


if __name__ == "__main__":
    seed_database()
