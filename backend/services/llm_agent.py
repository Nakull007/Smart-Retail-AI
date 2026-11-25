from __future__ import annotations

from datetime import date, timedelta

from google import genai
from google.genai import types
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models import Inventory, Product, Sale
from backend.services.forecasting import forecast_product
from backend.services.rag import retrieve_chunks


TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_inventory",
        description="Get current stock quantities and lead times. Use when the user asks about available stock or low inventory.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Optional product name to filter by."}
            },
        },
    ),
    types.FunctionDeclaration(
        name="forecast_demand",
        description="Get machine-learning demand predictions for a product over the next few days. Use for future demand questions.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Product name to forecast."},
                "days": {"type": "integer", "minimum": 1, "maximum": 30},
            },
            "required": ["product_name"],
        },
    ),
    types.FunctionDeclaration(
        name="get_top_selling_products",
        description="Get products with the highest total units sold in a recent period.",
        parameters_json_schema={
            "type": "object",
            "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 90}},
        },
    ),
    types.FunctionDeclaration(
        name="search_policy_docs",
        description="Search uploaded supplier, storage, product or inventory policy documents using semantic retrieval. Use for questions about written policies or manuals.",
        parameters_json_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
]


def _run_tool(name: str, args: dict, db: Session) -> dict:
    if name == "get_inventory":
        query = select(Inventory).join(Product).order_by(Product.name)
        product_name = args.get("product_name")
        if product_name:
            query = query.where(Product.name.ilike(f"%{product_name[:100]}%"))
        rows = db.scalars(query).all()
        return {
            "items": [
                {
                    "product": row.product.name,
                    "stock_qty": row.stock_qty,
                    "lead_time_days": row.lead_time_days,
                    "safety_stock": row.safety_stock,
                }
                for row in rows
            ]
        }

    if name == "forecast_demand":
        product_name = str(args.get("product_name", ""))[:160]
        product = db.scalar(select(Product).where(Product.name.ilike(f"%{product_name}%")))
        if not product:
            return {"error": f"No product matched '{product_name}'."}
        days = max(1, min(int(args.get("days", 7)), 30))
        return {"product": product.name, "forecast": forecast_product(db, product, days=days)}

    if name == "get_top_selling_products":
        days = max(1, min(int(args.get("days", 30)), 90))
        start_date = date.today() - timedelta(days=days)
        rows = db.execute(
            select(Product.name, func.sum(Sale.units_sold).label("units"))
            .join(Sale, Sale.product_id == Product.id)
            .where(Sale.sale_date >= start_date)
            .group_by(Product.id, Product.name)
            .order_by(func.sum(Sale.units_sold).desc())
            .limit(5)
        ).all()
        return {"period_days": days, "products": [{"product": row.name, "units_sold": int(row.units or 0)} for row in rows]}

    if name == "search_policy_docs":
        query = str(args.get("query", ""))[:1000]
        chunks = retrieve_chunks(db, query, top_k=4)
        return {"retrieved_chunks": chunks, "message": "No policy documents have been uploaded yet." if not chunks else "Use only these retrieved passages for document-specific claims."}

    # Never execute arbitrary function names returned by the model.
    return {"error": "Tool is not allow-listed."}


def ask_retail_agent(question: str, db: Session) -> tuple[str, list[dict]]:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file and restart the backend.")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    tool = types.Tool(function_declarations=TOOL_DECLARATIONS)
    system_instructions = (
        "You are SmartRetail Copilot, an assistant for a retail store manager. "
        "Use tools for live inventory, sales and forecast facts. Use search_policy_docs for questions about uploaded policies or manuals. "
        "Never invent stock counts, sales totals, forecast values or policy statements. "
        "When a tool returns no matching records, say so. Keep responses clear and actionable. "
        "When reporting a forecast, say it is a model estimate, not a guarantee."
    )
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=f"{system_instructions}\n\nUser request: {question}")],
    )
    first_response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=[user_content],
        config=types.GenerateContentConfig(tools=[tool], temperature=0.2),
    )

    function_calls = first_response.function_calls or []
    if not function_calls:
        return (first_response.text or "I couldn't generate a response."), []

    # Keep one model turn bounded; no tool can perform arbitrary writes.
    conversation = [user_content]
    if first_response.candidates:
        conversation.append(first_response.candidates[0].content)

    tool_results: list[dict] = []
    for function_call in function_calls[:4]:
        function_name = function_call.name
        arguments = dict(function_call.args or {})
        result = _run_tool(function_name, arguments, db)
        tool_results.append({"tool": function_name, "result": result})
        function_response_part = types.Part.from_function_response(
            name=function_name,
            response={"result": result},
        )
        conversation.append(types.Content(role="tool", parts=[function_response_part]))

    final_response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=conversation,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return (final_response.text or "I retrieved the requested data, but couldn't format a final answer."), tool_results
