from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.config import settings
from backend.database import init_db
from backend.routers.api import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="Retail demand forecasting, inventory optimization, RAG and a Gemini-powered assistant.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/")
def root():
    return {"message": "Smart Retail AI API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
