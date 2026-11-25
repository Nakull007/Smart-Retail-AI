from functools import lru_cache

from transformers import pipeline

from backend.config import settings


@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    # CPU inference keeps the project usable without a CUDA-capable GPU.
    return pipeline(
        task="sentiment-analysis",
        model=settings.SENTIMENT_MODEL,
        device=-1,
    )


def analyze_sentiment(text: str) -> dict:
    result = get_sentiment_pipeline()(text[:4000], truncation=True, max_length=512)[0]
    return {"label": result["label"], "score": float(result["score"])}
