from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[dict] = Field(default_factory=list)


class DocumentSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class ReviewRequest(BaseModel):
    review_text: str = Field(min_length=3, max_length=5000)
    product_id: int | None = None
