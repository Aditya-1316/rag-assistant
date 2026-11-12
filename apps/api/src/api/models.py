from pydantic import BaseModel, Field


class ProductItem(BaseModel):
    parent_asin: str
    title: str
    price: float | None = None
    average_rating: float | None = None
    rating_number: int | None = None
    image: str | None = None
    score: float | None = None


class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in RAG pipeline")


class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The answer to the query")
    products: list[ProductItem] = Field(
        default_factory=list,
        description="Retrieved products used to answer the query",
    )