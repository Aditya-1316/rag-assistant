from fastapi import APIRouter, Request
from api.models import RAGRequest, RAGResponse
from agents.retrieval_generation import rag_pipeline_with_products
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def chat(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:

    answer, products = rag_pipeline_with_products(payload.query)

    return RAGResponse(
        request_id=request.state.request_id,
        answer=answer,
        products=products,
    )

api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])