import os
import time
import logging
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from qdrant_client import QdrantClient
from api.core.config import config

logger = logging.getLogger(__name__)

gemini_client = genai.Client(
    api_key=config.GOOGLE_API_KEY,
    http_options=types.HttpOptions(
        timeout=15000,
        retry_options=types.HttpRetryOptions(
            attempts=1,
        ),
    ),
)

# Reuse Qdrant client connection globally
qdrant_url = config.QDRANT_URL or ("http://qdrant:6333" if os.path.exists("/.dockerenv") else "http://localhost:6333")
qdrant_kwargs = {"url": qdrant_url}
if qdrant_url.startswith("https://"):
    if config.QDRANT_API_KEY:
        qdrant_kwargs["api_key"] = config.QDRANT_API_KEY
    qdrant_kwargs["port"] = 443
qdrant_client = QdrantClient(**qdrant_kwargs)


def get_embedding(text: str, model: str = "gemini-embedding-001") -> list[float]:
    response = gemini_client.models.embed_content(
        model=model,
        contents=text,
    )
    return response.embeddings[0].values


def retrieve_data(query, qdrant_client, k=3):
    print(f"--> [1/3] Generating embedding for query: '{query}'...", flush=True)
    t0 = time.time()
    query_embedding = get_embedding(query)
    t_emb = time.time() - t0
    print(f"--> [1/3] Embedding completed in {t_emb:.2f}s", flush=True)

    print(f"--> [2/3] Querying Qdrant at {qdrant_url} (top_k={k})...", flush=True)
    t1 = time.time()
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-00",
        query=query_embedding,
        limit=k,
    )
    t_qdrant = time.time() - t1
    print(f"--> [2/3] Qdrant search completed in {t_qdrant:.2f}s (found {len(results.points)} items)", flush=True)

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []
    products = []

    for result in results.points:
        payload = result.payload or {}
        description = payload.get("description", "")
        title = description.split("\n", 1)[0].strip()
        if len(title) > 90:
            title = title[:87].rsplit(" ", 1)[0] + "..."

        parent_asin = payload.get("parent_asin", "")
        rating = payload.get("average_rating")
        price = payload.get("price")

        retrieved_context_ids.append(parent_asin)
        retrieved_context.append(description)
        retrieved_context_ratings.append(rating)
        similarity_scores.append(result.score)
        products.append(
            {
                "parent_asin": parent_asin,
                "title": title or parent_asin,
                "price": price,
                "average_rating": rating,
                "rating_number": payload.get("rating_number"),
                "image": payload.get("image"),
                "score": result.score,
            }
        )

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
        "products": products,
    }



def process_context(context, max_chars_per_item=800):
    formatted_context = ""

    for item, chunk in zip(context.get("products", []), context["retrieved_context"]):
        desc = (chunk[:max_chars_per_item] + "...") if len(chunk) > max_chars_per_item else chunk
        price_str = f"${item['price']}" if item.get("price") is not None else "N/A"
        formatted_context += (
            f"- ID: {item['parent_asin']}, price: {price_str}, "
            f"rating: {item.get('average_rating')}, description: {desc}\n"
        )

    return formatted_context



def build_prompt(preprocessed_context, question):
    
    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructtions:
- You need to answer the question based on the provided context only.
- Never use word context and refer to it as the available products.

Context:
{preprocessed_context}

Question:
{question}
"""
    
    return prompt


MODEL_CANDIDATES = [
    ("gemini-3.5-flash-lite", None),
    ("gemini-3.1-flash-lite", None),
    (
        "gemini-3.8-flash",
        types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low")
        ),
    ),
]


def generate_answer(prompt: str, model: str | None = None) -> str:
    """Generate an answer with fast 15s timeout and automatic fallback across Gemini models."""
    candidates = MODEL_CANDIDATES
    if model and all(name != model for name, _ in MODEL_CANDIDATES):
        candidates = [(model, None), *MODEL_CANDIDATES]

    last_error: Exception | None = None
    for model_name, gen_config in candidates:
        try:
            print(f"--> [3/3] Trying Gemini model '{model_name}'...", flush=True)
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=gen_config,
            )
            return response.text
        except genai_errors.APIError as exc:
            last_error = exc
            if exc.code in (400, 401, 403):
                raise
            print(
                f"--> [3/3] Model '{model_name}' failed ({exc.code}), trying fallback...",
                flush=True,
            )
        except Exception as exc:
            last_error = exc
            print(
                f"--> [3/3] Model '{model_name}' timed out/errored ({type(exc).__name__}), trying fallback...",
                flush=True,
            )

    raise last_error  # type: ignore[misc]


def rag_pipeline_with_products(question: str, top_k: int = 3) -> tuple[str, list[dict]]:
    """Run the RAG pipeline and return both the generated answer and retrieved product metadata."""
    t_start = time.time()
    print(f"\n==================== NEW RAG QUERY ====================", flush=True)
    print(f"Query: {question}", flush=True)

    retrieved_context = retrieve_data(question, qdrant_client, top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)

    t_gen_start = time.time()
    answer = generate_answer(prompt)
    t_gen = time.time() - t_gen_start
    print(f"--> [3/3] Gemini generated answer in {t_gen:.2f}s", flush=True)

    t_total = time.time() - t_start
    print(f"==== TOTAL RAG TIME: {t_total:.2f}s ====\n", flush=True)

    return answer, retrieved_context.get("products", [])


def rag_pipeline(question: str, top_k: int = 3) -> str:
    answer, _ = rag_pipeline_with_products(question, top_k=top_k)
    return answer