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
        retry_options=types.HttpRetryOptions(
            attempts=2,
            initial_delay=0.5,
            max_delay=2.0,
        )
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

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["average_rating"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }



def process_context(context, max_chars_per_item=800):
    formatted_context = ""

    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        desc = (chunk[:max_chars_per_item] + "...") if len(chunk) > max_chars_per_item else chunk
        formatted_context += (
            f"- ID: {id}, rating: {rating}, description: {desc}\n"
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
    (
        "gemini-3.8-flash",
        types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low")
        ),
    ),
    ("gemini-3.5-flash-lite", None),
    ("gemini-3.1-flash-lite", None),
]


def generate_answer(prompt: str, model: str | None = None) -> str:
    """Generate an answer with automatic fast fallback when a Gemini model is overloaded."""
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
            if exc.code in (404, 429, 500, 503):
                print(
                    f"--> [3/3] Model '{model_name}' unavailable ({exc.code}), trying fallback...",
                    flush=True,
                )
                continue
            raise

    raise last_error  # type: ignore[misc]


def rag_pipeline(question, top_k=3):
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

    return answer