import sys
from pathlib import Path
import streamlit as st
import requests

try:
    from chatbot_ui.core.config import config
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from chatbot_ui.core.config import config


st.set_page_config(
    page_title="RAG Shopping Assistant",
    page_icon="🛍️",
    layout="centered",
)


def get_api_url() -> str:
    """Resolve the API base URL from Streamlit secrets or environment config."""
    try:
        if "API_URL" in st.secrets:
            return str(st.secrets["API_URL"]).strip().rstrip("/")
    except Exception:
        pass
    return config.API_URL.strip().rstrip("/")


def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        kwargs.setdefault("timeout", 90)
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": f"Server error ({response.status_code}): {response.text or 'No details'}"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError as e:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": f"Connection error reaching {url}: {e}"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}

SAMPLE_QUESTIONS = [
    "Suggest some cool earphones",
    "What laptops are available in stock?",
    "Recommend wireless headphones with good ratings",
]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about the products in stock."}
    ]

with st.sidebar:
    st.title("🛍️ Shopping Assistant")
    st.caption("Ask questions about products in our Amazon catalog using semantic search + Gemini.")

    st.divider()
    st.subheader("Try a sample question")
    for sample in SAMPLE_QUESTIONS:
        if st.button(sample, use_container_width=True):
            st.session_state["pending_prompt"] = sample

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Ask me anything about the products in stock."}
        ]
        st.rerun()

    st.caption("Powered by FastAPI · Qdrant Cloud · Google Gemini")


def render_products(products: list[dict]) -> None:
    """Display retrieved Amazon product cards in a horizontal grid."""
    if not products:
        return

    st.markdown("**Matched Products:**")
    cols = st.columns(len(products))
    for col, product in zip(cols, products):
        with col:
            with st.container(border=True):
                if product.get("image"):
                    st.image(product["image"], width=160)
                st.markdown(f"**{product.get('title', product.get('parent_asin', 'Product'))}**")

                price = product.get("price")
                price_str = f"${price:,.2f}" if price is not None else "Price N/A"
                rating = product.get("average_rating")
                reviews = product.get("rating_number")
                rating_str = f"⭐ {rating}" if rating is not None else ""
                if reviews is not None and rating_str:
                    rating_str += f" ({reviews})"

                meta = " · ".join(part for part in (price_str, rating_str) if part)
                if meta:
                    st.caption(meta)

                asin = product.get("parent_asin")
                if asin:
                    st.markdown(f"[View on Amazon](https://www.amazon.com/dp/{asin})")


st.title("🛍️ AI Shopping Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("products"):
            render_products(message["products"])

user_input = st.chat_input("Ask about a product (e.g., laptops, earphones)...")
prompt = st.session_state.pop("pending_prompt", None) or user_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    products: list[dict] = []
    with st.chat_message("assistant"):
        with st.spinner("Searching products and generating answer..."):
            success, response_data = api_call(
                "post",
                f"{get_api_url()}/rag/",
                json={"query": prompt}
            )
            if success and "answer" in response_data:
                answer = response_data["answer"]
                products = response_data.get("products", [])
            else:
                answer = (
                    response_data.get("answer")
                    or response_data.get("message")
                    or response_data.get("detail")
                    or "Sorry, could not get a response from the RAG service."
                )
            st.write(answer)
            render_products(products)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "products": products}
    )