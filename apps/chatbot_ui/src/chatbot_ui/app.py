import sys
from pathlib import Path
import streamlit as st
import requests

try:
    from chatbot_ui.core.config import config
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from chatbot_ui.core.config import config


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

## Lets create a sidebar with a dropdown for the model list and providers
with st.sidebar:
    st.title("Settings")

    #Dropdown for model
    provider = st.selectbox("Provider", ["OpenAI", "Groq", "Google"])
    if provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-5-nano", "gpt-5-mini"])
    elif provider == "Groq":
        model_name = st.selectbox("Model", ["llama-3.3-70b-versatile"])
    else:
        model_name = st.selectbox("Model", ["gemini-2.5-flash"])

    # Save provider and model to session state
    st.session_state.provider = provider
    st.session_state.model_name = model_name


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching products and generating answer..."):
            success, response_data = api_call(
                "post",
                f"{get_api_url()}/rag/",
                json={"query": prompt}
            )
            if success and "answer" in response_data:
                answer = response_data["answer"]
            else:
                answer = response_data.get("answer") or response_data.get("message") or response_data.get("detail") or "Sorry, could not get a response from the RAG service."
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})