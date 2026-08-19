"""SmartDoc UI: ask questions about your PDFs, see where every answer came from.

This file only displays things. All retrieval and answering happens behind the
FastAPI service, which it calls over HTTP -- so the same endpoints could serve a
mobile app or a Slack bot without touching this code.
"""

import requests
import streamlit as st

from app import config

TIMEOUT = 120

st.set_page_config(page_title="SmartDoc", page_icon="📄", layout="centered")


def api_get(path: str) -> dict:
    """GET from the API, returning {} if it is not reachable."""
    try:
        response = requests.get(config.API_URL + path, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {}


def api_post(path: str, **kwargs) -> dict:
    """POST to the API, turning any transport failure into a readable error."""
    try:
        response = requests.post(config.API_URL + path, timeout=TIMEOUT, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        return {"_transport_error": str(error)}


# --- Sidebar: what is indexed, and how to add more -----------------------

health = api_get("/health")

with st.sidebar:
    st.header("Document library")

    if not health:
        st.error(
            "Cannot reach the API at {}.\n\nOpen another terminal in the "
            "project folder and run:\n\n`./run_api.sh`\n\nThen reload this "
            "page.".format(config.API_URL)
        )
    else:
        st.metric("Searchable chunks", health.get("chunks", 0))
        documents = health.get("documents") or []
        if documents:
            st.write("**Indexed documents**")
            for name in documents:
                st.write("- {}".format(name))
        else:
            st.info("No documents yet. Upload a PDF below.")

        st.caption("Embedding model: {}".format(health.get("embedding_model")))

        if not health.get("chat_key_configured"):
            st.warning(
                "No API key configured, so answers cannot be written. "
                "Search and citations still work."
            )

    st.divider()
    st.subheader("Add a document")
    uploaded = st.file_uploader("PDF only", type=["pdf"])
    if uploaded is not None and st.button("Add to library", width="stretch"):
        with st.spinner("Reading and indexing {}...".format(uploaded.name)):
            result = api_post(
                "/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            )
        if result.get("_transport_error"):
            st.error("Upload failed: {}".format(result["_transport_error"]))
        elif not result.get("ok"):
            st.error(result.get("error", "Upload failed."))
        else:
            st.success(
                "Added {} ({} chunks).".format(
                    result["document"], result["chunks_added"]
                )
            )
            if result.get("unreadable_pages"):
                st.warning(
                    "{} page(s) are images or scans and could not be indexed. "
                    "Open the original document to read those.".format(
                        result["unreadable_pages"]
                    )
                )
            st.rerun()


# --- Main panel ----------------------------------------------------------

st.title("📄 SmartDoc")
st.caption(
    "Ask a question about your documents. Every answer shows exactly where it "
    "came from."
)

question = st.text_input(
    "Your question",
    placeholder="e.g. What are the rules for family employees?",
)
asked = st.button("Ask", type="primary")

if asked and question.strip():
    with st.spinner("Searching your documents..."):
        result = api_post("/ask", json={"question": question})

    if result.get("_transport_error"):
        st.error("Could not reach the API: {}".format(result["_transport_error"]))
    else:
        answer = (result.get("answer") or "").strip()
        sources = result.get("sources") or []
        notice = result.get("notice") or ""

        # The answer itself
        if answer.startswith("I don't know"):
            st.info("🤔 " + answer)
        elif answer:
            st.markdown("### Answer")
            st.markdown(answer)

        # If the model could not be called, say so plainly -- and still show
        # what was found, because that is the genuinely useful half.
        if notice:
            st.warning("⚠️ " + notice)
            if sources:
                st.markdown(
                    "The relevant sections were still found. They are below, "
                    "so you can read them yourself."
                )

        # Citations: always shown, never optional.
        if sources:
            st.markdown("### Sources")
            for index, source in enumerate(sources, start=1):
                label = "[{}] {}".format(index, source["citation"])
                if source["content_type"] == "table":
                    label += "  ·  table"
                elif source["content_type"] == "image_only":
                    label += "  ·  unreadable page"
                if source.get("match_type") == "exact":
                    label += "  ·  exact match"
                with st.expander(label):
                    if source.get("match_type") == "exact":
                        st.caption("found by exact identifier match, not by similarity")
                    else:
                        st.caption(
                            "relevance distance {:.3f} (lower is closer)".format(
                                source["distance"]
                            )
                        )
                    st.text(source["text"][:4000])
        elif not notice and answer.startswith("I don't know"):
            st.caption("Nothing in the library was close enough to this question.")

elif asked:
    st.warning("Please type a question first.")
