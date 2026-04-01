import os
from typing import Dict, List

import pandas as pd
import streamlit as st

from src.config import NERConfig
from src.predictor import NERPredictor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Polyglot NER Explorer (Cloud)",
    page_icon="🔍",
    layout="centered",
)


# --- CUSTOM CSS (Loaded from separate file) ---
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("style.css")


# --- MODEL INITIALIZATION (@st.cache_resource) ---
@st.cache_resource
def get_predictor():
    """Initializes and caches the NER Predictor for single-process cloud deployment."""
    # Ensure tokens are set if provided via Streamlit Secrets
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    config = NERConfig()
    return NERPredictor(config)


# --- APP HEADER ---
st.markdown("<h1 class='header-text'>🔍 Polyglot NER Explorer</h1>", unsafe_allow_html=True)
st.markdown(
    """
<p style='text-align: center; color: #666; margin-top: 0.5rem; margin-bottom: 2rem;'>
Extract Named Entities from <b>Hungarian</b> and <b>German</b> text
using an integrated Transformer model.
</p>
""",
    unsafe_allow_html=True,
)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Cloud Configuration")
    st.info(
        "This version of the app uses an **Integrated Inference Engine** "
        "optimized for Streamlit Community Cloud."
    )
    st.divider()
    st.markdown("### 🏷️ Supported Entities")
    st.markdown("- **PER**: Persons\n- **ORG**: Organizations\n- **LOC**: Locations\n- **MISC**: Miscellaneous")

# --- MAIN INTERFACE ---
st.subheader("Analyze Text")
default_text = "Morgen treffen sich Vertreter der Deutschen Bahn am Potsdamer Platz in Berlin."
user_text = st.text_area("Input your text here:", placeholder=default_text, height=150)

# Color Mapping for Entities
COLOR_MAP = {"PER": "entity-per", "ORG": "entity-org", "LOC": "entity-loc", "MISC": "entity-misc"}


def highlight_entities(text: str, entities: List[Dict]):
    """Small utility to show entities in a readable way."""
    if not entities:
        return text

    # Sort entities by start position in reverse to avoid index shifting
    sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)

    html_text = text
    for ent in sorted_entities:
        start, end = ent["start"], ent["end"]
        label = ent["entity_group"]
        word = ent["word"]
        css_class = COLOR_MAP.get(label, "entity-misc")

        # Replace the range with highlighted HTML
        tag = f"<span class='entity-tag {css_class}'>{word} <small>[{label}]</small></span>"
        html_text = html_text[:start] + tag + html_text[end:]

    return html_text


cols = st.columns([1, 4])
with cols[0]:
    analyze_btn = st.button("Extract ✨", type="primary", use_container_width=True)

if analyze_btn:
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Integrated model is thinking..."):
            try:
                # Direct Integrated Inference
                predictor = get_predictor()
                results = predictor.predict(user_text)

                if not results:
                    st.info("No entities detected in the text.")
                else:
                    st.subheader("Analysis Results")

                    # Show Highlighted version
                    st.markdown("#### Annotated Text")
                    annotated = highlight_entities(user_text, results)
                    st.markdown(f"<div>{annotated}</div>", unsafe_allow_html=True)

                    st.divider()

                    # Show Table version
                    st.markdown("#### Entity Details")
                    df = pd.DataFrame(results)
                    if "score" in df.columns:
                        df["score"] = df["score"].apply(lambda x: f"{x:.2%}")
                    st.dataframe(
                        df[["word", "entity_group", "score"]],
                        hide_index=True,
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"An error occurred during integrated inference: {str(e)}")

# --- FOOTER ---
footer_html = """
<br><hr><center>
<small>Polyglot NER Project • Integrated Cloud Deployment</small>
</center>
"""
st.markdown(footer_html, unsafe_allow_html=True)
