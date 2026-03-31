import os
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Polyglot NER Explorer",
    page_icon="🔍",
    layout="centered",
)

# --- CUSTOM CSS (Loaded from separate file) ---
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --- APP HEADER ---
st.markdown("<h1 class='header-text'>🔍 Polyglot NER Explorer</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; color: #666; margin-top: 0.5rem; margin-bottom: 2rem;'>
Extract Named Entities from <b>Hungarian</b> and <b>German</b> text
using a fine-tuned Transformer model.
</p>
""", unsafe_allow_html=True)

# --- SIDEBAR / SETTINGS ---
with st.sidebar:
    st.title("⚙️ System Configuration")

    # Dynamic API URL (Environment Variable for Docker, fallback to localhost)
    default_backend = os.getenv("BACKEND_URL", "http://localhost:8000")
    API_URL = st.text_input("Backend API URL", value=default_backend)
    st.divider()

    # Check API Health
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            status = health_resp.json()
            if status.get("model_loaded"):
                st.success("✅ API Online & Model Loaded")
            else:
                st.warning("⚠️ API Online (Model Loading...)")
        else:
            st.error(f"❌ API Error ({health_resp.status_code})")
    except Exception:
        st.error("❌ API Offline")
        st.info("💡 Run 'python app.py' to start the backend.")

# --- MAIN INTERFACE ---
st.subheader("Analyze Text")
default_text = "Kovács János az OTP Bank igazgatója Budapesten."
user_text = st.text_area(
    "Input your text here:",
    value="",
    placeholder=default_text,
    height=150
)

# Color Mapping for Entities
COLOR_MAP = {
    "PER": "entity-per",
    "ORG": "entity-org",
    "LOC": "entity-loc",
    "MISC": "entity-misc"
}

def highlight_entities(text: str, entities: List[Dict]):
    """Small utility to show entities in a readable way."""
    if not entities:
        return text

    # Sort entities by start position in reverse to avoid index shifting
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)

    html_text = text
    for ent in sorted_entities:
        start, end = ent['start'], ent['end']
        label = ent['entity_group']
        word = ent['word']
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
        with st.spinner("Model is thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"items": user_text},
                    timeout=10
                )

                if response.status_code == 200:
                    results = response.json().get("results", [])

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
                        # Filter/Clean columns
                        if 'score' in df.columns:
                            df['score'] = df['score'].apply(lambda x: f"{x:.2%}")
                        st.dataframe(
                            df[['word', 'entity_group', 'score']],
                            hide_index=True,
                            use_container_width=True
                        )
                else:
                    st.error(f"API Error: Server returned {response.status_code}")
                    st.json(response.json())

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Is it running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# --- FOOTER ---
footer_html = """
<br><hr><center>
<small>Polyglot NER Project • Built with FastAPI & Streamlit</small>
</center>
"""
st.markdown(footer_html, unsafe_allow_html=True)
