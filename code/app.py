# =============================================================================
# app.py — Streamlit Web UI
# =============================================================================
# Provides a browser-based interface for the Vehicle Diagnostics Copilot.
#
# Two diagnosis modes:
#   1. Diagnose          — quick RAG-based diagnosis (chain.py)
#   2. Full Cost Diagnosis — 3-agent pipeline with validation + cost estimate (agents.py)
#
# Run with: streamlit run app.py
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_chain():
    from chain import document_chain
    return document_chain

@st.cache_resource(show_spinner=False)
def load_agents():
    from agents import run_agents
    return run_agents

# =============================================================================
# SECTION 1: Page Layout & Inputs
# =============================================================================

st.title("Intelligent Vehicle Diagnostics Copilot")

# Supported vehicles — matches the manuals in the FAISS index
VEHICLE_MODELS = {
    "": [],
    "Honda": ["Civic"],
    "Toyota": ["Camry", "Corolla"],
    "Ford": ["F-150"],
}

VEHICLE_YEARS = {
    ("Honda", "Civic"): ["2011"],
    ("Toyota", "Camry"): ["2018"],
    ("Toyota", "Corolla"): ["2019"],
    ("Ford", "F-150"): ["2018"],
}

# Vehicle details — used to provide context to the LLM
col1, col2, col3 = st.columns(3)
with col1:
    make = st.selectbox("Vehicle Make", options=list(VEHICLE_MODELS.keys()))
with col2:
    model_options = [""] + VEHICLE_MODELS.get(make, [])
    model = st.selectbox("Vehicle Model", options=model_options)
with col3:
    year_options = [""] + VEHICLE_YEARS.get((make, model), [])
    year = st.selectbox("Year", options=year_options)

# Optional OBD code and symptom description
obd_code = st.text_input("Enter OBD Code (optional)", placeholder="e.g. P0305")
symptoms = st.text_area("Describe your symptoms", placeholder="e.g. engine misfiring, rough idle")

# =============================================================================
# SECTION 2: Action Buttons
# =============================================================================

col1, col2 = st.columns(2)
with col1:
    but1 = st.button("Diagnose")
with col2:
    but2 = st.button("Full Cost Diagnosis")

# =============================================================================
# SECTION 3: Query Builder
# =============================================================================
# Combine vehicle context + OBD code + symptoms into a single query string.
# Example: "Vehicle: 2018 Toyota Camry. OBD Code: P0300. engine vibration"

def build_query():
    vehicle_context = f"Vehicle: {year} {make} {model}. " if (make or model or year) else ""
    obd_context = f"OBD Code: {obd_code}. " if obd_code else ""
    return f"{vehicle_context}{obd_context}{symptoms}"

def clean(text):
    """Remove backticks from LLM output to prevent green code formatting."""
    import re
    text = re.sub(r"```[\s\S]*?```", "", text)  # delete triple-backtick blocks entirely
    text = text.replace("`", "")                 # delete any remaining single backticks
    return text

# =============================================================================
# SECTION 4: Diagnosis Results
# =============================================================================

if but1:
    query = build_query()
    try:
        with st.spinner("Analyzing..."):
            result = load_chain().invoke({"question": query})
        st.markdown(clean(result))
    except Exception as e:
        st.error(f"Out of Scope")

elif but2:
    query = build_query()
    try:
        with st.spinner("This may take a moment..."):
            result = load_agents()(query)
        st.subheader("Diagnosis")
        st.markdown(clean(result["diagnosis"]))
        st.subheader("Validation")
        st.markdown(clean(result["validation"]))
        st.subheader("Cost Estimate")
        st.markdown(clean(result["cost_estimate"]))
    except Exception as e:
        st.error(f"Out of Scope")
