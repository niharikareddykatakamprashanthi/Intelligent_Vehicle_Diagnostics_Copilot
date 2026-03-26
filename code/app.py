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
from chain import document_chain
from agents import run_agents

# =============================================================================
# SECTION 1: Page Layout & Inputs
# =============================================================================

st.title("Intelligent Vehicle Diagnostics Copilot")

# Vehicle details — used to provide context to the LLM
col1, col2, col3 = st.columns(3)
with col1:
    make = st.text_input("Vehicle Make", placeholder="e.g. Toyota")
with col2:
    model = st.text_input("Vehicle Model", placeholder="e.g. Camry")
with col3:
    year = st.text_input("Year", placeholder="e.g. 2018")

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

# =============================================================================
# SECTION 4: Diagnosis Results
# =============================================================================

if but1:
    query = build_query()
    with st.spinner("Analyzing..."):
        result = document_chain.invoke({"question": query})
    st.markdown(result)

elif but2:
    query = build_query()
    with st.spinner("This may take a moment..."):
        result = run_agents(query)
    st.subheader("Diagnosis")
    st.markdown(result["diagnosis"])
    st.subheader("Validation")
    st.markdown(result["validation"])
    st.subheader("Cost Estimate")
    st.markdown(result["cost_estimate"])
