# =============================================================================
# api.py — FastAPI REST Backend
# =============================================================================
# Exposes the diagnostics engine as a REST API with two endpoints:
#
#   GET  /              — health check
#   POST /diagnose      — quick RAG-based diagnosis
#   POST /diagnose-full — full 3-agent pipeline (diagnosis + validation + cost)
#
# Run with: uvicorn api:app --reload
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
from chain import document_chain
from agents import run_agents

# =============================================================================
# SECTION 1: App Initialization
# =============================================================================

app = FastAPI(title="Intelligent Vehicle Diagnostics Copilot")

# =============================================================================
# SECTION 2: Request / Response Models
# =============================================================================
# Pydantic models define the shape of data coming in and going out of the API.

class DiagnosticRequest(BaseModel):
    symptoms: str           # required — describe the vehicle issue
    obd_code: str = None    # optional — e.g. "P0300"
    make: str = None        # optional — e.g. "Toyota"
    model: str = None       # optional — e.g. "Camry"
    year: str = None        # optional — e.g. "2018"

class DiagnosticResponse(BaseModel):
    question: str   # the constructed query sent to the LLM
    answer: str     # structured diagnostic answer

class FullDiagnosticResponse(BaseModel):
    question: str       # the constructed query sent to the LLM
    diagnosis: str      # Agent 1 output
    validation: str     # Agent 2 output
    cost_estimate: str  # Agent 3 output

# =============================================================================
# SECTION 3: Query Builder
# =============================================================================
# Combines vehicle context + OBD code + symptoms into a single query string.

def build_query(request: DiagnosticRequest) -> str:
    vehicle_context = f"Vehicle: {request.year} {request.make} {request.model}. " \
                      if (request.make or request.model or request.year) else ""
    obd_context = f"OBD Code: {request.obd_code}. " if request.obd_code else ""
    return f"{vehicle_context}{obd_context}{request.symptoms}"

# =============================================================================
# SECTION 4: API Endpoints
# =============================================================================

@app.get("/")
def root():
    """Health check — confirms the API is running."""
    return {"status": "running", "service": "Vehicle Diagnostics Copilot"}


@app.post("/diagnose", response_model=DiagnosticResponse)
def diagnose(request: DiagnosticRequest):
    """
    Quick diagnosis using the RAG pipeline.
    Returns a structured answer with causes, checks, OBD codes, and severity.
    """
    query = build_query(request)
    result = document_chain.invoke({"question": query})
    return DiagnosticResponse(question=query, answer=result)


@app.post("/diagnose-full", response_model=FullDiagnosticResponse)
def diagnose_full(request: DiagnosticRequest):
    """
    Full diagnosis using the 3-agent pipeline.
    Returns diagnosis + expert validation + repair cost estimate.
    """
    query = build_query(request)
    result = run_agents(query)
    return FullDiagnosticResponse(question=query, **result)
