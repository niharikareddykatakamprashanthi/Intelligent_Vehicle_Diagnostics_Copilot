# =============================================================================
# agents.py — LangGraph Multi-Agent Diagnostic Pipeline
# =============================================================================
# Responsible for running a 3-step agent pipeline using LangGraph:
#   Agent 1 — Diagnose    : uses RAG chain to diagnose the issue
#   Agent 2 — Validate    : senior expert reviews and critiques the diagnosis
#   Agent 3 — Cost Estimate: estimates parts, labor, time, and priority
#
# All 3 agents share a common state (DiagnosticState) that gets passed
# from one agent to the next through the graph.
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict
from chain import document_chain
import os
from dotenv import load_dotenv

# ── Environment Setup ─────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

# =============================================================================
# SECTION 1: Shared State
# =============================================================================
# DiagnosticState is a dictionary passed between all 3 agents.
# Each agent reads from it and writes its result back into it.

class DiagnosticState(TypedDict):
    question: str       # the user's original query
    diagnosis: str      # filled by Agent 1
    validation: str     # filled by Agent 2
    cost_estimate: str  # filled by Agent 3

# =============================================================================
# SECTION 2: Agent 1 — Diagnose
# =============================================================================
# Calls the RAG pipeline (document_chain) to generate a structured diagnosis
# using retrieved context from vehicle manuals and OBD codes.

def agent_diagnose(state: DiagnosticState) -> DiagnosticState:
    result = document_chain.invoke({"question": state["question"]})
    state["diagnosis"] = result
    return state

# =============================================================================
# SECTION 3: Agent 2 — Validate
# =============================================================================
# Acts as a senior expert reviewing Agent 1's diagnosis.
# Checks for accuracy, missing causes, or incorrect checks.

validate_prompt = ChatPromptTemplate.from_template("""
You are a senior vehicle diagnostics expert reviewing a junior technician's diagnosis.

Original Question: {question}

Diagnosis Given:
{diagnosis}

Reply in this format:

Validation Status: Approved / Needs Revision

Issues Found:
- ...

Corrections / Additions:
- ...
""")

validate_chain = validate_prompt | llm | StrOutputParser()

def agent_validate(state: DiagnosticState) -> DiagnosticState:
    result = validate_chain.invoke({
        "question": state["question"],
        "diagnosis": state["diagnosis"]
    })
    state["validation"] = result
    return state

# =============================================================================
# SECTION 4: Agent 3 — Cost Estimate
# =============================================================================
# Takes the diagnosis and estimates repair costs, labor, time, and priority.

cost_prompt = ChatPromptTemplate.from_template("""
You are an automotive repair cost estimator.

Based on the diagnosis below, estimate the repair cost and time.

IMPORTANT FORMATTING RULES — follow these strictly:
- NEVER use backticks (`) anywhere in your response
- NEVER use code formatting for any numbers or dollar amounts
- Write all numbers as plain text: write $50 not `$50`, write 200 not `200`
- Always use the $ sign directly before dollar amounts

Diagnosis:
{diagnosis}

Provide estimates in this format:

Estimated Cost:

Parts Breakdown:
- [Part Name]: $[min] - $[max]
- [Part Name]: $[min] - $[max]

Labor: $X (X hours at $100/hour)
Total: $[min] - $[max]

Estimated Time:
- ...

Difficulty Level:
- DIY Possible / Needs Professional

Priority:
- Fix Immediately / Can Wait / Monitor
""")

cost_chain = cost_prompt | llm | StrOutputParser()

def agent_cost_estimate(state: DiagnosticState) -> DiagnosticState:
    result = cost_chain.invoke({"diagnosis": state["diagnosis"]})
    state["cost_estimate"] = result
    return state

# =============================================================================
# SECTION 5: LangGraph Pipeline
# =============================================================================
# Connects all 3 agents into a sequential graph:
#   diagnose → validate → cost_estimate → END

graph = StateGraph(DiagnosticState)

graph.add_node("diagnose", agent_diagnose)
graph.add_node("validate", agent_validate)
graph.add_node("cost_estimate", agent_cost_estimate)

graph.set_entry_point("diagnose")
graph.add_edge("diagnose", "validate")
graph.add_edge("validate", "cost_estimate")
graph.add_edge("cost_estimate", END)

agent_pipeline = graph.compile()

# =============================================================================
# SECTION 6: Public Interface
# =============================================================================
# run_agents() is the single entry point called by app.py and api.py.

def run_agents(question: str) -> dict:
    """Run the full 3-agent pipeline and return all results."""
    result = agent_pipeline.invoke({"question": question})
    return {
        "diagnosis": result["diagnosis"],
        "validation": result["validation"],
        "cost_estimate": result["cost_estimate"]
    }
