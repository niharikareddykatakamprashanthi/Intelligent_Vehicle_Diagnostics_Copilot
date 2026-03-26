# =============================================================================
# chain.py — Core RAG Pipeline
# =============================================================================
# Responsible for:
#   1. Loading vehicle manuals (PDF) and OBD codes (JSON)
#   2. Building and persisting FAISS vector indexes
#   3. Hybrid retrieval: BM25 (keyword) + FAISS (semantic)
#   4. HyDE (Hypothetical Document Embeddings) for better retrieval
#   5. LLM-based diagnostics with structured output
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.documents import Document
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# ── Environment Setup ─────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Base directory = project root (one level up from code/)
# Using __file__ ensures paths work regardless of where the app is launched from
BASE_DIR = Path(__file__).parent.parent

# =============================================================================
# SECTION 1: Document Loading
# =============================================================================
# Load PDFs from repair and owner manual folders, then tag each chunk with its
# source so the LLM knows where the information came from.

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Repair manuals (Honda Civic, Toyota Camry, Corolla, Ford F150)
repair_docs = PyPDFDirectoryLoader(str(BASE_DIR / 'data/manuals/repair_manual')).load()
repair_chunks = text_splitter.split_documents(repair_docs)
for chunk in repair_chunks:
    chunk.metadata["source"] = "repair_manual"

# Owner manuals (Toyota Camry)
owner_docs = PyPDFDirectoryLoader(str(BASE_DIR / 'data/manuals/owner_manual')).load()
owner_chunks = text_splitter.split_documents(owner_docs)
for chunk in owner_chunks:
    chunk.metadata["source"] = "owner_manual"

# Combined manual chunks (no OBD codes — kept separate for better retrieval)
manual_chunks = owner_chunks + repair_chunks

# =============================================================================
# SECTION 2: OBD Code Documents
# =============================================================================
# Load OBD codes from JSON and convert each code into a Document object
# so it can be embedded and retrieved just like manual text.

with open(str(BASE_DIR / 'data/formatted_obd.json')) as f:
    obd_data = json.load(f)

obd_docs = [
    Document(
        page_content=f"OBD Code {item['code']}: {item['description']}",
        metadata={"source": "obd_codes", "system": item["system"]}
    )
    for item in obd_data
]

# =============================================================================
# SECTION 3: Vector Store (FAISS Indexes)
# =============================================================================
# Two separate FAISS indexes:
#   - faiss_index     : manuals only  → retrieves repair/diagnostic procedures
#   - faiss_obd_index : OBD codes only → retrieves semantically related error codes
#
# Indexes are saved locally on first build and loaded on subsequent runs
# to avoid re-embedding on every startup.

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Manual FAISS index
FAISS_INDEX_PATH = str(BASE_DIR / "faiss_index")
if Path(FAISS_INDEX_PATH).exists():
    vectors = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vectors = FAISS.from_documents(documents=manual_chunks, embedding=embeddings)
    vectors.save_local(FAISS_INDEX_PATH)

# OBD FAISS index
OBD_INDEX_PATH = str(BASE_DIR / "faiss_obd_index")
if Path(OBD_INDEX_PATH).exists():
    obd_vectors = FAISS.load_local(OBD_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    obd_vectors = FAISS.from_documents(documents=obd_docs, embedding=embeddings)
    obd_vectors.save_local(OBD_INDEX_PATH)

# =============================================================================
# SECTION 4: Retrievers
# =============================================================================
# Hybrid retrieval strategy:
#   - BM25  : keyword-based — finds exact matches (good for specific part names, codes)
#   - FAISS : semantic-based — finds meaning matches (good for symptoms, related codes)
#
# Manual retriever  → BM25 + FAISS on manual_chunks
# OBD retriever     → BM25 + FAISS on obd_docs
# retrieve_combined → merges all results, deduplicates, ensures both sources contribute

# Manual retrievers
bm25_manual = BM25Retriever.from_documents(manual_chunks)
bm25_manual.k = 3

faiss_manual_retriever = vectors.as_retriever(
    search_type="mmr",  # MMR = Maximum Marginal Relevance (avoids returning duplicate chunks)
    search_kwargs={"k": 5, "fetch_k": 30, "lambda_mult": 0.7}
)

# OBD retrievers
obd_bm25 = BM25Retriever.from_documents(obd_docs)
obd_bm25.k = 3

obd_faiss_retriever = obd_vectors.as_retriever(
    search_kwargs={"k": 6}  # fetch more OBD codes to surface the full related family
)

def retrieve_combined(query: str):
    """Retrieve from all 4 retrievers and merge, deduplicating by page content."""
    manual_faiss_results = faiss_manual_retriever.invoke(query)
    manual_bm25_results  = bm25_manual.invoke(query)
    obd_bm25_results     = obd_bm25.invoke(query)
    obd_faiss_results    = obd_faiss_retriever.invoke(query)

    seen = set()
    combined = []
    for doc in manual_faiss_results + manual_bm25_results + obd_bm25_results + obd_faiss_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)
    return combined

def format_docs(docs):
    """Format retrieved documents into a single context string for the LLM."""
    return "\n\n".join([
        f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}"
        for doc in docs
    ])

# =============================================================================
# SECTION 5: LLM and Prompts
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# ── HyDE Prompt ──────────────────────────────────────────────────────────────
# HyDE (Hypothetical Document Embeddings):
# Instead of embedding the user's question directly, we first ask the LLM to
# generate a hypothetical answer passage, then retrieve based on that passage.
# This improves retrieval quality because the hypothetical passage is closer
# in style and content to actual manual text.

hyde_prompt = ChatPromptTemplate.from_template("""
Write a passage from a vehicle repair manual or owner's manual that would directly answer this question.
Include technical details such as symptoms, causes, diagnostic steps, or repair procedures if relevant.

Question: {question}
Passage:
""")

hyde_chain = hyde_prompt | llm | StrOutputParser()

# ── Diagnostic Prompt ─────────────────────────────────────────────────────────
# Main prompt that takes the retrieved context and generates a structured diagnosis.

prompt = ChatPromptTemplate.from_template("""
You are an expert vehicle diagnostics assistant helping mechanics and car owners diagnose and fix vehicle issues.
Answer the question using ONLY the provided context from the vehicle manuals.
Focus only on mechanical and combustion-related causes.
Ignore unrelated issues such as battery, immobilizer, or starting problems.
- Differentiate between Standard OBD-II codes and Manufacturer-specific codes
- In Related Error Codes: list ALL OBD codes present in the context that are relevant, do not truncate

Rules:
- Always mention the specific system (engine, transmission, brakes, etc.) involved
- List each Possible Cause as a separate bullet point — minimum 3 causes if available in context
- If the question mentions when or where the symptom occurs (e.g., at highway speed, when cold, under load), use that context to identify the most likely cause first
- Include OBD codes, torque specs, or part numbers if available in the context
- If diagnostic steps are available, list them in order
- For Severity: consider both the OBD code AND the symptom. If the symptom occurs at high speed or affects safety (braking, steering, highway driving), mark High
- Only say "This information is not available in the provided manuals" if the context has absolutely no relevant information
- Do not make assumptions or use external knowledge outside the provided context
- If safety is a concern, clearly flag it

Context:
{context}

Question: {question}

Answer STRICTLY in this format:

System:
- ...

Possible Causes:
- ...

Recommended Checks:
- ...

Related Error Codes:
- ...

Severity:
- Low / Medium / High

Confidence Score:
- 0.xx

Answer:
""")

# =============================================================================
# SECTION 6: RAG Chain
# =============================================================================
# Pipeline:
#   question → HyDE generates hypothetical passage
#            → retrieve_combined fetches relevant docs (manuals + OBD)
#            → format_docs formats them into context string
#            → prompt + LLM generates structured diagnosis

document_chain = (
    RunnableParallel(
        context=RunnableLambda(lambda x: hyde_chain.invoke({"question": x["question"]}))
                | RunnableLambda(retrieve_combined)
                | RunnableLambda(format_docs),
        question=RunnableLambda(lambda x: x["question"])
    )
    | prompt | llm | StrOutputParser()
)

# =============================================================================
# Quick Test (run directly: python chain.py)
# =============================================================================
if __name__ == "__main__":
    test_queries = [
        "What are the causes and fixes for engine misfire?",
        "Why does a car overheat and how to fix it?",
        "Why is my car not starting?"
    ]
    for query in test_queries:
        answer = document_chain.invoke({"question": query})
        print(f"Q: {query}\nA: {answer}\n{'='*60}\n")
