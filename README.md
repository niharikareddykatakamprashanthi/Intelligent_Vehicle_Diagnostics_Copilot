<h1>Intelligent Vehicle Diagnostics Copilot</h1>

A RAG-based AI assistant that diagnoses vehicle issues using actual repair manuals and OBD code databases — not just the LLM's general knowledge.

You describe a symptom or paste an OBD code, and it tells you what system is affected, what is likely causing the issue, what to check step by step, which error codes are related, how severe the problem is, and what it will cost to fix.



<h2>The Problem This Solves</h2>

When a check engine light comes on or a car starts behaving oddly, most people either take it straight to a mechanic without any context, or they search online and get generic answers that may not apply to their specific vehicle.

AI chatbots are not much better. They answer from training data which is generalized, sometimes outdated, and not tied to any actual vehicle documentation. You get plausible-sounding answers with no way to verify where they came from.

This project fixes that. Instead of relying on what the LLM already knows, it first retrieves relevant content from real vehicle repair manuals and a structured OBD code database, then uses that retrieved content as the basis for the answer. Every answer is grounded in actual source material from the specific vehicle's documentation.



<h2>What It Does</h2>

There are two ways to use the system:

Diagnose — you enter a symptom, optionally an OBD code, and optionally your vehicle's make, model, and year. The system retrieves relevant sections from repair manuals and OBD databases and returns a structured answer covering the affected system, possible causes, recommended diagnostic steps, related error codes, severity level, and a confidence score.

Full Cost Diagnosis — runs the same diagnosis but then passes it through two more agents. The second agent acts as a senior technician and reviews the diagnosis for accuracy. The third agent estimates the cost of repair including parts breakdown, labor hours, total cost range, repair time, difficulty level, and whether it can be done DIY or needs a professional.



<h2>How It Works — Step by Step</h2>

When you submit a question, here is exactly what happens inside the system:

<h3>Step 1 — HyDE (Hypothetical Document Embeddings)</h3>

Your question is not used directly as the retrieval query. Instead, the LLM first generates a hypothetical answer — a fake passage written in the style of a repair manual that would answer your question. This hypothetical passage is then used as the retrieval query.

Why? Because your question ("why is my engine misfiring?") and the actual manual text ("cylinder misfire can result from worn spark plugs, faulty ignition coils, or injector failure") are written very differently. Embedding the hypothetical passage instead closes that gap and produces far better retrieval results.

<h3>Step 2 — Hybrid Retrieval</h3>

The hypothetical passage is passed to two retrievers running in parallel:

BM25 is a keyword-based retrieval method. It tokenizes the query, builds an inverted index, and scores documents based on how frequently and how rarely the query words appear across the corpus. It is excellent at finding exact matches — specific part names, OBD codes like P0300, or technical terms.

FAISS is a vector-based semantic search. It converts text into numerical vectors using OpenAI's text-embedding-3-small model, then finds the closest vectors in the index using cosine similarity. It understands meaning, so it can match "rough combustion in cylinder 3" to content about misfires even if the word "misfire" is never used.

Both retrievers return results, which are merged and deduplicated before being passed to the LLM.

<h3>Step 3 — Dual FAISS Indexes</h3>

The system maintains two completely separate FAISS indexes:

The manual index contains chunks from all repair and owner manual PDFs. Each chunk is tagged with its source (repair_manual or owner_manual) so the LLM knows where the information came from. This index is used to retrieve actual diagnostic procedures, torque specs, part numbers, and repair steps.

The OBD index contains one document per OBD code. Keeping this separate from the manual index is important — if they were combined, a query about engine misfires would mostly return OBD code documents (since they explicitly mention "misfire") and drown out the actual repair procedures. With separate indexes, both sources always contribute to the final context.

<h3>Step 4 — Context Assembly</h3>

The retrieved documents from all four retrievers (BM25 manual, FAISS manual, BM25 OBD, FAISS OBD) are merged and formatted into a single context block. Each document is labeled with its source. This full context is injected into the diagnostic prompt.

<h3>Step 5 — LLM Generation</h3>

The LLM (GPT-4o-mini) reads the context and generates a structured answer. It is instructed to use only what is in the context, not its own general knowledge. The output format is fixed: System, Possible Causes, Recommended Checks, Related Error Codes, Severity, and Confidence Score.

<h3>Step 6 — Multi-Agent Review (Full Cost Diagnosis only)</h3>

If Full Cost Diagnosis is selected, the output from Step 5 is passed through two more agents via a LangGraph state machine:

Agent 2 is a validation agent. It plays the role of a senior technician reviewing the junior technician's diagnosis. It checks for missing causes, incorrect checks, or anything that needs revision, and outputs a Validation Status along with corrections.

Agent 3 is a cost estimation agent. It reads the diagnosis and outputs a parts breakdown with cost ranges, estimated labor, total repair cost range, estimated time, difficulty level (DIY or professional), and priority (fix immediately, can wait, or monitor).

All three agents share a common state object that gets passed from one to the next through the graph.



<h2>Project Structure</h2>

```
Intelligent_Vehicle_Diagnostics_Copilot/
│
├── code/
│   ├── chain.py          # Core RAG pipeline — document loading, vector stores,
│   │                     # retrievers, HyDE, prompts, and the final LLM chain
│   │
│   ├── agents.py         # LangGraph multi-agent pipeline — DiagnosticState,
│   │                     # three agent functions, graph definition, and run_agents()
│   │
│   ├── app.py            # Streamlit web UI — input fields, buttons, query builder,
│   │                     # and result display for both diagnosis modes
│   │
│   └── api.py            # FastAPI REST backend — /diagnose and /diagnose-full
│                         # endpoints with Pydantic request/response models
│
├── data/
│   ├── manuals/
│   │   ├── repair_manual/    # PDF repair manuals — one file per vehicle
│   │   └── owner_manual/     # PDF owner manuals
│   │
│   └── formatted_obd.json    # OBD code database — each entry has a code,
│                             # description, and system tag (engine, transmission, etc.)
│
├── faiss_index/          # FAISS index for manual chunks (auto-built on first run)
├── faiss_obd_index/      # FAISS index for OBD code documents (auto-built on first run)
│
├── requirements.txt      # All Python dependencies with comments
├── .env                  # OPENAI_API_KEY — not committed to git
└── .gitignore
```



<h2>Setup</h2>

<h3>1. Clone and install</h3>

```bash
git clone https://github.com/niharikareddykatakamprashanthi/Intelligent_Vehicle_Diagnostics_Copilot.git
cd Intelligent_Vehicle_Diagnostics_Copilot
pip install -r requirements.txt
```

<h3>2. Add your OpenAI API key</h3>

Create a .env file in the project root and add your key:

```
OPENAI_API_KEY=your_key_here
```

The app uses GPT-4o-mini for generation and text-embedding-3-small for embeddings. Both are billed through your OpenAI account.

<h3>3. Add vehicle manuals</h3>

Place PDF files into the correct folders:

```
data/manuals/repair_manual/   ← repair manuals (e.g. 2018_Camry_repair_manual.pdf)
data/manuals/owner_manual/    ← owner manuals  (e.g. 2018-camry_owner_manual.pdf)
```

On the first run, the app will read all PDFs, split them into 1000-character chunks with 200-character overlap, embed each chunk using text-embedding-3-small, and save two FAISS indexes to disk. This takes 1-2 minutes depending on how many PDFs you have. Every run after that loads the saved indexes instantly.

If you add new PDFs later, delete the faiss_index/ folder so it rebuilds with the new content included.



<h2>Running</h2>

<h3>Streamlit UI</h3>

```bash
cd code
streamlit run app.py
```

Opens in your browser at http://localhost:8501

<h3>FastAPI Backend</h3>

```bash
cd code
uvicorn api:app --reload
```

Interactive API docs available at http://localhost:8000/docs



<h2>API Reference</h2>

<h3>GET /</h3>

Health check. Confirms the API is running.

```json
{ "status": "running", "service": "Vehicle Diagnostics Copilot" }
```

<h3>POST /diagnose</h3>

Quick diagnosis using the RAG pipeline. Only symptoms is required. Adding make, model, year, and obd_code gives the LLM more context and produces more specific answers.

Request:
```json
{
  "symptoms": "engine misfiring at highway speed",
  "obd_code": "P0300",
  "make": "Toyota",
  "model": "Camry",
  "year": "2018"
}
```

Response:
```json
{
  "question": "Vehicle: 2018 Toyota Camry. OBD Code: P0300. engine misfiring at highway speed",
  "answer": "System:\n- Engine\n\nPossible Causes:\n- ..."
}
```

<h3>POST /diagnose-full</h3>

Full 3-agent diagnosis. Same request body as /diagnose. Returns three separate fields — one per agent.

Response:
```json
{
  "question": "...",
  "diagnosis": "structured diagnosis from Agent 1",
  "validation": "expert review from Agent 2",
  "cost_estimate": "parts, labor, and time estimate from Agent 3"
}
```



<h2>Sample Queries to Try</h2>

These cover different systems and query types to test the full range of the system:

- What causes engine misfire in a Honda Civic?
- P0300 code with engine vibration at highway speed
- Toyota Camry 2018 — transmission slipping
- Car vibrates when braking — what is the cause?
- Why is my car overheating?
- P0420 — catalyst system efficiency below threshold
- Toyota Corolla 2019 — rough idle
- Ford F150 — engine oil pressure low warning
- Why does my car stall at low speeds?
- What does P0301 mean and how do I fix it?

For vehicle-specific queries, fill in the Make, Model, and Year fields in the UI. The query sent to the LLM will include that context automatically.



<h2>Things Worth Knowing</h2>

Adding more vehicles — drop the PDF into data/manuals/repair_manual/ and delete the faiss_index/ folder. It rebuilds on the next startup with the new manual included.

First load is slow — embedding thousands of PDF chunks takes time depending on file size and count. After the first run the indexes are saved and every subsequent load is instant.

Confidence score — the LLM assigns this based on how much relevant context was found during retrieval. A score of 0.90+ means the manual had strong coverage of the topic. A lower score usually means the question touches on something the manuals do not cover well.

OBD code families — querying P0300 (random misfire) also surfaces P0301 through P0308 (cylinder-specific misfires) because the OBD FAISS index uses semantic search to find related codes, not just exact keyword matches.

Severity at highway speed — the prompt is tuned to mark severity as High when the symptom occurs at highway speed or involves braking or steering, regardless of what the OBD code alone would suggest.



<h2>Supported Vehicles</h2>

- Honda Civic 2011 — Repair Manual
- Toyota Camry 2018 — Repair Manual + Owner Manual
- Toyota Corolla 2019 — Repair Manual
- Ford F150 2018 — Repair Manual

More vehicles can be added at any time by dropping in their PDF manuals.



<h2>Tech Stack</h2>

- LangChain — document loading, text splitting, prompt templates, retrieval chains
- LangGraph — multi-agent state machine (StateGraph, TypedDict state)
- OpenAI — GPT-4o-mini for answer generation, text-embedding-3-small for embeddings
- FAISS — vector similarity search with MMR to avoid duplicate chunks
- BM25 — keyword-based retrieval using inverse document frequency scoring
- Streamlit — browser-based UI
- FastAPI — REST API with automatic OpenAPI docs
- python-dotenv — environment variable management
