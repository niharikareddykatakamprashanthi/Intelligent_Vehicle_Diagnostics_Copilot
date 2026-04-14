# =============================================================================
# evaluate.py — RAGAS Evaluation for the Vehicle Diagnostics Copilot
# =============================================================================
# Compatible with RAGAS 0.4.x (collections metrics API)
#
# Metrics:
#   Faithfulness      — Answer is grounded in retrieved context (no hallucination)
#   AnswerRelevancy   — Answer addresses the question asked
#   ContextRelevance  — Retrieved chunks are relevant to the question
#
# Note: RAGAS 0.4.x collections metrics use .score() directly — they are not
# compatible with ragas.evaluate(). Each metric is scored per sample manually.
#
# Run with:
#   cd code
#   python evaluate.py
#
# Results print to stdout and save to ragas_results.csv in the project root.
# =============================================================================

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from ragas.metrics.collections.faithfulness.metric import Faithfulness
from ragas.metrics.collections.answer_relevancy.metric import AnswerRelevancy
from ragas.metrics.collections.context_relevance.metric import ContextRelevance
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

from chain import hyde_chain, retrieve_combined, format_docs, prompt, llm

# =============================================================================
# SECTION 1: Test Questions
# =============================================================================

TEST_QUESTIONS = [
    "What causes engine misfire in a Honda Civic?",
    "2018 Toyota Camry P0300 engine vibration at highway speed",
    "Toyota Camry 2018 transmission slipping between gears",
    "Why does my car overheat and how do I fix it?",
    "P0420 catalyst system efficiency below threshold",
    "Toyota Corolla 2019 rough idle at startup",
    "Ford F150 2018 engine oil pressure low warning light",
    "Car vibrates and shakes when braking at highway speed",
    "Why does my car stall at low speeds?",
    "What does P0301 mean and what are the recommended repair steps?",
]

# =============================================================================
# SECTION 2: Pipeline Runner
# =============================================================================

def run_pipeline(question: str) -> dict:
    """Run HyDE → retrieval → generation. Returns question, answer, contexts."""
    hyde_passage = hyde_chain.invoke({"question": question})
    docs = retrieve_combined(hyde_passage)
    context_str = format_docs(docs)
    answer = (prompt | llm).invoke({
        "context": context_str,
        "question": question,
    }).content
    return {
        "question": question,
        "answer": answer,
        "contexts": [doc.page_content for doc in docs],
    }

# =============================================================================
# SECTION 3: Main
# =============================================================================

def main():
    print("=" * 65)
    print("RAGAS Evaluation — Intelligent Vehicle Diagnostics Copilot")
    print("=" * 65)

    # ── Step 1: Generate answers ───────────────────────────────────────────────
    print(f"\nGenerating answers for {len(TEST_QUESTIONS)} questions...\n")
    samples = []
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"  [{i + 1}/{len(TEST_QUESTIONS)}] {q[:70]}")
        samples.append(run_pipeline(q))

    # ── Step 2: Initialise metrics ─────────────────────────────────────────────
    openai_client = AsyncOpenAI()
    ragas_llm = llm_factory("gpt-4o-mini", client=openai_client, max_tokens=4096)
    ragas_embeddings = RagasOpenAIEmbeddings(
        client=openai_client, model="text-embedding-3-small"
    )

    faithfulness_metric   = Faithfulness(llm=ragas_llm)
    answer_relevancy_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    context_relevance_metric = ContextRelevance(llm=ragas_llm)

    # ── Step 3: Score each sample ──────────────────────────────────────────────
    print("\nRunning RAGAS metrics (this makes additional LLM calls)...\n")
    records = []
    for i, s in enumerate(samples):
        q   = s["question"]
        ans = s["answer"]
        ctx = s["contexts"]

        print(f"  Scoring [{i + 1}/{len(samples)}] {q[:60]}")

        f_result  = faithfulness_metric.score(
            user_input=q, response=ans, retrieved_contexts=ctx
        )
        ar_result = answer_relevancy_metric.score(
            user_input=q, response=ans
        )
        cr_result = context_relevance_metric.score(
            user_input=q, retrieved_contexts=ctx
        )

        records.append({
            "question":          q,
            "faithfulness":      f_result.value,
            "answer_relevancy":  ar_result.value,
            "context_relevance": cr_result.value,
        })

    df = pd.DataFrame(records)

    # ── Step 4: Print per-question results ────────────────────────────────────
    print("\n" + "=" * 65)
    print("Per-Question Results")
    print("=" * 65)

    for _, row in df.iterrows():
        print(f"\nQ: {row['question'][:70]}")
        for col in ["faithfulness", "answer_relevancy", "context_relevance"]:
            val = row[col]
            print(f"  {col:<25} {val:.4f}")

    # ── Step 5: Aggregate scores ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Aggregate Scores (mean across all questions)")
    print("=" * 65)
    for col in ["faithfulness", "answer_relevancy", "context_relevance"]:
        mean_val = df[col].mean()
        bar_len  = int(mean_val * 30)
        bar      = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {col:<25} {mean_val:.4f}  |{bar}|")

    # ── Step 6: Save CSV ───────────────────────────────────────────────────────
    output_path = Path(__file__).parent.parent / "ragas_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
