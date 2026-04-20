"""
End-to-end pipeline test suite.
Tests ingestion → embedding → retrieval → generation → guardrails.
Run: python3 -m pytest tests/test_pipeline.py -v
"""
import os
import sys
import json
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ingestion import ingest_file
from pipeline.embeddings import embed_query, embed_texts
from vectorstore.chroma_manager import ChromaManager
from pipeline.retrieval import HybridRetriever, reciprocal_rank_fusion
from pipeline.generation import get_pipeline
from pipeline.guardrails import check_hallucination

SAMPLE_DOC_1 = "data/sample_docs/company_handbook.txt"
SAMPLE_DOC_2 = "data/sample_docs/product_faq.txt"
TEST_DATASET  = "evaluation/test_dataset.json"
REPORT_PATH   = "evaluation/test_report.json"

# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def indexed_retriever():
    """Index both sample docs once for the whole session."""
    chroma = ChromaManager()
    chroma.delete_collection()  # fresh slate

    for path in [SAMPLE_DOC_1, SAMPLE_DOC_2]:
        chunks = ingest_file(path)
        for c in chunks:
            c.metadata["source_file"] = os.path.basename(path)
        chroma.add_documents(chunks)

    retriever = HybridRetriever()
    retriever.invalidate_bm25_cache()
    return retriever


@pytest.fixture(scope="session")
def pipeline():
    return get_pipeline()


@pytest.fixture(scope="session")
def test_questions():
    with open(TEST_DATASET) as f:
        return json.load(f)


# ── Layer 1: Ingestion ────────────────────────────────────────

class TestIngestion:
    def test_txt_loads(self):
        chunks = ingest_file(SAMPLE_DOC_1)
        assert len(chunks) > 0, "No chunks produced from handbook"

    def test_chunks_have_metadata(self):
        chunks = ingest_file(SAMPLE_DOC_1)
        for c in chunks:
            assert "source_file" in c.metadata or "chunk_id" in c.metadata

    def test_chunk_size_reasonable(self):
        chunks = ingest_file(SAMPLE_DOC_1)
        lengths = [len(c.page_content.split()) for c in chunks]
        assert max(lengths) <= 600, f"Chunk too large: {max(lengths)} words"
        assert min(lengths) >= 5,   f"Chunk too small: {min(lengths)} words"

    def test_both_docs_ingest(self):
        c1 = ingest_file(SAMPLE_DOC_1)
        c2 = ingest_file(SAMPLE_DOC_2)
        assert len(c1) > 0 and len(c2) > 0


# ── Layer 2: Embeddings ───────────────────────────────────────

class TestEmbeddings:
    def test_embed_query_returns_vector(self):
        vec = embed_query("What is the leave policy?")
        assert isinstance(vec, list)
        assert len(vec) == 384  # all-MiniLM-L6-v2 dimension

    def test_embed_texts_batch(self):
        vecs = embed_texts(["Hello world", "Leave policy", "Remote work"])
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_similar_queries_close(self):
        import math
        v1 = embed_query("annual leave days")
        v2 = embed_query("how many vacation days")
        dot = sum(a * b for a, b in zip(v1, v2))
        mag = math.sqrt(sum(a**2 for a in v1)) * math.sqrt(sum(b**2 for b in v2))
        cosine = dot / mag
        assert cosine > 0.5, f"Semantically similar queries have low cosine: {cosine:.3f}"


# ── Layer 3: Retrieval ────────────────────────────────────────

class TestRetrieval:
    def test_returns_results(self, indexed_retriever):
        hits = indexed_retriever.search("annual leave policy")
        assert len(hits) > 0

    def test_results_have_text(self, indexed_retriever):
        hits = indexed_retriever.search("remote work days")
        for h in hits:
            assert "text" in h and len(h["text"]) > 10

    def test_rerank_scores_present(self, indexed_retriever):
        hits = indexed_retriever.search("performance bonus rating")
        assert all("rerank_score" in h for h in hits)

    def test_correct_doc_retrieved_leave(self, indexed_retriever):
        hits = indexed_retriever.search("How many annual leave days do employees get?")
        top_text = hits[0]["text"].lower()
        assert any(kw in top_text for kw in ["leave", "20 days", "annual"]), \
            f"Wrong content retrieved: {top_text[:100]}"

    def test_correct_doc_retrieved_api(self, indexed_retriever):
        hits = indexed_retriever.search("API rate limits free tier")
        top_text = hits[0]["text"].lower()
        assert any(kw in top_text for kw in ["api", "rate", "100"]), \
            f"Wrong content retrieved: {top_text[:100]}"

    def test_rrf_fusion(self):
        dense = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.8}]
        bm25  = [{"text": "b", "score": 5.0}, {"text": "c", "score": 3.0}]
        fused = reciprocal_rank_fusion(dense, bm25)
        assert fused[0]["text"] == "b", "RRF should rank doc appearing in both lists highest"


# ── Layer 4: Generation ───────────────────────────────────────

class TestGeneration:
    def test_pipeline_returns_answer(self, indexed_retriever, pipeline):
        hits = indexed_retriever.search("How many sick leave days?")
        result = pipeline.generate("How many sick leave days?", hits)
        assert "answer" in result
        assert len(result["answer"]) > 20

    def test_answer_mentions_number(self, indexed_retriever, pipeline):
        hits = indexed_retriever.search("daily meal allowance travel")
        result = pipeline.generate("What is the daily meal allowance during travel?", hits)
        assert "60" in result["answer"], f"Expected $60 in answer, got: {result['answer'][:200]}"

    def test_answer_has_provider(self, indexed_retriever, pipeline):
        hits = indexed_retriever.search("password reset")
        result = pipeline.generate("How do I reset my password?", hits)
        assert result["provider"] in ("lmstudio", "groq")

    def test_response_time(self, indexed_retriever, pipeline):
        hits = indexed_retriever.search("401k match")
        start = time.time()
        pipeline.generate("How does the 401k match work?", hits)
        elapsed = time.time() - start
        assert elapsed < 60, f"Generation took too long: {elapsed:.1f}s"


# ── Layer 5: Guardrails ───────────────────────────────────────

class TestGuardrails:
    def test_grounded_answer_passes(self):
        hits = [{"text": "Employees get 20 days of annual leave per year.", "metadata": {}}]
        result = check_hallucination("Employees receive 20 days of annual leave.", hits)
        assert result["hallucination_detected"] is False or result["score"] > 0.4

    def test_fabricated_answer_flagged(self):
        hits = [{"text": "Employees get 20 days of annual leave per year.", "metadata": {}}]
        result = check_hallucination(
            "Employees get unlimited leave and free flights to Mars.", hits
        )
        assert result["hallucination_detected"] is True or result["score"] < 0.8

    def test_result_has_required_keys(self):
        hits = [{"text": "Some context.", "metadata": {}}]
        result = check_hallucination("Some answer.", hits)
        assert all(k in result for k in ["hallucination_detected", "score", "flagged_sentences"])


# ── Layer 6: End-to-End Accuracy ─────────────────────────────

class TestEndToEndAccuracy:
    """Runs all 15 test questions and measures keyword hit rate."""

    def test_full_accuracy(self, indexed_retriever, pipeline, test_questions):
        hits_count = 0
        results = []

        for item in test_questions:
            q = item["question"]
            expected = item["ground_truth"].lower()

            retrieval_hits = indexed_retriever.search(q)
            gen_result = pipeline.generate(q, retrieval_hits)
            answer = gen_result["answer"].lower()
            hallucination = check_hallucination(gen_result["answer"], retrieval_hits)

            # extract key numeric/keyword facts from ground truth for matching
            key_terms = [w for w in expected.split() if len(w) > 3 and w.isalpha() or w.replace("$","").replace("%","").isdigit()]
            matched = sum(1 for t in key_terms[:5] if t in answer)
            keyword_hit = matched >= 2

            if keyword_hit:
                hits_count += 1

            results.append({
                "question": q,
                "ground_truth": item["ground_truth"],
                "answer": gen_result["answer"],
                "keyword_hit": keyword_hit,
                "hallucination_detected": hallucination["hallucination_detected"],
                "hallucination_score": hallucination["score"],
                "provider": gen_result["provider"],
            })

        accuracy = hits_count / len(test_questions)
        hallucination_rate = sum(1 for r in results if r["hallucination_detected"]) / len(results)

        # save full report
        report = {
            "total_questions": len(test_questions),
            "keyword_accuracy": round(accuracy, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "per_question": results,
        }
        os.makedirs("evaluation", exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*50}")
        print(f"  Keyword Accuracy  : {accuracy:.1%} ({hits_count}/{len(test_questions)})")
        print(f"  Hallucination Rate: {hallucination_rate:.1%}")
        print(f"  Report saved to   : {REPORT_PATH}")
        print(f"{'='*50}")

        assert accuracy >= 0.5, f"Accuracy too low: {accuracy:.1%}"
