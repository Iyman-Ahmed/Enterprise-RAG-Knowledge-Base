"""
End-to-end local test: ingestion → retrieval → generation → source citation.
Run with: python test_local.py
Requires LM Studio running at http://localhost:1234
"""
import os, sys, tempfile, textwrap, time

# ── 0. Silence ChromaDB telemetry noise ──────────────────────────────────────
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m~\033[0m"

errors = []


def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f"  →  {detail}" if detail else ""))
        errors.append(label)


# ── 1. Config sanity ──────────────────────────────────────────────────────────
print("\n[1] Config")
from pipeline.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL, TOP_K_RERANK

check("CHUNK_SIZE >= 800",      CHUNK_SIZE >= 800,      f"got {CHUNK_SIZE}")
check("CHUNK_OVERLAP >= 150",   CHUNK_OVERLAP >= 150,   f"got {CHUNK_OVERLAP}")
check("TOP_K_RETRIEVAL >= 12",  TOP_K_RETRIEVAL >= 12,  f"got {TOP_K_RETRIEVAL}")
check("TOP_K_RERANK >= 6",      TOP_K_RERANK >= 6,      f"got {TOP_K_RERANK}")

# ── 2. pdfplumber table extraction ────────────────────────────────────────────
print("\n[2] PDF table extraction (pdfplumber)")

# Build a minimal PDF with a two-row table using only stdlib + pypdf/pdfplumber
# We'll create a PDF via raw PDF syntax (no extra deps)
MINIMAL_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]
/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 220>>
stream
BT /F1 12 Tf
50 700 Td (Interest Rates Table) Tj
0 -20 Td (Purchases | Monthly Rate | Annual Rate) Tj
0 -20 Td (Standard purchases | 2.144% | 28.9%) Tj
0 -20 Td (Cash advances | 2.207% | 29.9%) Tj
0 -20 Td (Balance transfers | 2.144% | 28.9%) Tj
ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000546 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
633
%%EOF"""

with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
    f.write(MINIMAL_PDF)
    tmp_pdf = f.name

try:
    from pipeline.ingestion import _load_pdf
    docs = _load_pdf(tmp_pdf)
    combined = " ".join(d.page_content for d in docs)
    check("PDF loads without error",       len(docs) > 0)
    check("Page metadata present",         all("page" in d.metadata for d in docs))
    check("Rate content extracted",        "28.9" in combined or "2.144" in combined,
          f"got: {combined[:200]!r}")
finally:
    os.unlink(tmp_pdf)

# ── 3. Ingestion pipeline (txt sample doc) ────────────────────────────────────
print("\n[3] Ingestion pipeline")
from pipeline.ingestion import ingest_file

chunks = ingest_file("data/sample_docs/company_handbook.txt")
check("Chunks produced",            len(chunks) > 0,          f"got {len(chunks)}")
check("source_file metadata set",   all(c.metadata.get("source_file") for c in chunks))
check("chunk_id metadata set",      all("chunk_id" in c.metadata for c in chunks))
check("Chunk size reasonable",
      all(len(c.page_content) <= CHUNK_SIZE * 1.2 for c in chunks),
      "some chunks exceed limit")

# ── 4. Vector store round-trip ────────────────────────────────────────────────
print("\n[4] Vector store")
from vectorstore.chroma_manager import ChromaManager

cm = ChromaManager()
cm.delete_collection()                         # start clean
cm.add_documents(chunks)

count = cm.count()
check("Documents indexed",          count == len(chunks),     f"{count} vs {len(chunks)}")

hits = cm.similarity_search("What is the company mission?", top_k=3)
check("Similarity search returns results", len(hits) > 0)
check("Hits have score field",             all("score" in h for h in hits))
check("Hits have metadata",               all("metadata" in h for h in hits))
check("Source file in hit metadata",
      all(h["metadata"].get("source_file") for h in hits))

# ── 5. Hybrid retrieval ───────────────────────────────────────────────────────
print("\n[5] Hybrid retrieval")
from pipeline.retrieval import HybridRetriever

retriever = HybridRetriever()
results = retriever.search("company values and culture")
check("Hybrid search returns results",   len(results) > 0)
check(f"Returns <= TOP_K_RERANK={TOP_K_RERANK}", len(results) <= TOP_K_RERANK)
check("Results have text field",         all("text" in r for r in results))

# ── 6. LM Studio connectivity ─────────────────────────────────────────────────
print("\n[6] LM Studio")
import urllib.request, json

try:
    with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=3) as r:
        data = json.loads(r.read())
    models = [m["id"] for m in data.get("data", [])]
    check("LM Studio reachable",       True)
    check("A model is loaded",         len(models) > 0, f"models: {models}")
    lm_ok = True
except Exception as e:
    print(f"  {SKIP}  LM Studio not reachable ({e}) — skipping generation test")
    lm_ok = False

# ── 7. End-to-end generation + source citation ────────────────────────────────
print("\n[7] End-to-end generation")
if lm_ok:
    from pipeline.generation import get_pipeline

    pipeline = get_pipeline()
    question = "How many days of paid annual leave are employees entitled to?"
    t0 = time.time()
    response = pipeline.generate(question, results)
    latency = time.time() - t0

    answer = response.get("answer", "")
    sources = response.get("sources", [])

    check("Answer is non-empty",              len(answer) > 20,     f"got: {answer[:80]!r}")
    check("Sources list returned",            len(sources) > 0,     f"sources: {sources}")
    check("Source has source_file",
          any(s.get("source_file") for s in sources))
    check("Answer contains source citation",
          "[Source:" in answer or "Source:" in answer,
          f"answer snippet: {answer[:200]!r}")
    check(f"Latency < 30s",                  latency < 30,         f"{latency:.1f}s")
    print(f"\n  Answer preview:\n{textwrap.indent(answer[:400], '    ')}")
else:
    print(f"  {SKIP}  Skipped (LM Studio offline)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
if errors:
    print(f"\033[91mFAILED\033[0m  {len(errors)} check(s) failed:")
    for e in errors:
        print(f"  • {e}")
    sys.exit(1)
else:
    print(f"\033[92mALL CHECKS PASSED\033[0m")
