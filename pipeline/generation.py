"""
LLM Generation — dual pipeline:
  - LMStudioPipeline  → local dev via LM Studio (FREE, no API key)
  - GroqPipeline      → HuggingFace deployment via Groq API (FREE tier)

Switch with LLM_PROVIDER env var: "lmstudio" | "groq"
"""
import re
from typing import List, Dict, Any
from openai import OpenAI
from groq import Groq

from pipeline.config import (
    LLM_PROVIDER,
    LM_STUDIO_URL,
    LM_STUDIO_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)

SYSTEM_PROMPT = """You are an enterprise knowledge base assistant.
Answer questions strictly based on the provided context.
Do NOT include any citations, source references, or file names in your answer text.
Sources are shown separately in the UI below your answer.
If the context does not contain enough information, say "I don't have enough information in the provided documents."
Never fabricate facts."""

ANSWER_TEMPLATE = """Context:
{context}

Question: {question}

Instructions: Answer using only the context above. Do not mention filenames, page numbers, or source labels in your answer.
"""


def _strip_source_block(text: str) -> str:
    """Remove any trailing Sources/citations block the LLM appends."""
    text = re.sub(r'\n+\[?Sources?:?\]?.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _build_context(hits: List[Dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        source = hit.get("metadata", {}).get("source_file", "unknown")
        page = hit.get("metadata", {}).get("page", "?")
        parts.append(f"[{i}] Source: {source}, Page {page}\n{hit['text']}")
    return "\n\n---\n\n".join(parts)


# ── LM Studio Pipeline (LOCAL — FREE) ────────────────────────────────────────

class LMStudioPipeline:
    """
    LM STUDIO SETUP (do this once):
    1. Download LM Studio from https://lmstudio.ai
    2. Search and download: "Meta-Llama-3.1-8B-Instruct-GGUF" (Q4 version, ~5GB)
    3. Go to Local Server tab → click Start Server
    4. Server runs at http://localhost:1234
    That's it — no API key needed.
    """

    def __init__(self):
        self.client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        self.model = LM_STUDIO_MODEL

    def generate(self, question: str, hits: List[Dict]) -> Dict[str, Any]:
        context = _build_context(hits)
        prompt = ANSWER_TEMPLATE.format(context=context, question=question)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        answer = _strip_source_block(response.choices[0].message.content)
        return {
            "answer": answer,
            "sources": [h.get("metadata", {}) for h in hits],
            "provider": "lmstudio",
            "model": self.model,
        }


# ── Groq Pipeline (HUGGINGFACE DEPLOYMENT — FREE TIER) ───────────────────────

class GroqPipeline:
    """
    GROQ SETUP (for HuggingFace deployment):
    1. Get free API key at https://console.groq.com
    2. Set GROQ_API_KEY in HuggingFace Space secrets
    3. Set LLM_PROVIDER=groq in HuggingFace Space env vars
    Rate limits: 30 req/min, 14,400 req/day on free tier.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL

    def generate(self, question: str, hits: List[Dict]) -> Dict[str, Any]:
        context = _build_context(hits)
        prompt = ANSWER_TEMPLATE.format(context=context, question=question)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        answer = _strip_source_block(response.choices[0].message.content)
        return {
            "answer": answer,
            "sources": [h.get("metadata", {}) for h in hits],
            "provider": "groq",
            "model": self.model,
        }


# ── Factory — picks pipeline based on LLM_PROVIDER env var ───────────────────

def get_pipeline():
    if LLM_PROVIDER == "groq":
        return GroqPipeline()
    return LMStudioPipeline()
