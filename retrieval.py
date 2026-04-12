"""
Sci-Verify Retrieval System
Multi-Level Verification: Abstract-First approach with optional full-text.
Uses OpenAlex (PyAlex) for discovery and FAISS for semantic snippet retrieval.
"""

import os
import logging
import tempfile
import numpy as np
import requests
from typing import Optional

from sentence_transformers import SentenceTransformer
import faiss
import pyalex
from pyalex import Works

from models import PaperSource

logger = logging.getLogger(__name__)

# Configure pyalex polite pool
pyalex.config.email = os.getenv("OPENALEX_EMAIL", "sciverify@example.com")

# Lazy-loaded embedding model
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully.")
    return _embedding_model


# ---------- OpenAlex Helpers ----------

def reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


def search_openalex(query: str, max_results: int = 5) -> list[PaperSource]:
    """Search OpenAlex for papers, sorted by citation count."""
    logger.info(f"[RETRIEVAL] Searching OpenAlex: query='{query[:80]}...', max_results={max_results}")
    try:
        results = (
            Works()
            .search(query)
            .sort(cited_by_count="desc")
            .get(per_page=max_results)
        )
    except Exception as e:
        logger.error(f"[RETRIEVAL] OpenAlex search FAILED: {e}")
        return []

    logger.info(f"[RETRIEVAL] OpenAlex returned {len(results)} results.")

    papers = []
    for i, work in enumerate(results):
        authors = []
        for authorship in work.get("authorships", []):
            name = authorship.get("author", {}).get("display_name", "Unknown")
            authors.append(name)

        abstract = ""
        if work.get("abstract_inverted_index"):
            abstract = reconstruct_abstract(work["abstract_inverted_index"])
        elif work.get("abstract"):
            abstract = work["abstract"]

        oa_url = None
        oa_info = work.get("open_access", {})
        if oa_info.get("is_oa") and oa_info.get("oa_url"):
            oa_url = oa_info["oa_url"]

        doi = work.get("doi", "") or ""

        paper = PaperSource(
            source_id=i,
            title=work.get("title", "Untitled") or "Untitled",
            doi=doi,
            authors=authors,
            abstract=abstract,
            oa_pdf_url=oa_url,
            publication_date=work.get("publication_date"),
            openalex_id=work.get("id"),
        )
        papers.append(paper)
        logger.info(
            f"[RETRIEVAL]   Paper {i}: '{paper.title[:60]}' | DOI: {doi} | "
            f"Abstract: {len(abstract)} chars | OA: {'Yes' if oa_url else 'No'}"
        )

    return papers


# ---------- Full-Text Extraction (Level 2) ----------

def try_extract_full_text(paper: PaperSource) -> Optional[str]:
    """Attempt to download and parse full text from an OA PDF.
    Returns None on failure — caller falls back to abstract."""
    if not paper.oa_pdf_url:
        return None
    logger.info(f"[RETRIEVAL] Attempting full-text extraction: {paper.oa_pdf_url}")
    try:
        import pymupdf4llm

        response = requests.get(paper.oa_pdf_url, timeout=15)
        if response.status_code != 200:
            logger.warning(f"[RETRIEVAL] PDF download failed: HTTP {response.status_code}")
            return None

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        md_text = pymupdf4llm.to_markdown(tmp_path)
        os.unlink(tmp_path)

        if len(md_text) < 100:
            logger.warning(f"[RETRIEVAL] Extracted text too short ({len(md_text)} chars), falling back to abstract.")
            return None
        logger.info(f"[RETRIEVAL] Full-text extracted: {len(md_text)} chars.")
        return md_text
    except Exception as e:
        logger.warning(f"[RETRIEVAL] Full-text extraction FAILED for {paper.doi}: {e}")
        return None


def get_paper_text(paper: PaperSource, attempt_full_text: bool = False) -> str:
    """Get the best available text for a paper (Abstract-First)."""
    if attempt_full_text and paper.oa_pdf_url:
        full_text = try_extract_full_text(paper)
        if full_text:
            paper.full_text = full_text
            return full_text
    return paper.abstract or ""


# ---------- FAISS Indexing ----------

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def build_snippet_index(
    papers: list[PaperSource],
    attempt_full_text: bool = False,
) -> tuple:
    """Build a FAISS index from paper texts.
    Returns (index, snippets_list, source_id_list)."""
    logger.info(f"[FAISS] Building snippet index from {len(papers)} papers...")
    model = get_embedding_model()

    all_snippets: list[str] = []
    snippet_source_ids: list[int] = []

    for paper in papers:
        text = get_paper_text(paper, attempt_full_text)
        if not text:
            logger.warning(f"[FAISS] No text for paper {paper.source_id} ('{paper.title[:40]}'), skipping.")
            continue
        chunks = chunk_text(text)
        logger.debug(f"[FAISS] Paper {paper.source_id}: {len(chunks)} chunks from {len(text)} chars.")
        for chunk in chunks:
            all_snippets.append(chunk)
            snippet_source_ids.append(paper.source_id)

    if not all_snippets:
        logger.warning("[FAISS] No snippets to index — returning empty index.")
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        return index, [], []

    logger.info(f"[FAISS] Encoding {len(all_snippets)} snippets...")
    embeddings = model.encode(all_snippets, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"[FAISS] Index built: {index.ntotal} vectors, dim={dim}.")

    return index, all_snippets, snippet_source_ids


def retrieve_relevant_snippets(
    claim_text: str,
    index,
    all_snippets: list[str],
    snippet_source_ids: list[int],
    top_k: int = 3,
) -> list[tuple[str, int, float]]:
    """Retrieve top-k relevant snippets for a claim.
    Returns list of (snippet_text, source_id, score)."""
    if not all_snippets or index.ntotal == 0:
        logger.warning(f"[FAISS] Empty index — cannot retrieve for claim: '{claim_text[:50]}...'")
        return []

    model = get_embedding_model()
    query_emb = model.encode([claim_text], show_progress_bar=False)
    query_emb = np.array(query_emb, dtype="float32")
    faiss.normalize_L2(query_emb)

    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_emb, k)

    results = []
    for j, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        results.append(
            (all_snippets[idx], snippet_source_ids[idx], float(distances[0][j]))
        )
    logger.info(
        f"[FAISS] Retrieved {len(results)} snippets for claim: '{claim_text[:50]}...' | "
        f"Scores: {[f'{r[2]:.3f}' for r in results]}"
    )
    return results
