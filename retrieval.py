"""
Sci-Verify Retrieval System
============================
Handles all paper discovery, metadata extraction, and text retrieval.

Key components:
  1. OpenAlex Search — find papers via the free OpenAlex API (pyalex wrapper)
  2. Progressive Relaxation Search — 4-level cascading queries to handle
     hallucinated titles, authors, or dates from AI-generated text
  3. Full-Text Extraction — download OA PDFs and parse with PyMuPDF4LLM
  4. Abstract Fallback — if PDF is unavailable/paywalled, use the abstract
  5. FAISS Indexing — chunk text, embed with sentence-transformers, and index
     for semantic similarity retrieval
"""

import os
import logging
import tempfile
import numpy as np
import requests
from typing import Optional

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import pyalex
from pyalex import Works, Authors
from thefuzz import fuzz

from models import PaperSource, CitationTarget, MetadataDelta

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN CONCEPT MAP
# ══════════════════════════════════════════════════════════════════════════════
# OpenAlex uses concept IDs to categorize papers by field.
# This map is used in Level 4 of the Progressive Relaxation Search to lock
# queries to a specific academic domain (prevents cross-field false positives).

DOMAIN_MAP = {
    "computer science": "C41008148",
    "biology": "C86803240",
    "medicine": "C71924100",
    "physics": "C121332964",
    "chemistry": "C185592680",
    "mathematics": "C33923547",
    "psychology": "C15744967",
    "economics": "C162324750",
    "sociology": "C144024400"
}


# ══════════════════════════════════════════════════════════════════════════════
# METADATA DELTA GENERATION
# ══════════════════════════════════════════════════════════════════════════════
# Compares claimed citation metadata (from the AI text) against the real paper
# found in OpenAlex. Used in the UI to show "Claimed vs Found" side-by-side.

def generate_delta(target: CitationTarget, found_paper: PaperSource) -> MetadataDelta:
    """Compare claimed vs found metadata and generate a MetadataDelta for the UI."""
    delta = MetadataDelta(
        claimed_title=target.title or target.core_topic or "",
        found_title=found_paper.title or "",
        claimed_year=target.year,
        found_year=int(found_paper.publication_date[:4]) if found_paper.publication_date and len(found_paper.publication_date) >= 4 else None,
    )
    # Flag year mismatch (e.g., ChatGPT said 2019 but paper is from 2017)
    if delta.claimed_year and delta.found_year and delta.claimed_year != delta.found_year:
        delta.is_year_mismatch = True
    # Flag title alias (e.g., truncated or abbreviated title, fuzzy ratio < 60%)
    if target.title and found_paper.title and fuzz.ratio(target.title.lower(), found_paper.title.lower()) < 60:
        delta.is_title_alias = True
    return delta


# Configure pyalex polite pool — providing an email gets faster rate limits
pyalex.config.email = os.getenv("OPENALEX_EMAIL", "sciverify@example.com")


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL (LAZY-LOADED)
# ══════════════════════════════════════════════════════════════════════════════
# The sentence-transformer model is loaded once and reused across all requests.
# all-MiniLM-L6-v2 is small (~90MB) and fast — good for real-time FAISS indexing.

_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model (singleton pattern)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully.")
    return _embedding_model


# ══════════════════════════════════════════════════════════════════════════════
# OPENALEX HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex's inverted index format.
    
    OpenAlex stores abstracts as {word: [position1, position2, ...]} to save space.
    This function converts it back to a readable string.
    """
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


def search_openalex(query: str, author_name: str = None, is_oa: bool = False, max_results: int = 5, concept_id: str = None) -> list[PaperSource]:
    """Search OpenAlex for papers matching the given query.
    
    Args:
        query: Search string (title, keywords, or topic)
        author_name: Optional author name to filter by (resolved to OpenAlex ID)
        is_oa: If True, only return Open Access papers
        max_results: Maximum number of results to return
        concept_id: Optional OpenAlex concept ID to filter by domain (e.g., "C41008148" for CS)
    
    Returns: List of PaperSource objects with metadata, abstract, and OA URL (if available)
    """
    logger.info(f"[RETRIEVAL] Searching OpenAlex: query='{query[:50]}...', author='{author_name}', max_results={max_results}, is_oa={is_oa}, concept_id='{concept_id}'")
    
    # Build filter dictionary
    query_filters = {}
    if is_oa:
        query_filters["is_oa"] = True
    if concept_id:
        query_filters["concepts.id"] = concept_id

    # Resolve author name to OpenAlex author ID (for precise filtering)
    if author_name:
        try:
            authors = Authors().search(author_name).get(per_page=1)
            if authors:
                author_id = authors[0]['id'].split("/")[-1]
                query_filters["author"] = {"id": author_id}
                logger.debug(f"[RETRIEVAL] Resolved author '{author_name}' to OpenAlex ID '{author_id}'")
            else:
                logger.warning(f"[RETRIEVAL] Could not resolve author '{author_name}', falling back to keyword search.")
        except Exception as e:
            logger.warning(f"[RETRIEVAL] Author resolution failed ({e}), continuing without author filter.")

    # Execute the search
    try:
        w = Works().search(query)
        if query_filters:
            w = w.filter(**query_filters)
        results = w.get(per_page=max_results)
    except Exception as e:
        logger.error(f"[RETRIEVAL] OpenAlex search FAILED: {e}")
        return []

    logger.info(f"[RETRIEVAL] OpenAlex returned {len(results)} results.")

    # Parse each result into a PaperSource object
    papers = []
    for i, work in enumerate(results):
        # Extract author names from the authorships array
        authors = []
        for authorship in work.get("authorships", []):
            name = authorship.get("author", {}).get("display_name", "Unknown")
            authors.append(name)

        # Reconstruct abstract from inverted index (OpenAlex's storage format)
        abstract = ""
        if work.get("abstract_inverted_index"):
            abstract = reconstruct_abstract(work["abstract_inverted_index"])
        elif work.get("abstract"):
            abstract = work["abstract"]

        # Check for Open Access PDF availability
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


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESSIVE RELAXATION SEARCH (for Hallucination Check workflow)
# ══════════════════════════════════════════════════════════════════════════════
# Cascading 4-level search strategy to handle AI-hallucinated citation metadata.
# Each level forgives one more piece of metadata:
#
#   Level 1: Title + Author + Year     (exact match — most precise)
#   Level 2: Title + Author            (year forgiveness — date hallucinated)
#   Level 3: Title only                (author forgiveness — author hallucinated)
#     Level 3.5: Fuzzy substring rescue (handles truncated/paraphrased titles)
#   Level 4: Author + Topic + Domain   (title forgiveness — title hallucinated)

def verify_openalex_citation(target: CitationTarget) -> tuple[Optional[PaperSource], str, Optional[MetadataDelta]]:
    """Implements Progressive Relaxation Search to robustly find a citation.
    
    Returns:
        (matched_paper, status_reasoning, metadata_delta)
        or (None, "0 hits found...", None) if all levels fail
    """
    title = target.title or ""
    year = target.year
    author_name = target.authors[0] if target.authors else None
    domain_lower = target.domain.lower() if target.domain else ""
    concept_id = DOMAIN_MAP.get(domain_lower)
    
    logger.info(f"[CASCADING SEARCH] Target: Title='{title}', Author='{author_name}', Year={year}, Domain='{domain_lower}' Concept='{concept_id}'")

    def query_with_author(q_title, q_author, q_year, max_res=3, q_concept=None):
        """Helper: search OpenAlex with author filter."""
        return search_openalex(query=q_title, author_name=q_author, max_results=max_res, concept_id=q_concept)

    # ── Level 1: Strict match (Title + Author + Year) ──
    if title and author_name and year:
        # OpenAlex doesn't support exact year filtering via PyAlex easily,
        # so we fetch results and filter by year in Python
        results = query_with_author(title, author_name, None, max_res=10)
        for r in results:
            if r.publication_date and str(year) in r.publication_date:
                return r, "Matched exactly by Title, Author, and Year.", generate_delta(target, r)

    # ── Level 2: Date Forgiveness (Title + Author, ignore year) ──
    if title and author_name:
        results = query_with_author(title, author_name, None, max_res=1)
        if results:
            return results[0], "Matched by Title and Author (Date hallucinated).", generate_delta(target, results[0])

    # ── Level 3: Author Forgiveness (Title only, strict fuzzy > 85%) ──
    if title:
        results = search_openalex(query=title, author_name=None, max_results=2)
        for r in results:
            if fuzz.ratio(title.lower(), r.title.lower()) > 85:
                return r, "Matched strictly by Title (Author hallucinated).", generate_delta(target, r)
                
        # ── Level 3.5: Fuzzy Substring Rescue ──
        # Handles cases where the claimed title is truncated or slightly reworded
        for r in results:
            tit_ratio = fuzz.partial_ratio(title.lower(), r.title.lower())
            abs_ratio = fuzz.partial_ratio(title.lower(), r.abstract.lower()) if r.abstract else 0
            if tit_ratio > 80 or abs_ratio > 80:
                return r, "Matched by partial substring fuzzy rescue on Title/Abstract.", generate_delta(target, r)

    # ── Level 4: Title Forgiveness — The Abbreviation Rescue ──
    # When the title is completely hallucinated, try Author + Topic + Domain concept lock
    if author_name and year and target.core_topic:
        logger.info(f"[CASCADING SEARCH] Falling back to Author={author_name} + Topic='{target.core_topic}' with Concept={concept_id}")
        results = query_with_author(target.core_topic, author_name, None, max_res=10, q_concept=concept_id)
        for r in results:
            year_match = r.publication_date and str(year) in r.publication_date
            tit_ratio = fuzz.partial_ratio(target.core_topic.lower(), r.title.lower())
            abs_ratio = fuzz.partial_ratio(target.core_topic.lower(), r.abstract.lower()) if r.abstract else 0
            
            if year_match or tit_ratio > 80 or abs_ratio > 80:
                return r, f"Matched via domain-locked Author + Topic (Fuzzy rescue/Year match, Title '{title}' hallucinated into alias).", generate_delta(target, r)

    # All 4 levels failed — paper is likely hallucinated
    return None, "0 hits found in cascading OpenAlex registry search.", None


# ══════════════════════════════════════════════════════════════════════════════
# FULL-TEXT EXTRACTION (PDF → Markdown)
# ══════════════════════════════════════════════════════════════════════════════
# Attempts to download and parse an Open Access PDF into markdown text.
# Returns None on any failure — the caller falls back to abstract.

def try_extract_full_text(paper: PaperSource) -> Optional[str]:
    """Attempt to download and parse full text from an OA PDF.
    
    Returns the extracted markdown text, or None on failure.
    Caller should fall back to paper.abstract when this returns None.
    """
    if not paper.oa_pdf_url:
        return None
    logger.info(f"[RETRIEVAL] Attempting full-text extraction: {paper.oa_pdf_url}")
    try:
        import pymupdf4llm

        # Download the PDF (15-second timeout)
        response = requests.get(paper.oa_pdf_url, timeout=15)
        if response.status_code != 200:
            logger.warning(f"[RETRIEVAL] PDF download failed: HTTP {response.status_code}")
            return None

        # Write to a temp file for pymupdf4llm to parse
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Convert PDF → markdown (preserves structure better than raw text)
        md_text = pymupdf4llm.to_markdown(tmp_path)
        os.unlink(tmp_path)  # Clean up temp file

        # Sanity check: if extracted text is too short, it's probably garbage
        if len(md_text) < 100:
            logger.warning(f"[RETRIEVAL] Extracted text too short ({len(md_text)} chars), falling back to abstract.")
            return None
        logger.info(f"[RETRIEVAL] Full-text extracted: {len(md_text)} chars.")
        return md_text
    except Exception as e:
        logger.warning(f"[RETRIEVAL] Full-text extraction FAILED for {paper.doi}: {e}")
        return None


def get_paper_text(paper: PaperSource, attempt_full_text: bool = False) -> str:
    """Get the best available text for a paper (Abstract-First strategy).
    
    Priority:
      1. Full-text PDF (if attempt_full_text=True and OA URL exists)
      2. Abstract (always available from OpenAlex)
      3. Empty string (if no abstract either)
    """
    if attempt_full_text and paper.oa_pdf_url:
        full_text = try_extract_full_text(paper)
        if full_text:
            paper.full_text = full_text
            return full_text
    # Fallback: use abstract (works for both OA and paywalled papers)
    return paper.abstract or ""


# ══════════════════════════════════════════════════════════════════════════════
# FAISS INDEXING & RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════
# Chunks paper text into overlapping windows, embeds them with sentence-transformers,
# and builds a FAISS inner-product index for fast semantic similarity search.

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 150) -> list[str]:
    """Split text into overlapping character-level chunks.
    
    Overlapping ensures that sentences spanning chunk boundaries are captured.
    Default: 500-char chunks with 150-char overlap → 350-char step size.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def build_snippet_index(
    papers: list[PaperSource],
    attempt_full_text: bool = False,
) -> tuple:
    """Build a FAISS index from paper texts (full-text or abstract fallback).
    
    For each paper:
      1. Get the best available text (PDF → abstract → empty)
      2. Chunk it into overlapping windows
      3. Embed each chunk with sentence-transformers
      4. Add to FAISS inner-product index (cosine similarity after L2 normalization)
    
    Returns: (faiss_index, snippets_list, source_id_list)
    """
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

    # Handle edge case: no text available for any paper
    if not all_snippets:
        logger.warning("[FAISS] No snippets to index — returning empty index.")
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        return index, [], []

    # Encode all chunks and build the FAISS index
    logger.info(f"[FAISS] Encoding {len(all_snippets)} snippets...")
    embeddings = model.encode(all_snippets, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim after normalization
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
    """Retrieve top-k relevant snippets for a claim using FAISS semantic search.
    
    Args:
        claim_text: The claim or quote to find support for
        index: Pre-built FAISS index
        all_snippets: The original chunk texts (parallel with the index)
        snippet_source_ids: Source paper ID for each chunk
        top_k: Number of top results to return
    
    Returns: List of (snippet_text, source_id, similarity_score)
    """
    if not all_snippets or index.ntotal == 0:
        logger.warning(f"[FAISS] Empty index — cannot retrieve for claim: '{claim_text[:50]}...'")
        return []

    # Encode the claim and search
    model = get_embedding_model()
    query_emb = model.encode([claim_text], show_progress_bar=False)
    query_emb = np.array(query_emb, dtype="float32")
    faiss.normalize_L2(query_emb)

    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_emb, k)

    results = []
    for j, idx in enumerate(indices[0]):
        if idx < 0:
            continue  # FAISS returns -1 for missing results
        results.append(
            (all_snippets[idx], snippet_source_ids[idx], float(distances[0][j]))
        )
    logger.info(
        f"[FAISS] Retrieved {len(results)} snippets for claim: '{claim_text[:50]}...' | "
        f"Scores: {[f'{r[2]:.3f}' for r in results]}"
    )
    return results
