"""
Sci-Verify Data Models
=======================
Pydantic models for structured outputs, provenance tracking, and verification results.

Hierarchy:
  CitationTarget      — What the AI claims (title, author, year, topic, domain)
  ExtractedCitation    — A single claim/quote from the user's text + its CitationTarget
  PaperSource          — A real paper found in OpenAlex (with full metadata)
  MetadataDelta        — Claimed vs Found comparison (for the provenance UI)
  VerificationResult   — Final verdict for one claim (status + reasoning + matched paper)
  VerificationReport   — Collection of all results (used internally)
  GeneratorOutput      — Structured LLM output for the Deep Search synthesis
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime, timezone


# ══════════════════════════════════════════════════════════════════════════════
# CITATION TARGET (what the AI text claims)
# ══════════════════════════════════════════════════════════════════════════════

class CitationTarget(BaseModel):
    """Metadata representing the target citation extracted from user text.
    
    The Decomposer LLM fills this in. Fields may be null if the AI text
    doesn't explicitly state them (e.g., only mentions "the Raft paper"
    without giving the full title).
    """
    title: Optional[str] = Field(default=None, description="Exact full paper title if explicitly stated, otherwise null")
    authors: list[str] = Field(default_factory=list, description="Extracted authors")
    year: Optional[int] = Field(default=None, description="Extracted year of publication")
    core_topic: str = Field(description="2-5 word keyword capturing the specific mechanism or algorithm for fallback searches")
    domain: Optional[str] = Field(default=None, description="Academic domain e.g. 'computer science', 'biology', 'medicine', 'physics'")


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTED CITATION (a claim from the user's text)
# ══════════════════════════════════════════════════════════════════════════════

class ExtractedCitation(BaseModel):
    """A single sentence or quote from the user's input text, paired with its citation target.
    
    The 'is_explicit_quote' flag determines which verification engine is used:
      - True  → Engine A (fuzzy string matching against paper text)
      - False → Engine B (LLM NLI semantic evaluation)
    """
    claim_id: int = Field(description="Unique identifier")
    text: str = Field(description="The exact text of the claim or quote")
    is_explicit_quote: bool = Field(description="True if the text was enclosed in literal quotation marks")
    target_metadata: CitationTarget = Field(description="The parsed citation it refers to")


class DecomposedCitations(BaseModel):
    """Output wrapper for the Decomposer node."""
    citations: list[ExtractedCitation] = Field(default_factory=list, description="List of all extracted citations. Can be empty if none found.")


# ══════════════════════════════════════════════════════════════════════════════
# PAPER SOURCE (a real paper found in OpenAlex)
# ══════════════════════════════════════════════════════════════════════════════

class PaperSource(BaseModel):
    """A retrieved scientific paper with full provenance metadata.
    
    Populated by search_openalex() in retrieval.py.
    The 'full_text' field is set to "Retrieved" if PDF extraction succeeded,
    or None if only the abstract is available.
    """
    source_id: int = Field(description="Unique identifier for this source")
    title: str = Field(description="Paper title")
    doi: str = Field(description="Versioned DOI")
    authors: list[str] = Field(default_factory=list)
    abstract: str = Field(default="")
    full_text: Optional[str] = Field(default=None)
    oa_pdf_url: Optional[str] = Field(default=None)
    publication_date: Optional[str] = Field(default=None)
    openalex_id: Optional[str] = Field(default=None)
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp — provenance chain",
    )
    relevance_score: Optional[int] = Field(default=None, description="LLM relevance score out of 100")


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION STATUS (final verdict for each claim)
# ══════════════════════════════════════════════════════════════════════════════

class VerificationStatus(str, Enum):
    """The final status assigned to each citation after verification.
    
    Green ✅ statuses (claim is supported):
      VERIFIED_QUOTE    — Explicit quote found in paper via fuzzy matching (≥85%)
      SUPPORTED_SUMMARY — Semantic claim supported by NLI evaluation
    
    Red ❌ statuses (claim is not supported):
      HALLUCINATED_QUOTE   — Explicit quote NOT found in the paper's full text
      UNSUPPORTED_SUMMARY  — Semantic claim NOT supported by any chunk
      HALLUCINATED_PAPER   — The cited paper doesn't exist in OpenAlex at all
    
    Yellow ⚠️ status (inconclusive):
      UNKNOWN_PAYWALLED — Paper exists but is paywalled; abstract-only check was insufficient
    """
    VERIFIED_QUOTE = "Verified Exact Quote ✅"
    SUPPORTED_SUMMARY = "Supported Semantic Claim ✅"
    HALLUCINATED_QUOTE = "Hallucinated Quote ❌"
    UNSUPPORTED_SUMMARY = "Unsupported Claim ❌"
    HALLUCINATED_PAPER = "Hallucinated Paper ❌"
    UNKNOWN_PAYWALLED = "Unknown / Paywalled ⚠️"


# ══════════════════════════════════════════════════════════════════════════════
# METADATA DELTA (claimed vs found comparison)
# ══════════════════════════════════════════════════════════════════════════════

class MetadataDelta(BaseModel):
    """Tracks discrepancies between claimed and found metadata.
    
    Displayed in the UI as a side-by-side "Claimed Metadata" vs "Found in OpenAlex"
    comparison, with warning badges for mismatches.
    """
    claimed_title: str = ""
    found_title: str = ""
    claimed_year: Optional[int] = None
    found_year: Optional[int] = None
    is_year_mismatch: bool = False      # True if claimed year ≠ found year
    is_title_alias: bool = False        # True if title fuzzy ratio < 60%


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION RESULT (single claim verdict)
# ══════════════════════════════════════════════════════════════════════════════

class VerificationResult(BaseModel):
    """The complete result of verifying a single ExtractedCitation.
    
    Contains the final status, the matched paper (if found), similarity score
    (for quotes), auditor reasoning, and the metadata delta for the UI.
    """
    claim_id: int
    status: VerificationStatus
    matched_paper: Optional[PaperSource] = None
    similarity_score: float = Field(default=0.0, description="String similarity out of 100")
    reasoning: str = Field(default="", description="Brief explanation")
    metadata_delta: Optional[MetadataDelta] = Field(default=None, description="Claimed vs Found comparison for provenance UI")


class VerificationReport(BaseModel):
    """The complete Verification Matrix (all citations + all results)."""
    citations: list[ExtractedCitation] = Field(default_factory=list)
    results: list[VerificationResult] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# GENERATOR OUTPUT (Deep Search synthesis)
# ══════════════════════════════════════════════════════════════════════════════

class GeneratorOutput(BaseModel):
    """Structured output for the Generator node (Deep Search workflow).
    
    Not currently used as a direct Pydantic parse target — the LLM output
    is parsed as raw JSON instead. Kept for potential future structured output.
    """
    draft: str = Field(description="The synthesized research draft")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence; low = 'I don't know'",
    )
    cited_dois: list[str] = Field(
        default_factory=list,
        description="DOIs explicitly cited in the draft",
    )
