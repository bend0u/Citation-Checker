"""
Sci-Verify Data Models
Pydantic models for structured outputs, provenance tracking,
and the Factual Support Matrix.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime, timezone


class CitationTarget(BaseModel):
    """Metadata representing the target citation of a sentence or quote."""
    title: Optional[str] = Field(default=None, description="Inferred or literal title of the paper")
    authors: list[str] = Field(default_factory=list, description="Extracted authors")
    year: Optional[int] = Field(default=None, description="Extracted year of publication")
    core_topic: str = Field(description="Max 3 word minimalist keyword summarizing the core topic for fallback searches")

class ExtractedCitation(BaseModel):
    """A single sentence or quote and its target citation metadata."""
    claim_id: int = Field(description="Unique identifier")
    text: str = Field(description="The exact text of the claim or quote")
    is_explicit_quote: bool = Field(description="True if the text was enclosed in literal quotation marks")
    target_metadata: CitationTarget = Field(description="The parsed citation it refers to")

class DecomposedCitations(BaseModel):
    """Output wrapper for the Decomposer node."""
    citations: list[ExtractedCitation] = Field(default_factory=list, description="List of all extracted citations. Can be empty if none found.")


class PaperSource(BaseModel):
    """A retrieved scientific paper with full provenance metadata."""
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
        description="ISO timestamp – provenance chain",
    )
    relevance_score: Optional[int] = Field(default=None, description="LLM relevance score out of 100")


class VerificationStatus(str, Enum):
    """The final status of the verification engine."""
    VERIFIED_QUOTE = "Verified Exact Quote ✅"
    SUPPORTED_SUMMARY = "Supported Semantic Claim ✅"
    HALLUCINATED_QUOTE = "Hallucinated Quote ❌"
    UNSUPPORTED_SUMMARY = "Unsupported Claim ❌"
    HALLUCINATED_PAPER = "Hallucinated Paper ❌"
    UNKNOWN_PAYWALLED = "Unknown / Paywalled ⚠️"

class VerificationResult(BaseModel):
    """The result of verifying an ExtractedCitation."""
    claim_id: int
    status: VerificationStatus
    matched_paper: Optional[PaperSource] = None
    similarity_score: float = Field(default=0.0, description="String similarity out of 100")
    reasoning: str = Field(default="", description="Brief explanation")

class VerificationReport(BaseModel):
    """The complete Verification Matrix."""
    citations: list[ExtractedCitation] = Field(default_factory=list)
    results: list[VerificationResult] = Field(default_factory=list)


class GeneratorOutput(BaseModel):
    """Structured output for the Generator node."""
    draft: str = Field(description="The synthesized research draft")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence; low = 'I don't know'",
    )
    cited_dois: list[str] = Field(
        default_factory=list,
        description="DOIs explicitly cited in the draft",
    )
