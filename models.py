"""
Sci-Verify Data Models
Pydantic models for structured outputs, provenance tracking,
and the Factual Support Matrix.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime, timezone


class NLILabel(str, Enum):
    """Natural Language Inference labels for claim verification."""
    ENTAILMENT = "Entailment"
    NEUTRAL = "Neutral"
    CONTRADICTION = "Contradiction"


class AtomicClaim(BaseModel):
    """A single, verifiable scientific claim extracted from the draft."""
    claim_id: int = Field(description="Unique identifier for this claim")
    text: str = Field(description="The atomic claim text")
    cited_doi: Optional[str] = Field(
        default=None,
        description="DOI cited by the generator for this claim, if any",
    )


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


class NLIResult(BaseModel):
    """Result of an NLI evaluation between a claim and a source."""
    label: NLILabel = Field(description="Entailment, Neutral, or Contradiction")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    source_quote: str = Field(
        default="", description="Direct quote from the source"
    )
    reasoning: str = Field(default="", description="Brief reasoning")


class SupportMatrixEntry(BaseModel):
    """A single cell in the Factual Support Matrix."""
    claim_id: int
    source_id: int
    result: NLIResult


class FactualSupportMatrix(BaseModel):
    """The complete Factual Support Matrix with computed metrics."""
    claims: list[AtomicClaim] = Field(default_factory=list)
    sources: list[PaperSource] = Field(default_factory=list)
    entries: list[SupportMatrixEntry] = Field(default_factory=list)
    citation_accuracy: float = Field(default=0.0)
    citation_thoroughness: float = Field(default=0.0)


class DecomposedClaims(BaseModel):
    """Structured output for GPT-4o-mini atomic decomposition."""
    claims: list[str] = Field(description="List of atomic scientific claims")


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
