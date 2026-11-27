from typing import Literal

from pydantic import BaseModel


class BtwRouteDecision(BaseModel):
    needs_web_search: bool


class RouterDecision(BaseModel):
    route: Literal["retrieve", "verify_claim", "direct_answer"]


class RelevancyDecision(BaseModel):
    is_relevant: bool
    reason: str


class SupersedingPaper(BaseModel):
    title: str
    url: str
    summary: str


class ClaimVerificationResult(BaseModel):
    is_superseded: bool
    verdict_summary: str
    superseding_papers: list[SupersedingPaper]


SemanticCategory = Literal["Concept", "Method", "Dataset", "Metric", "Finding"]


class ExtractedEntity(BaseModel):
    name: str
    entity_type: SemanticCategory
    description: str
    evidence_quote: str
    chunk_indices: list[int]


class ExtractedRelation(BaseModel):
    source: str
    target: str
    relation: str
    description: str
    evidence_quote: str
    chunk_index: int


class ChunkBatchGraph(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


class EvidenceItem(BaseModel):
    paper_title: str
    section_or_page: str
    content_type: Literal["Published Material", "Retrieved Evidence", "Extracted Information"]
    statement: str
    source_quote: str


class StructuredTable(BaseModel):
    title: str
    headers: list[str]
    rows: list[list[str]]


class ResearchToolOutput(BaseModel):
    tool_id: str
    tool_title: str
    evidence_tier: str
    verified_evidence: list[EvidenceItem]
    structured_tables: list[StructuredTable]
    ai_generated_analysis: str
    ai_generated_ideas_or_drafts: str | None = None
    grounding_notes: list[str] = []
