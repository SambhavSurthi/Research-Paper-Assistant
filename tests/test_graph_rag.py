from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from backend.graph_store import (
    add_paper_to_graph,
    graph_retrieve,
    run_readonly_cypher,
)
from backend.graph_visualizer import build_graph_html
from backend.models import (
    ChunkBatchGraph,
    EvidenceItem,
    ExtractedEntity,
    ExtractedRelation,
    ResearchToolOutput,
    StructuredTable,
)
from backend.paper_loader import load_markdown, load_pdf, restamp_title
from backend.research_tools import (
    RESEARCH_TOOLS,
    extract_pdf_full_context,
    format_tool_output_markdown,
    run_research_tool,
)


def test_markdown_hierarchy_extraction(tmp_path):
    md_file = tmp_path / "sample_paper.md"
    md_file.write_text(
        "# Abstract\nThis paper introduces GraphRAG.\n\n"
        "# 1. Methodology\nHigh-level overview.\n\n"
        "## 1.1 Knowledge Graph\nEntities and relations are stored in Neo4j.\n\n"
        "### 1.1.1 Embeddings\nChunks carry 1536-dim vectors.\n",
        encoding="utf-8",
    )
    docs = load_markdown(str(md_file))
    assert len(docs) == 4
    paths = [d.metadata["section_path"] for d in docs]
    assert paths[0] == "Abstract"
    assert paths[1] == "1. Methodology"
    assert paths[2] == "1. Methodology > 1.1 Knowledge Graph"
    assert paths[3] == "1. Methodology > 1.1 Knowledge Graph > 1.1.1 Embeddings"
    assert docs[2].metadata["parent_section_id"] == docs[1].metadata["section_id"]
    assert [d.metadata["chunk_index"] for d in docs] == [0, 1, 2, 3]


def test_pdf_hierarchy_and_restamp():
    docs = load_pdf("documents/Openclaw_Research_Report.pdf")
    assert len(docs) > 20
    assert all("section_path" in d.metadata and "chunk_id" in d.metadata for d in docs)
    old_pid = docs[0].metadata["paper_id"]
    restamp_title(docs, "Renamed_Report")
    assert docs[0].metadata["title"] == "Renamed_Report"
    assert docs[0].metadata["paper_id"] != old_pid
    assert docs[0].metadata["chunk_id"].startswith(docs[0].metadata["paper_id"])


def test_readonly_cypher_blocks_writes():
    for bad in ("CREATE (n:Test)", "MATCH (n) DELETE n", "MERGE (n:X {id:1})", "MATCH (n) SET n.a=1"):
        try:
            run_readonly_cypher(bad)
            raise AssertionError(f"Expected ValueError for: {bad}")
        except ValueError:
            pass


def test_two_level_graph_ingest_and_hierarchical_retrieval_mocked():
    docs = [
        Document(
            page_content="Transformers rely on self-attention mechanisms.",
            metadata={
                "title": "Attention Paper",
                "source": "paper.pdf",
                "doc_type": "pdf",
                "paper_id": "p1",
                "section_id": "p1::sec_0",
                "section_title": "1. Introduction",
                "section_path": "1. Introduction",
                "section_level": 1,
                "section_order": 0,
                "parent_section_id": None,
                "chunk_id": "p1::chunk_0",
                "chunk_index": 0,
                "page": 1,
            },
        ),
        Document(
            page_content="Multi-head attention projects queries, keys, and values in parallel.",
            metadata={
                "title": "Attention Paper",
                "source": "paper.pdf",
                "doc_type": "pdf",
                "paper_id": "p1",
                "section_id": "p1::sec_1",
                "section_title": "2. Architecture",
                "section_path": "2. Architecture",
                "section_level": 1,
                "section_order": 1,
                "parent_section_id": None,
                "chunk_id": "p1::chunk_1",
                "chunk_index": 1,
                "page": 2,
            },
        ),
    ]

    fake_kg = ChunkBatchGraph(
        entities=[
            ExtractedEntity(
                name="Self-Attention",
                entity_type="Concept",
                description="Attention over sequence positions",
                evidence_quote="Transformers rely on self-attention mechanisms.",
                chunk_indices=[0],
            ),
            ExtractedEntity(
                name="Multi-Head Attention",
                entity_type="Method",
                description="Parallel attention heads",
                evidence_quote="Multi-head attention projects queries, keys, and values in parallel.",
                chunk_indices=[1],
            ),
        ],
        relations=[
            ExtractedRelation(
                source="Multi-Head Attention",
                target="Self-Attention",
                relation="EXTENDS",
                description="Runs multiple self-attention heads in parallel",
                evidence_quote="Multi-head attention projects queries, keys, and values in parallel.",
                chunk_index=1,
            )
        ],
    )

    cypher_calls: list[str] = []

    def fake_run_cypher(stmt, params=None):
        cypher_calls.append(stmt.strip())
        if "c.embedding IS NOT NULL" in stmt and "MATCH (c:Chunk" in stmt:
            return [
                {
                    "id": "s1::p1::chunk_0",
                    "paper_id": "s1::p1",
                    "chunk_index": 0,
                    "embedding": [1.0] + [0.0] * 1535,
                },
                {
                    "id": "s1::p1::chunk_1",
                    "paper_id": "s1::p1",
                    "chunk_index": 1,
                    "embedding": [0.9, 0.4] + [0.0] * 1534,
                },
            ]
        return []

    with (
        patch("backend.graph_store.ensure_schema"),
        patch("backend.graph_store.run_cypher", side_effect=fake_run_cypher),
        patch("backend.graph_store.embeddings.embed_documents", return_value=[[1.0] + [0.0] * 1535, [0.9, 0.4] + [0.0] * 1534]),
        patch("backend.graph_store._extract_batch_kg", return_value=fake_kg),
    ):
        stats = add_paper_to_graph(docs, session_id="s1", extract_kg=True)
        assert stats["sections"] == 2
        assert stats["chunks"] == 2
        assert stats["entities"] == 2
        assert stats["relations"] == 1
        assert any("MERGE (sec)-[:HAS_CHUNK]->(c)" in s for s in cypher_calls)
        assert any("MERGE (a)-[:NEXT_CHUNK]->(b)" in s for s in cypher_calls)
        assert any("MERGE (c)-[m:MENTIONS]->(e)" in s for s in cypher_calls)
        assert any("MERGE (e)-[prov:PROVENANCE_IN" in s for s in cypher_calls)
        assert any("MERGE (e)-[fp:FROM_PAPER]->(p)" in s for s in cypher_calls)

    def fake_retrieve_cypher(stmt, params=None):
        if "db.index.vector.queryNodes" in stmt:
            return [{
                "id": "s1::p1::chunk_0",
                "text": "Transformers rely on self-attention mechanisms.",
                "title": "Attention Paper",
                "paper_id": "s1::p1",
                "section_id": "s1::p1::sec_0",
                "section_title": "1. Introduction",
                "section_path": "1. Introduction",
                "page": 1,
                "chunk_index": 0,
                "score": 0.95,
            }]
        if "MATCH (e:Entity" in stmt and "any(tok IN" in stmt:
            return [{"id": "s1::ent::self-attention", "name": "Self-Attention", "type": "Concept", "description": "Attention"}]
        if "MATCH (seed:Chunk" in stmt and "NEXT_CHUNK" in stmt:
            return [{
                "id": "s1::p1::chunk_1",
                "text": "Multi-head attention projects queries, keys, and values in parallel.",
                "title": "Attention Paper",
                "paper_id": "s1::p1",
                "section_id": "s1::p1::sec_1",
                "section_title": "2. Architecture",
                "section_path": "2. Architecture",
                "page": 2,
                "chunk_index": 1,
                "reason": "sequential_neighbor",
            }]
        if "MATCH (e1:Entity)" in stmt or "RELATED_TO" in stmt:
            return [{
                "source": "Multi-Head Attention",
                "source_type": "Method",
                "relation": "EXTENDS",
                "target": "Self-Attention",
                "target_type": "Concept",
                "description": "Parallel heads",
                "paper_title": "Attention Paper",
                "section_path": "2. Architecture",
                "page": 2,
                "evidence_quote": "Multi-head attention projects queries, keys, and values in parallel.",
            }]
        return []

    with (
        patch("backend.graph_store.is_neo4j_configured", return_value=True),
        patch("backend.graph_store.embeddings.embed_query", return_value=[1.0] + [0.0] * 1535),
        patch("backend.graph_store.run_cypher", side_effect=fake_retrieve_cypher),
    ):
        retrieved = graph_retrieve("How does self-attention work?", session_id="s1", k=2)
        reasons = [d.metadata["retrieval_reason"] for d in retrieved]
        assert "vector_seed" in reasons
        assert "sequential_neighbor" in reasons
        assert "knowledge_graph_triples" in reasons
        kg_doc = [d for d in retrieved if d.metadata["retrieval_reason"] == "knowledge_graph_triples"][0]
        assert "Provenance: Attention Paper | 2. Architecture | p.2" in kg_doc.page_content


def test_visualizer_html_generation():
    html = build_graph_html({
        "nodes": [
            {"id": "p1", "label": "Paper", "full_label": "Paper", "group": "Paper", "details": ""},
            {"id": "m1", "label": "GraphRAG", "full_label": "GraphRAG (Method)", "group": "Method", "details": "Method"},
        ],
        "edges": [{"from": "m1", "to": "p1", "label": "FROM_PAPER"}],
    })
    assert "<canvas id=\"graph\">" in html
    assert "FROM_PAPER" in html


def test_all_13_research_tools_and_pdf_execution():
    expected_ids = {
        "paper_summarization",
        "abstract_extraction",
        "methodology_explanation",
        "algorithm_equation_explanation",
        "dataset_metric_extraction",
        "results_analysis",
        "contribution_limitation_future_work",
        "research_gap_discovery",
        "paper_comparison",
        "cross_paper_synthesis",
        "contradiction_detection",
        "literature_review_draft",
        "research_question_generation",
    }
    assert set(RESEARCH_TOOLS.keys()) == expected_ids
    for spec in RESEARCH_TOOLS.values():
        assert "VERIFIED PAPER EVIDENCE" in spec.system_prompt
        assert "AI-GENERATED CONTENT" in spec.system_prompt

    pdf_bytes = Path("documents/Openclaw_Research_Report.pdf").read_bytes()
    payload = extract_pdf_full_context(pdf_bytes, "Openclaw_Research_Report.pdf")
    assert payload["num_pages"] > 0
    assert "Page 1" in payload["text"]
    assert len(payload["b64"]) > 100

    fake_out = ResearchToolOutput(
        tool_id="paper_summarization",
        tool_title="1. Paper Summarization",
        evidence_tier="Direct PDF",
        verified_evidence=[
            EvidenceItem(
                paper_title="Openclaw Research Report",
                section_or_page="Page 1",
                content_type="Published Material",
                statement="OpenClaw evolved through multiple branding stages.",
                source_quote="OpenClaw research report",
            )
        ],
        structured_tables=[
            StructuredTable(title="Overview", headers=["Dimension", "Finding"], rows=[["Scope", "Agentic system"]])
        ],
        ai_generated_analysis="**[AI Analysis]** Structured summary of the report.",
        ai_generated_ideas_or_drafts=None,
        grounding_notes=["Grounded strictly in uploaded PDF."],
    )

    with patch("backend.research_tools._TOOL_LLM") as mock_llm:
        mock_llm.invoke.return_value = fake_out
        res = run_research_tool("paper_summarization", pdf_payloads=[payload], attach_raw_pdf=True)
        assert mock_llm.invoke.called
        assert res.evidence_tier == "Direct PDF"
        md = format_tool_output_markdown(res)
        assert "Verified Paper Evidence" in md
        assert "[AI-Generated Analysis]" in md
