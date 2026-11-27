import os

from dotenv import load_dotenv

from backend.content_guard import ContentRejected, assess_research_document, scrub_text
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 1536  # text-embedding-3-small

# ── Singletons ────────────────────────────────────────────────────────────────

base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_file_store = LocalFileStore("./embedding_cache/")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)


# ── Collection ───────────────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    return f"ai_research_paper_assistant_{session_id.replace('-', '_')}"


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> dict:
    """Index paper chunks into Qdrant and (when configured) the Neo4j knowledge graph.

    Non-research text and instruction-override documents are refused before any
    embedding, vector write, or graph write.
    """
    if not docs:
        raise ContentRejected("Nothing to index.", "empty")
    kind = str(docs[0].metadata.get("doc_type") or "pdf")
    if kind not in {"pdf", "text", "markdown", "webpage", "arxiv"}:
        kind = "text"
    text = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
    verdict = assess_research_document(text, source_kind=kind)
    if not verdict.allowed:
        raise ContentRejected(verdict.reason, verdict.category)
    for doc in docs:
        doc.page_content = scrub_text(doc.page_content or "")

    qdrant_ok = False
    qdrant_err = None
    try:
        get_vectorstore(session_id).add_documents(docs)
        qdrant_ok = True
    except Exception as exc:
        qdrant_err = str(exc)

    graph_stats = None
    graph_err = None
    try:
        from backend.graph_store import add_paper_to_graph, is_neo4j_configured

        if is_neo4j_configured():
            graph_stats = add_paper_to_graph(docs, session_id)
    except Exception as exc:
        graph_err = str(exc)

    if not qdrant_ok and graph_stats is None:
        raise RuntimeError(f"Failed to index paper (Qdrant: {qdrant_err}; Neo4j: {graph_err})")

    return {
        "qdrant_chunks": len(docs) if qdrant_ok else 0,
        "qdrant_error": qdrant_err,
        "graph": graph_stats,
        "graph_error": graph_err,
    }


def list_papers(session_id: str) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()

    try:
        from backend.graph_store import is_neo4j_configured, list_papers_in_graph

        if is_neo4j_configured():
            for title in list_papers_in_graph(session_id):
                if title not in seen:
                    seen.add(title)
                    titles.append(title)
    except Exception:
        pass

    collection_name = get_collection_name(session_id)
    try:
        if qdrant_client.collection_exists(collection_name):
            offset = None
            while True:
                points, offset = qdrant_client.scroll(
                    collection_name=collection_name,
                    with_payload=True,
                    limit=100,
                    offset=offset,
                )
                for point in points:
                    title = (point.payload or {}).get("metadata", {}).get("title")
                    if title and title not in seen:
                        seen.add(title)
                        titles.append(title)
                if offset is None:
                    break
    except Exception:
        if not titles:
            raise

    return titles


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    return get_vectorstore(session_id).similarity_search(query, k=k)


def search_indexed(query: str, session_id: str, k: int = 4) -> list[Document]:
    """Search the shared corpus and the current session together."""
    from backend.session_store import GLOBAL_SESSION_ID

    docs: list[Document] = []
    seen: set[str] = set()
    for sid in (GLOBAL_SESSION_ID, session_id):
        if not sid:
            continue
        try:
            hits = search(query, sid, k=k)
        except Exception:
            hits = []
        for doc in hits:
            marker = doc.page_content[:240]
            if marker in seen:
                continue
            seen.add(marker)
            docs.append(doc)
    return docs[: max(k, 8)]


def _docs_from_neo4j(session_id: str) -> list[Document]:
    from backend.graph_store import is_neo4j_configured, run_cypher

    if not is_neo4j_configured():
        return []
    rows = run_cypher(
        """
        MATCH (c:Chunk {session_id: $session_id})
        RETURN c.title AS title,
               c.text AS text,
               c.chunk_index AS chunk_index,
               c.section_path AS section_path,
               c.page AS page,
               c.paper_id AS paper_id,
               c.id AS chunk_id
        ORDER BY c.paper_id ASC, c.chunk_index ASC
        """,
        {"session_id": session_id},
    )
    docs: list[Document] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "title": row.get("title") or "Untitled paper",
                    "chunk_index": row.get("chunk_index") if row.get("chunk_index") is not None else 0,
                    "section_path": row.get("section_path") or "",
                    "page": row.get("page"),
                    "paper_id": row.get("paper_id") or "",
                    "chunk_id": row.get("chunk_id") or "",
                    "source_store": "neo4j",
                },
            )
        )
    return docs


def _docs_from_qdrant(session_id: str) -> list[Document]:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return []

    docs: list[Document] = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=256,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            text = str(payload.get("page_content") or payload.get("text") or "").strip()
            meta = payload.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "title": meta.get("title") or payload.get("title") or "Untitled paper",
                        "chunk_index": meta.get("chunk_index") if meta.get("chunk_index") is not None else 0,
                        "section_path": meta.get("section_path") or "",
                        "page": meta.get("page"),
                        "paper_id": meta.get("paper_id") or "",
                        "chunk_id": meta.get("chunk_id") or str(point.id),
                        "source_store": "qdrant",
                    },
                )
            )
        if offset is None:
            break

    docs.sort(key=lambda d: (str(d.metadata.get("paper_id") or d.metadata.get("title")), int(d.metadata.get("chunk_index") or 0)))
    return docs


def _paper_key(doc: Document) -> str:
    title = str(doc.metadata.get("title") or "").strip().lower()
    if title:
        return title
    return str(doc.metadata.get("paper_id") or "paper")


def _merge_paper_copies(*groups: list[Document]) -> list[Document]:
    """Keep one complete copy of each paper and return chunks in reading order."""
    by_paper: dict[str, list[Document]] = {}
    order: list[str] = []
    for docs in groups:
        for doc in docs:
            key = _paper_key(doc)
            if key not in by_paper:
                by_paper[key] = []
                order.append(key)
            by_paper[key].append(doc)

    merged: list[Document] = []
    for key in order:
        copies: dict[str, list[Document]] = {}
        for doc in by_paper[key]:
            store = str(doc.metadata.get("source_store") or "store")
            copies.setdefault(store, []).append(doc)
        chosen = max(copies.values(), key=lambda docs: sum(len(d.page_content) for d in docs))
        chosen.sort(key=lambda d: int(d.metadata.get("chunk_index") or 0))
        deduped: list[Document] = []
        seen_chunks: set[tuple] = set()
        for doc in chosen:
            marker = (int(doc.metadata.get("chunk_index") or 0), doc.page_content[:160])
            if marker in seen_chunks:
                continue
            seen_chunks.add(marker)
            deduped.append(doc)
        merged.extend(deduped)
    return merged


def load_session_corpus(session_id: str) -> list[Document]:
    """Return every indexed chunk for the session, in paper reading order."""
    neo_docs: list[Document] = []
    qdrant_docs: list[Document] = []
    try:
        neo_docs = _docs_from_neo4j(session_id)
    except Exception:
        neo_docs = []
    try:
        qdrant_docs = _docs_from_qdrant(session_id)
    except Exception:
        qdrant_docs = []
    return _merge_paper_copies(neo_docs, qdrant_docs)


def load_any_indexed_papers() -> list[Document]:
    """Return the complete text of every paper indexed in any session."""
    neo_docs: list[Document] = []
    try:
        from backend.graph_store import is_neo4j_configured, run_cypher

        if is_neo4j_configured():
            rows = run_cypher(
                """
                MATCH (c:Chunk)
                WHERE c.text IS NOT NULL AND trim(c.text) <> ''
                RETURN c.title AS title,
                       c.text AS text,
                       c.chunk_index AS chunk_index,
                       c.section_path AS section_path,
                       c.page AS page,
                       c.paper_id AS paper_id,
                       c.id AS chunk_id,
                       c.session_id AS session_id
                ORDER BY c.paper_id ASC, c.chunk_index ASC
                """
            )
            for row in rows:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                neo_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "title": row.get("title") or "Untitled paper",
                            "chunk_index": row.get("chunk_index") if row.get("chunk_index") is not None else 0,
                            "section_path": row.get("section_path") or "",
                            "page": row.get("page"),
                            "paper_id": row.get("paper_id") or "",
                            "chunk_id": row.get("chunk_id") or "",
                            "source_store": "neo4j",
                        },
                    )
                )
    except Exception:
        neo_docs = []

    qdrant_docs: list[Document] = []
    try:
        for collection in qdrant_client.get_collections().collections:
            name = collection.name
            if not name.startswith("ai_research_paper_assistant_"):
                continue
            offset = None
            while True:
                points, offset = qdrant_client.scroll(
                    collection_name=name,
                    with_payload=True,
                    limit=256,
                    offset=offset,
                )
                for point in points:
                    payload = point.payload or {}
                    text = str(payload.get("page_content") or payload.get("text") or "").strip()
                    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                    if not text:
                        continue
                    qdrant_docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "title": meta.get("title") or payload.get("title") or "Untitled paper",
                                "chunk_index": meta.get("chunk_index") if meta.get("chunk_index") is not None else 0,
                                "section_path": meta.get("section_path") or "",
                                "page": meta.get("page"),
                                "paper_id": meta.get("paper_id") or "",
                                "chunk_id": meta.get("chunk_id") or str(point.id),
                                "source_store": "qdrant",
                            },
                        )
                    )
                if offset is None:
                    break
    except Exception:
        qdrant_docs = []

    return _merge_paper_copies(neo_docs, qdrant_docs)


def papers_for_answer(session_id: str) -> list[Document]:
    """Complete text of the shared corpus plus this session, otherwise every indexed paper."""
    from backend.session_store import GLOBAL_SESSION_ID

    groups: list[list[Document]] = []
    for sid in (GLOBAL_SESSION_ID, session_id):
        if not sid:
            continue
        try:
            found = load_session_corpus(sid)
        except Exception:
            found = []
        if found:
            groups.append(found)
    if groups:
        return _merge_paper_copies(*groups)
    try:
        return load_any_indexed_papers()
    except Exception:
        return []


def format_corpus(docs: list[Document], max_chars: int = 350_000) -> str:
    """Join chunks back into complete papers, capped so the prompt stays inside the model window."""
    if not docs:
        return ""

    papers: dict[str, list[Document]] = {}
    order: list[str] = []
    for doc in docs:
        title = str(doc.metadata.get("title") or "Untitled paper")
        if title not in papers:
            papers[title] = []
            order.append(title)
        papers[title].append(doc)

    parts: list[str] = []
    used = 0
    for title in order:
        chunks = sorted(papers[title], key=lambda d: int(d.metadata.get("chunk_index") or 0))
        body_bits = []
        for chunk in chunks:
            section = chunk.metadata.get("section_path") or ""
            page = chunk.metadata.get("page")
            loc = ""
            if section or page is not None:
                page_bit = f"p.{page}" if page is not None else ""
                loc = " ".join(bit for bit in (section, page_bit) if bit)
                loc = f"[{loc}]\n"
            body_bits.append(f"{loc}{scrub_text(chunk.page_content)}")
        block = f"===== PAPER: {title} =====\n" + "\n\n".join(body_bits)
        if used + len(block) > max_chars:
            remain = max_chars - used
            if remain > 800:
                parts.append(block[:remain] + "\n\n[...remainder of the corpus truncated for context length...]")
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)
