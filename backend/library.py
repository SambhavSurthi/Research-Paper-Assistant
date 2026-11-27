"""Index saved session sources into a session graph and the shared chat corpus."""

from __future__ import annotations

from langchain_core.documents import Document

from backend.content_guard import ContentRejected, assess_research_document, scrub_text
from backend.paper_loader import load_arxiv, load_document, load_text, load_webpage, restamp_title
from backend.session_store import (
    GLOBAL_SESSION_ID,
    attach_snapshot,
    iter_saved_sources,
    snapshot_file,
    source_file,
)
from backend.vector_store import add_paper, list_papers, load_session_corpus


def documents_for_source(source: dict) -> list[Document]:
    """Load one saved PDF, text file, web page, or arXiv paper."""
    kind = source.get("kind")
    label = source.get("label") or "Untitled"
    stored = source_file(source)
    snapshot = snapshot_file(source)

    if kind in ("pdf", "text") and stored is not None:
        docs = load_document(str(stored))
        restamp_title(docs, label)
        return guard_documents(docs, kind)

    if kind == "webpage" and source.get("url"):
        try:
            return guard_documents(load_webpage(source["url"]), "webpage")
        except ContentRejected:
            raise
        except Exception:
            if snapshot is not None:
                docs = load_text(str(snapshot))
                restamp_title(docs, label)
                return guard_documents(docs, "webpage")
            raise

    if kind == "arxiv" and source.get("query"):
        try:
            return guard_documents(load_arxiv(source["query"]), "arxiv")
        except ContentRejected:
            raise
        except Exception:
            if snapshot is not None:
                docs = load_text(str(snapshot))
                restamp_title(docs, label)
                return guard_documents(docs, "arxiv")
            raise

    if snapshot is not None:
        docs = load_text(str(snapshot))
        restamp_title(docs, label)
        return guard_documents(docs, kind or "text")
    raise ValueError(f"No stored file for {label}")


def _title_of(docs: list[Document], fallback: str) -> str:
    if docs:
        title = docs[0].metadata.get("title")
        if title:
            return str(title)
    return fallback


def _remember_text(session_id: str, source: dict, docs: list[Document]) -> None:
    if source.get("snapshot_path"):
        return
    text = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
    if text.strip():
        attach_snapshot(session_id, source["id"], text)


def guard_documents(docs: list[Document], source_kind: str) -> list[Document]:
    """Refuse non-research text. Strip instruction-override spans before indexing."""
    text = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
    verdict = assess_research_document(text, source_kind=source_kind or "pdf")
    if not verdict.allowed:
        raise ContentRejected(verdict.reason, verdict.category)
    for doc in docs:
        doc.page_content = scrub_text(doc.page_content or "")
    return docs


def index_documents(docs: list[Document], session_id: str, *, also_global: bool = True) -> dict:
    kind = str(docs[0].metadata.get("doc_type") or "pdf") if docs else "pdf"
    docs = guard_documents(docs, kind)
    result = add_paper(docs, session_id)
    if also_global and session_id != GLOBAL_SESSION_ID:
        try:
            result["global"] = add_paper(docs, GLOBAL_SESSION_ID)
        except Exception as exc:
            result["global_error"] = str(exc)
    return result


def index_saved_source(session_id: str, source: dict, *, also_global: bool = True) -> dict:
    docs = documents_for_source(source)
    _remember_text(session_id, source, docs)
    return index_documents(docs, session_id, also_global=also_global)


def index_all_sources(session_ids: list[str]) -> dict:
    """Index every saved source, and any already-indexed paper, into the shared chat corpus."""
    indexed: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    try:
        already = set(list_papers(GLOBAL_SESSION_ID))
    except Exception:
        already = set()

    for session_id, source in iter_saved_sources(session_ids):
        label = source.get("label") or "Untitled"
        try:
            docs = documents_for_source(source)
            _remember_text(session_id, source, docs)
            title = _title_of(docs, label)
            try:
                session_titles = set(list_papers(session_id))
            except Exception:
                session_titles = set()
            if title not in session_titles:
                add_paper(docs, session_id)
            if title in already:
                skipped.append(title)
                continue
            add_paper(docs, GLOBAL_SESSION_ID)
            already.add(title)
            indexed.append(title)
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    for session_id in session_ids:
        if session_id == GLOBAL_SESSION_ID:
            continue
        try:
            corpus = load_session_corpus(session_id)
        except Exception as exc:
            errors.append(f"{session_id}: {exc}")
            continue
        grouped: dict[str, list[Document]] = {}
        for doc in corpus:
            title = str(doc.metadata.get("title") or "Untitled paper")
            grouped.setdefault(title, []).append(doc)
        for title, docs in grouped.items():
            if title in already:
                continue
            try:
                docs.sort(key=lambda d: int(d.metadata.get("chunk_index") or 0))
                kind = str(docs[0].metadata.get("doc_type") or "pdf")
                docs = guard_documents(docs, kind)
                add_paper(docs, GLOBAL_SESSION_ID)
                already.add(title)
                indexed.append(title)
            except Exception as exc:
                errors.append(f"{title}: {exc}")

    return {"indexed": indexed, "skipped": skipped, "errors": errors}
