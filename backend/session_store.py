"""Per-session files: saved sources, chat history, and research-tool history."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

DATA_ROOT = Path("session_data")
GLOBAL_SESSION_ID = "global-corpus"
EVAL_SESSION_ID = "evaluation"


def _record_path(session_id: str) -> Path:
    return DATA_ROOT / session_id / "record.json"


def load_record(session_id: str) -> dict:
    path = _record_path(session_id)
    if not path.exists():
        return {"sources": [], "chats": [], "tool_history": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"sources": [], "chats": [], "tool_history": []}
    data.setdefault("sources", [])
    data.setdefault("chats", [])
    data.setdefault("tool_history", [])
    return data


def save_record(session_id: str, record: dict) -> None:
    path = _record_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def list_sources(session_id: str) -> list[dict]:
    return list(load_record(session_id).get("sources") or [])


def iter_saved_sources(session_ids: list[str]):
    for session_id in session_ids:
        if session_id in (GLOBAL_SESSION_ID, EVAL_SESSION_ID):
            continue
        for source in list_sources(session_id):
            yield session_id, source


def source_file(source: dict) -> Path | None:
    rel = source.get("stored_path")
    if not rel:
        return None
    path = DATA_ROOT / rel
    return path if path.exists() else None


def snapshot_file(source: dict) -> Path | None:
    rel = source.get("snapshot_path")
    if not rel:
        return None
    path = DATA_ROOT / rel
    return path if path.exists() else None


def _existing_source(record: dict, *, kind: str, label: str, url: str | None, query: str | None) -> dict | None:
    for source in record["sources"]:
        if source.get("kind") != kind:
            continue
        if url and source.get("url") == url:
            return source
        if query and source.get("query") == query:
            return source
        if source.get("label") == label:
            return source
    return None


def add_source(
    session_id: str,
    *,
    kind: str,
    label: str,
    raw: bytes | None = None,
    suffix: str = "",
    url: str | None = None,
    query: str | None = None,
) -> tuple[dict, bool]:
    """Save a source on the session. Returns (source, created)."""
    record = load_record(session_id)
    existing = _existing_source(record, kind=kind, label=label, url=url, query=query)
    if existing:
        return existing, False

    source_id = uuid.uuid4().hex[:12]
    stored_path = None
    if raw is not None:
        rel = f"{session_id}/sources/{source_id}{suffix}"
        dest = DATA_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        stored_path = rel

    source = {
        "id": source_id,
        "kind": kind,
        "label": label,
        "stored_path": stored_path,
        "snapshot_path": None,
        "url": url,
        "query": query,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    record["sources"].append(source)
    save_record(session_id, record)
    return source, True


def remove_source(session_id: str, source_id: str) -> None:
    """Delete a saved source and its files. Used when a document fails the safety check."""
    record = load_record(session_id)
    kept = []
    for source in record["sources"]:
        if source.get("id") != source_id:
            kept.append(source)
            continue
        for rel in (source.get("stored_path"), source.get("snapshot_path")):
            if not rel:
                continue
            path = DATA_ROOT / rel
            if path.exists():
                path.unlink()
    record["sources"] = kept
    save_record(session_id, record)


def attach_snapshot(session_id: str, source_id: str, text: str) -> None:
    record = load_record(session_id)
    rel = f"{session_id}/sources/{source_id}.txt"
    dest = DATA_ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    for source in record["sources"]:
        if source.get("id") == source_id:
            source["snapshot_path"] = rel
            break
    save_record(session_id, record)


def save_chats(session_id: str, chats: list[dict]) -> None:
    record = load_record(session_id)
    record["chats"] = chats
    save_record(session_id, record)


def save_tool_history(session_id: str, history: list[dict]) -> None:
    record = load_record(session_id)
    record["tool_history"] = history
    save_record(session_id, record)
