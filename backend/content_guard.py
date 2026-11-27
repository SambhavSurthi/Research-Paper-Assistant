"""Block non-research uploads and instruction-override text before it is stored or sent to a model.

Checks run locally. Rejected text is not forwarded to an LLM, an index, or a snapshot.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import fitz

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_DOWNLOAD_BYTES = MAX_UPLOAD_BYTES
MAX_PDF_PAGES = 200
MAX_USER_CHARS = 8_000
MIN_RESEARCH_CHARS = 800
_SHORT_INJECTION_CHARS = 6_000

_ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_PORTS = {80, 443, 8443}
_BLOCKED_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.google",
}

_HIDDEN_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
_FENCE_RE = re.compile(r"</?\s*untrusted_research_text\s*>", re.IGNORECASE)

_CRITICAL_RES = (
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|prompts|rules|directions|messages)",
    r"disregard\s+(?:your|the|all|any)\s+(?:previous|prior|system|above)",
    r"forget\s+(?:all|everything|your)\s+(?:previous|prior|instructions|rules|training)",
    r"(?:new|updated)\s+instructions?\s*:",
    r"(?:system|developer)\s*:\s*you\s+(?:must|will|are|should)",
    r"do\s+not\s+(?:follow|obey)\s+(?:the\s+)?(?:previous|original|system)",
    r"reveal\s+(?:your|the)\s+(?:hidden\s+|system\s+)?prompt",
    r"print\s+(?:your|the)\s+(?:hidden\s+|system\s+)?prompt",
    r"\bjailbreak\b",
    r"\bDAN\s+mode\b",
    r"<\|/?im_start\|>",
    r"\[/?INST\]",
    r"###\s*(?:system|instruction)",
    r"override\s+(?:the\s+)?(?:safety|system|previous)\s+(?:rules|instructions|prompt)",
    r"bypass\s+(?:your\s+)?(?:safety|guardrails|restrictions|filters)",
    r"<\s*/?\s*system\s*>",
)
_CRITICAL_RE = re.compile("|".join(f"(?:{pat})" for pat in _CRITICAL_RES), re.IGNORECASE)

_SUSPICIOUS_RE = re.compile(
    r"\byou are now\b|\bfrom now on\b|\bdeveloper mode\b|\bnew role\b",
    re.IGNORECASE,
)
_B64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{80,}={0,2}(?![A-Za-z0-9+/])")

_RESEARCH_GROUPS: dict[str, re.Pattern[str]] = {
    "abstract": re.compile(r"\babstract\b", re.IGNORECASE),
    "introduction": re.compile(r"\bintroduction\b", re.IGNORECASE),
    "methods": re.compile(r"\b(?:methods?|methodology|approach)\b", re.IGNORECASE),
    "results": re.compile(r"\b(?:experiments?|evaluation|results)\b", re.IGNORECASE),
    "conclusion": re.compile(r"\bconclusions?\b", re.IGNORECASE),
    "related": re.compile(r"\b(?:related work|literature review)\b", re.IGNORECASE),
    "contribution": re.compile(r"\b(?:this paper|we propose|we present|our (?:method|approach))\b", re.IGNORECASE),
    "citation": re.compile(
        r"\bet al\.?|\bdoi\s*:|\barxiv\s*:\s*\d{4}\.\d{4,5}|\[\d{1,3}\]",
        re.IGNORECASE,
    ),
}

_NON_RESEARCH_RE = re.compile(
    r"\b(?:curriculum vitae|résumé|resume)\b"
    r"|\bwork experience\b"
    r"|\b(?:invoice|amount due|bill to)\b"
    r"|\b(?:bank account|routing number|social security)\b"
    r"|\bdear hiring manager\b",
    re.IGNORECASE,
)


class ContentRejected(Exception):
    """Raised when text must not be indexed, snapshotted, or sent to a model."""

    def __init__(self, reason: str, category: str) -> None:
        self.reason = reason
        self.category = category
        super().__init__(reason)


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    category: str
    reason: str
    neutralize: bool = False


def screen_user_message(text: str) -> GuardResult:
    """Reject a chat or tool instruction that tries to override the assistant."""
    raw = text or ""
    if not raw.strip():
        return GuardResult(False, "empty", "Enter a research question. Empty messages are not sent to the model.")
    if len(raw) > MAX_USER_CHARS:
        return GuardResult(
            False,
            "too_long",
            "That message is too long to send. Ask a shorter research question, or upload the paper from the sidebar.",
        )
    if len(_HIDDEN_RE.findall(raw)) > 5 or _critical_count(raw) or _hidden_payload_count(raw):
        return _injection_message()
    suspicious = len(_SUSPICIOUS_RE.findall(raw))
    if suspicious >= 2 or (suspicious >= 1 and len(raw) < 280):
        return _injection_message()
    return GuardResult(True, "question", "Question accepted.")


def screen_url(url: str) -> GuardResult:
    """Allow only public http(s) addresses. Private hosts and other schemes are refused."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return GuardResult(False, "unsafe_url", "Only public http and https research links can be loaded.")
    if parsed.username or parsed.password:
        return GuardResult(False, "unsafe_url", "Links with embedded credentials cannot be loaded.")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host or host in _BLOCKED_HOSTS or host.endswith(".local") or host.endswith(".internal"):
        return GuardResult(False, "unsafe_url", "That address is not a public research link, so it was not fetched.")
    if parsed.port is not None and parsed.port not in _ALLOWED_PORTS:
        return GuardResult(False, "unsafe_url", "Only standard web ports can be loaded for research documents.")
    if not _host_is_public(host):
        return GuardResult(
            False,
            "unsafe_url",
            "That address points at a private or local network, so it was not fetched.",
        )
    return GuardResult(True, "url", "URL accepted.")


def screen_arxiv_query(query: str) -> GuardResult:
    text = (query or "").strip()
    if not text:
        return GuardResult(False, "empty", "Enter an arXiv id or a paper title.")
    if len(text) > 300 or _critical_count(text) or _hidden_payload_count(text):
        return _injection_message()
    return GuardResult(True, "arxiv", "arXiv query accepted.")


def screen_upload_bytes(filename: str, raw: bytes) -> GuardResult:
    """Classify an uploaded file before it is saved."""
    name = filename or "upload"
    if _critical_count(name) or _hidden_payload_count(name):
        return _injection_message()
    if not raw:
        return GuardResult(False, "empty", f"{name} is empty and was not saved.")
    if len(raw) > MAX_UPLOAD_BYTES:
        return GuardResult(False, "too_large", f"{name} is larger than 20 MB and was not saved.")
    suffix = Path(name).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        return GuardResult(
            False,
            "file_type",
            "Only PDF, TXT, and Markdown research papers can be uploaded.",
        )
    if suffix == ".pdf":
        if not raw.startswith(b"%PDF"):
            return GuardResult(False, "file_type", f"{name} is not a valid PDF and was not saved.")
        try:
            text = _pdf_text(raw)
        except ContentRejected as exc:
            return GuardResult(False, exc.category, str(exc))
        kind = "pdf"
    else:
        if b"\x00" in raw:
            return GuardResult(False, "file_type", f"{name} is not a plain-text research document and was not saved.")
        text = raw.decode("utf-8", errors="replace")
        kind = "text"
    return assess_research_document(text, source_kind=kind)


def assess_research_document(text: str, *, source_kind: str = "pdf") -> GuardResult:
    """Accept research papers. Refuse other documents and instruction-override files."""
    visible = _HIDDEN_RE.sub("", text or "")
    stripped = visible.strip()
    if len(stripped) < MIN_RESEARCH_CHARS:
        return GuardResult(
            False,
            "not_research",
            "This document is too short to be a research paper. It was not indexed or sent to the model.",
        )

    groups = {name for name, pattern in _RESEARCH_GROUPS.items() if pattern.search(stripped)}
    has_front = "abstract" in groups or "introduction" in groups
    has_citation = "citation" in groups
    min_groups = 2 if source_kind == "arxiv" else 3
    looks_like_research = has_front and has_citation and len(groups) >= min_groups
    non_research = len(_NON_RESEARCH_RE.findall(stripped))
    critical = _critical_count(text or "") + _hidden_payload_count(text or "")
    if len(_HIDDEN_RE.findall(text or "")) > 12:
        critical += 1

    if source_kind != "arxiv" and non_research >= 2 and len(groups) < 4:
        return GuardResult(
            False,
            "not_research",
            "This looks like a resume, invoice, or other non-research document. Only research papers can be uploaded, and it was not sent to the model.",
        )
    if critical and (not looks_like_research or len(stripped) < _SHORT_INJECTION_CHARS or critical >= 3):
        return _injection_document()
    if not looks_like_research:
        return GuardResult(
            False,
            "not_research",
            "Only research documents can be uploaded. This file does not have the structure of a paper, such as an abstract or introduction plus citations, so it was not indexed or sent to the model.",
        )
    return GuardResult(True, "research", "Research document accepted.", neutralize=critical > 0)


def scrub_text(text: str) -> str:
    """Remove hidden characters and instruction-override spans. Safe to call more than once."""
    if not text:
        return ""
    cleaned = _HIDDEN_RE.sub("", text).replace("\x00", "")
    cleaned = _FENCE_RE.sub("[removed fence]", cleaned)
    cleaned = _CRITICAL_RE.sub("[removed instruction-like text]", cleaned)
    return cleaned


def fence_untrusted(text: str) -> str:
    """Wrap document text so the model treats it as data."""
    return "<untrusted_research_text>\n" + scrub_text(text) + "\n</untrusted_research_text>"


def guarded_read(url: str, timeout: int = 30) -> tuple[bytes, str]:
    """Fetch a public URL and refuse private redirects or oversized downloads."""
    verdict = screen_url(url)
    if not verdict.allowed:
        raise ContentRejected(verdict.reason, verdict.category)
    opener = urllib.request.build_opener(_GuardedRedirectHandler)
    request = urllib.request.Request(url, headers={"User-Agent": "AIRearchPaperAssistant/1.0"})
    with opener.open(request, timeout=timeout) as response:
        final_url = response.geturl()
        final = screen_url(final_url)
        if not final.allowed:
            raise ContentRejected(final.reason, final.category)
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ContentRejected(
            "That download is larger than the 20 MB research-document limit and was not saved.",
            "too_large",
        )
    return data, final_url


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        verdict = screen_url(newurl)
        if not verdict.allowed:
            raise ContentRejected(verdict.reason, verdict.category)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _injection_message() -> GuardResult:
    return GuardResult(
        False,
        "injection",
        "That message tries to override the assistant's instructions. It was not sent to the model.",
    )


def _injection_document() -> GuardResult:
    return GuardResult(
        False,
        "injection",
        "This document contains instruction-override text. It was not indexed or sent to the model.",
    )


def _critical_count(text: str) -> int:
    return len(_CRITICAL_RE.findall(text or ""))


def _hidden_payload_count(text: str) -> int:
    hits = 0
    for match in _B64_RE.findall(text or ""):
        try:
            import base64

            decoded = base64.b64decode(match, validate=False).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if _CRITICAL_RE.search(decoded):
            hits += 1
    return hits


def _host_is_public(hostname: str) -> bool:
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        return _ip_is_public(literal)
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if not _ip_is_public(ip):
            return False
    return True


def _ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _pdf_text(raw: bytes) -> str:
    parts: list[str] = []
    try:
        pdf = fitz.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise ContentRejected("This PDF could not be read, so it was not saved.", "file_type") from exc
    with pdf:
        if pdf.page_count > MAX_PDF_PAGES:
            raise ContentRejected(
                f"Research uploads are limited to {MAX_PDF_PAGES} pages. This PDF was not saved.",
                "too_large",
            )
        if pdf.page_count == 0:
            return ""
        total = 0
        for page in pdf:
            text = page.get_text("text") or ""
            parts.append(text)
            total += len(text)
            if total > 1_500_000:
                break
    return "\n".join(parts)
