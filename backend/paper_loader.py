import hashlib
import re
import tempfile
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.content_guard import ContentRejected, guarded_read, screen_arxiv_query

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)

_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<num>(?:[1-9]\d?)(?:\.\d{1,2}){0,3})\.?\s+(?P<name>[A-Z][^\n]{2,90})$"
)
_ROMAN_HEADING_RE = re.compile(
    r"^(?P<num>[IVX]{1,5})\.\s+(?P<name>[A-Z][^\n]{2,90})$"
)
_STANDARD_SECTION_RE = re.compile(
    r"^(?:abstract|introduction|background|related\s+work|literature\s+review|"
    r"preliminaries|problem\s+formulation|methodology|method|methods|approach|"
    r"system\s+architecture|architecture|implementation|experimental\s+setup|"
    r"experiments?|evaluation|results|analysis|discussion|limitations|"
    r"future\s+work|conclusions?|acknowledg(?:e)?ments?|references|appendix(?:\s+[A-Z0-9]+)?)$",
    re.IGNORECASE,
)


def make_paper_id(title: str, source: str) -> str:
    """Return a deterministic short identifier for a paper."""
    raw = f"{title.strip()}::{source.strip()}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _heading_level_from_text(line: str, fallback_level: int = 1) -> int:
    m = _NUMBERED_HEADING_RE.match(line)
    if m:
        return min(m.group("num").count(".") + 1, 4)
    if _ROMAN_HEADING_RE.match(line):
        return 1
    return fallback_level


def _looks_like_heading(line: str, font_size: float | None = None, body_size: float | None = None) -> bool:
    stripped = line.strip()
    words = stripped.split()
    if len(stripped) < 3 or len(stripped) > 100 or len(words) > 12:
        return False
    if stripped.endswith((".", ",", ";", "?", "!")) and not _NUMBERED_HEADING_RE.match(stripped):
        return False
    if "—" in stripped or "http://" in stripped or "https://" in stripped:
        return False
    if _STANDARD_SECTION_RE.match(stripped):
        return True
    if _NUMBERED_HEADING_RE.match(stripped) or _ROMAN_HEADING_RE.match(stripped):
        return True
    if font_size is not None and body_size is not None and body_size > 0:
        if font_size >= body_size * 1.22 and len(words) <= 10 and stripped[0].isupper():
            return True
    return False


def _build_hierarchical_chunks(
    sections: list[dict],
    title: str,
    source: str,
    doc_type: str,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """Split extracted sections into chunks while preserving full document hierarchy."""
    clean_sections = [s for s in sections if (s.get("text") or "").strip()]
    if not clean_sections:
        return []

    paper_id = make_paper_id(title, source)
    stack: list[tuple[int, str, str, str]] = []  # (level, section_id, section_title, section_path)
    docs: list[Document] = []
    chunk_index = 0

    for sec_order, sec in enumerate(clean_sections):
        sec_title = (sec.get("title") or f"Section {sec_order + 1}").strip()
        sec_level = max(1, int(sec.get("level") or 1))
        sec_page = sec.get("page")
        sec_id = f"{paper_id}::sec_{sec_order}"

        while stack and stack[-1][0] >= sec_level:
            stack.pop()

        if stack:
            parent_id = stack[-1][1]
            parent_title = stack[-1][2]
            sec_path = f"{stack[-1][3]} > {sec_title}"
        else:
            parent_id = None
            parent_title = None
            sec_path = sec_title

        stack.append((sec_level, sec_id, sec_title, sec_path))

        base_meta = {
            "title": title,
            "source": source,
            "doc_type": doc_type,
            "paper_id": paper_id,
            "section_id": sec_id,
            "section_title": sec_title,
            "section_path": sec_path,
            "section_level": sec_level,
            "section_order": sec_order,
            "parent_section_id": parent_id,
            "parent_section_title": parent_title,
        }
        if sec_page is not None:
            base_meta["page"] = sec_page

        split_docs = splitter.create_documents([sec["text"].strip()], metadatas=[base_meta])
        for d in split_docs:
            d.metadata["chunk_index"] = chunk_index
            d.metadata["chunk_id"] = f"{paper_id}::chunk_{chunk_index}"
            docs.append(d)
            chunk_index += 1

    return docs


def _extract_pdf_sections(file_path: str) -> tuple[list[dict], str]:
    """Extract hierarchical sections from a PDF using font-size + heading heuristics."""
    with fitz.open(file_path) as pdf:
        pdf_title = (pdf.metadata.get("title") or "").strip() if pdf.metadata else ""

        font_counter: Counter[float] = Counter()
        page_lines: list[tuple[int, str, float]] = []

        for page_idx, page in enumerate(pdf, start=1):
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(sp.get("text", "") for sp in spans).strip()
                    if not text:
                        continue
                    sizes = [round(float(sp.get("size", 0)), 1) for sp in spans if sp.get("text", "").strip()]
                    max_size = max(sizes) if sizes else 0.0
                    if max_size > 0:
                        font_counter[max_size] += len(text)
                    page_lines.append((page_idx, text, max_size))

        body_size = font_counter.most_common(1)[0][0] if font_counter else 10.0

        sections: list[dict] = []
        current_title = "Preamble"
        current_level = 1
        current_page = 1
        buffer: list[str] = []

        for page_idx, text, font_size in page_lines:
            if _looks_like_heading(text, font_size=font_size, body_size=body_size):
                if buffer:
                    sections.append({
                        "title": current_title,
                        "level": current_level,
                        "page": current_page,
                        "text": "\n".join(buffer),
                    })
                    buffer = []
                current_title = text
                if _NUMBERED_HEADING_RE.match(text) or _ROMAN_HEADING_RE.match(text):
                    current_level = _heading_level_from_text(text, fallback_level=1)
                elif font_size >= body_size * 1.45:
                    current_level = 1
                elif font_size >= body_size * 1.18:
                    current_level = 2
                else:
                    current_level = 1
                current_page = page_idx
            else:
                buffer.append(text)

        if buffer:
            sections.append({
                "title": current_title,
                "level": current_level,
                "page": current_page,
                "text": "\n".join(buffer),
            })

        # Fallback to page-level sections if no structural headings were detected
        if len(sections) <= 1:
            page_sections = []
            for page_idx, page in enumerate(pdf, start=1):
                txt = (page.get_text("text") or "").strip()
                if txt:
                    page_sections.append({
                        "title": f"Page {page_idx}",
                        "level": 1,
                        "page": page_idx,
                        "text": txt,
                    })
            if page_sections:
                sections = page_sections

        return sections, pdf_title


def _extract_markdown_sections(text: str) -> list[dict]:
    """Parse ATX markdown headings (# .. ####) into hierarchical sections."""
    sections: list[dict] = []
    current_title = "Introduction"
    current_level = 1
    buffer: list[str] = []
    in_code_block = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            buffer.append(line)
            continue
        if not in_code_block:
            m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if m:
                if "\n".join(buffer).strip():
                    sections.append({
                        "title": current_title,
                        "level": current_level,
                        "page": None,
                        "text": "\n".join(buffer),
                    })
                buffer = []
                current_level = min(len(m.group(1)), 4)
                current_title = m.group(2).strip()
                continue
        buffer.append(line)

    if "\n".join(buffer).strip():
        sections.append({
            "title": current_title,
            "level": current_level,
            "page": None,
            "text": "\n".join(buffer),
        })
    return sections


def _extract_text_sections(text: str) -> list[dict]:
    """Detect numbered or standard academic headings in plain text."""
    sections: list[dict] = []
    current_title = "Document Body"
    current_level = 1
    buffer: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if _looks_like_heading(stripped):
            if "\n".join(buffer).strip():
                sections.append({
                    "title": current_title,
                    "level": current_level,
                    "page": None,
                    "text": "\n".join(buffer),
                })
            buffer = []
            current_title = stripped
            current_level = _heading_level_from_text(stripped, fallback_level=1)
        else:
            buffer.append(line)

    if "\n".join(buffer).strip():
        sections.append({
            "title": current_title,
            "level": current_level,
            "page": None,
            "text": "\n".join(buffer),
        })
    return sections


def restamp_title(docs: list[Document], title: str) -> list[Document]:
    """Update paper title and deterministic IDs when an uploaded temp file gets its real filename."""
    if not docs:
        return docs
    source = docs[0].metadata.get("source", title)
    old_paper_id = docs[0].metadata.get("paper_id")
    new_paper_id = make_paper_id(title, source)
    for doc in docs:
        doc.metadata["title"] = title
        if old_paper_id and old_paper_id != new_paper_id:
            for key in ("paper_id", "section_id", "parent_section_id", "chunk_id"):
                val = doc.metadata.get(key)
                if isinstance(val, str):
                    doc.metadata[key] = val.replace(old_paper_id, new_paper_id, 1)
    return docs


def load_pdf(file_path: str) -> list[Document]:
    sections, _ = _extract_pdf_sections(file_path)
    title = Path(file_path).stem
    return _build_hierarchical_chunks(sections, title=title, source=file_path, doc_type="pdf", splitter=_splitter)


def load_text(file_path: str) -> list[Document]:
    raw_docs = TextLoader(file_path, encoding="utf-8").load()
    full_text = "\n\n".join(d.page_content for d in raw_docs)
    sections = _extract_text_sections(full_text)
    title = Path(file_path).stem
    return _build_hierarchical_chunks(sections, title=title, source=file_path, doc_type="text", splitter=_splitter)


def load_markdown(file_path: str) -> list[Document]:
    raw_docs = TextLoader(file_path, encoding="utf-8").load()
    full_text = "\n\n".join(d.page_content for d in raw_docs)
    sections = _extract_markdown_sections(full_text)
    title = Path(file_path).stem
    return _build_hierarchical_chunks(sections, title=title, source=file_path, doc_type="markdown", splitter=_md_splitter)


def load_webpage(url: str) -> list[Document]:
    html_bytes, final_url = guarded_read(url, timeout=30)
    html = html_bytes.decode("utf-8", errors="replace")
    url = final_url

    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string.strip() if soup.title and soup.title.string else url)
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    sections: list[dict] = []
    current_title = "Overview"
    current_level = 1
    buffer: list[str] = []

    for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre"]):
        text = elem.get_text(" ", strip=True)
        if not text:
            continue
        if elem.name in ("h1", "h2", "h3", "h4"):
            if buffer:
                sections.append({
                    "title": current_title,
                    "level": current_level,
                    "page": None,
                    "text": "\n".join(buffer),
                })
                buffer = []
            current_title = text[:120]
            current_level = int(elem.name[1])
        else:
            buffer.append(text)

    if buffer:
        sections.append({
            "title": current_title,
            "level": current_level,
            "page": None,
            "text": "\n".join(buffer),
        })

    if not sections:
        plain = soup.get_text("\n", strip=True)
        sections = [{"title": "Overview", "level": 1, "page": None, "text": plain}]

    return _build_hierarchical_chunks(sections, title=title, source=url, doc_type="webpage", splitter=_splitter)


def _extract_arxiv_id(query: str) -> str | None:
    """Return bare ArXiv ID (no version suffix) if one appears in the query."""
    m = _ARXIV_ID_RE.search(query)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    return None


def _arxiv_api_lookup(arxiv_id: str) -> str:
    """Fetch paper title by ID from the ArXiv Atom API."""
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    xml_bytes, _ = guarded_read(url, timeout=10)
    xml = xml_bytes.decode("utf-8", errors="replace")
    titles = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)
    return titles[1].strip() if len(titles) > 1 else arxiv_id


def _arxiv_search(query: str) -> str:
    """Search ArXiv Atom API by title phrase and return the top result's bare paper ID."""
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = f"https://export.arxiv.org/api/query?search_query={search_query}&max_results=1&sortBy=relevance"
    xml_bytes, _ = guarded_read(url, timeout=15)
    xml = xml_bytes.decode("utf-8", errors="replace")
    m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml)
    if not m:
        raise ValueError(f"No ArXiv paper found for: {query}")
    return re.sub(r"v\d+$", "", m.group(1))


def _load_arxiv_by_id(arxiv_id: str) -> list[Document]:
    """Download and hierarchically chunk an ArXiv paper PDF by its bare ID."""
    pdf_url = f"https://arxiv.org/pdf/{urllib.parse.quote(arxiv_id)}"
    pdf_bytes, _ = guarded_read(pdf_url, timeout=60)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        sections, pdf_title = _extract_pdf_sections(tmp_path)
        if not sections:
            raise ValueError(f"Could not load PDF for ArXiv ID: {arxiv_id}")
        title = pdf_title or _arxiv_api_lookup(arxiv_id)
        return _build_hierarchical_chunks(
            sections,
            title=title,
            source=f"https://arxiv.org/abs/{arxiv_id}",
            doc_type="arxiv",
            splitter=_splitter,
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def load_arxiv(query: str) -> list[Document]:
    verdict = screen_arxiv_query(query)
    if not verdict.allowed:
        raise ContentRejected(verdict.reason, verdict.category)
    arxiv_id = _extract_arxiv_id(query) or _arxiv_search(query)
    return _load_arxiv_by_id(arxiv_id)


def load_document(source: str) -> list[Document]:
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    if source.startswith(("http://", "https://")):
        return load_webpage(source)
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source)
    if ext == ".txt":
        return load_text(source)
    if ext in (".md", ".markdown"):
        return load_markdown(source)
    raise ValueError(f"Unsupported file type: {ext}")
