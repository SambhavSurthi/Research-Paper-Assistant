#!/usr/bin/env python3
"""Build Capstone-Report-v2.docx from the v1 document as a style template.

The template's styles.xml, theme, headers and footers (with page-number fields)
are preserved. All body content is replaced with project-accurate material,
compact A4-width figures, properly typeset formula images, and a linked TOC.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
TEMPLATE = BASE / "Capstone-Report-v1.docx"
OUTPUT = BASE / "Capstone-Report-v3.docx"
ASSETS = BASE / "assets"
TECTONIC = Path.home() / "AppData/Local/tectonic_msvc/tectonic.exe"

BLUE_DARK = RGBColor(0x17, 0x36, 0x5D)
BLUE_MID = RGBColor(0x1F, 0x3E, 0x6F)
BLUE_LIGHT = RGBColor(0x2D, 0x75, 0xB5)


# ------------------------------------------------------------------ style helpers
def get_style(doc, name):
    for s in doc.styles:
        if s.name == name:
            return s
    return doc.styles["Normal"]


def set_cell_bg(cell, color="DCE6F1"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def mark_update_fields(docx_path: Path) -> None:
    """Ask Word to refresh fields (TOC page numbers and hyperlinks) on open."""
    with zipfile.ZipFile(docx_path, "r") as zin:
        items = {n: zin.read(n) for n in zin.namelist()}
    settings = items["word/settings.xml"].decode("utf-8")
    if "updateFields" not in settings:
        settings = settings.replace(
            "<w:settings", '<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
            if 'xmlns:w=' not in settings else "<w:settings", 1)
        idx = settings.find(">", settings.find("<w:settings"))
        settings = settings[:idx + 1] + '<w:updateFields w:val="true"/>' + settings[idx + 1:]
    items["word/settings.xml"] = settings.encode("utf-8")
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in items.items():
            zout.writestr(name, data)


# ----------------------------------------------------------------------- document
class Report:
    def __init__(self):
        self.doc = Document(TEMPLATE)
        body = self.doc.element.body
        for child in list(body):
            if child.tag != qn("w:sectPr"):
                body.remove(child)
        self.fig_count = 0
        self.tab_count = 0
        self.toc_entries: list[tuple[str, int, str]] = []
        self._bm_id = 1000

    def _p(self, text="", style="Body Text", color=None, align=None,
           size=None, bold=None, italic=None, before=None, after=None,
           line=1.15, page_break=False, keep_next=False):
        p = self.doc.add_paragraph(style=get_style(self.doc, style))
        if text:
            p.add_run(text)
        if color is not None:
            for r in p.runs:
                r.font.color.rgb = color
        if size is not None:
            for r in p.runs:
                r.font.size = Pt(size)
        if bold is not None:
            for r in p.runs:
                r.bold = bold
        if italic is not None:
            for r in p.runs:
                r.italic = italic
        pf = p.paragraph_format
        if align is not None:
            pf.alignment = align
        if before is not None:
            pf.space_before = Pt(before)
        if after is not None:
            pf.space_after = Pt(after)
        pf.line_spacing = line
        pf.page_break_before = page_break
        pf.keep_with_next = keep_next
        return p

    def title(self, text, *, color=BLUE_DARK, size=16, align=WD_ALIGN_PARAGRAPH.CENTER,
              before=8, after=6, line=1.1):
        return self._p(text, style="Heading 6", color=color, size=size,
                       align=align, bold=True, before=before, after=after, line=line)

    def _bookmark(self, p, name: str) -> None:
        self._bm_id += 1
        start = OxmlElement("w:bookmarkStart")
        start.set(qn("w:id"), str(self._bm_id))
        start.set(qn("w:name"), name)
        end = OxmlElement("w:bookmarkEnd")
        end.set(qn("w:id"), str(self._bm_id))
        p._p.insert(1, start)
        p._p.append(end)

    def chapter(self, text):
        p = self._p(text, style="Heading 4", color=BLUE_DARK, size=16,
                    before=6, after=10, line=1.1, page_break=True, keep_next=True)
        name = "ch_" + str(len(self.toc_entries))
        self._bookmark(p, name)
        self.toc_entries.append((text, 1, name))
        return p

    def section(self, text):
        p = self._p(text, style="Heading 2", color=BLUE_LIGHT, size=14,
                    before=14, after=6, line=1.1, keep_next=True)
        name = "sec_" + str(len(self.toc_entries))
        self._bookmark(p, name)
        self.toc_entries.append((text, 2, name))
        return p

    def subsection(self, text):
        return self._p(text, style="Heading 7", color=BLUE_DARK, size=12,
                       before=10, after=4, line=1.1, keep_next=True)

    def body(self, text):
        return self._p(text, style="Body Text", align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                       line=1.15, after=6)

    def bullet(self, text):
        return self._p("•  " + text, style="List Paragraph",
                       align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.15, after=3)

    def numbered(self, n, text):
        return self._p(f"{n}.  " + text, style="List Paragraph",
                       align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.15, after=3)

    def image(self, path: Path, caption: str, width_cm=15.5):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        self.fig_count += 1
        cap = self._p(f"Figure {self.fig_count}. {caption}", style="Body Text",
                      align=WD_ALIGN_PARAGRAPH.CENTER, size=10, after=10, line=1.0)
        for r in cap.runs:
            r.italic = True

    def table(self, headers, rows, caption, widths=None):
        self.tab_count += 1
        cap = self._p(f"Table {self.tab_count}. {caption}", style="Heading 7",
                      color=BLUE_DARK, size=10.5, before=8, after=4, line=1.0, keep_next=True)
        t = self.doc.add_table(rows=1, cols=len(headers))
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement("w:" + edge)
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), "4")
            e.set(qn("w:color"), "7F7F7F")
            borders.append(e)
        t._tbl.tblPr.append(borders)
        t.autofit = True
        for i, h in enumerate(headers):
            c = t.rows[0].cells[i]
            c.text = h
            set_cell_bg(c)
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                for r in p.runs:
                    r.bold = True
                    r.font.size = Pt(9.5)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = str(v)
                for p in cells[i].paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.line_spacing = 1.0
                    for r in p.runs:
                        r.font.size = Pt(9)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def toc_marker(self):
        self.toc_marker_p = self._p("__TOC_MARKER__", style="Body Text")

    def populate_toc(self, page_map):
        created = []
        h = self._p("TABLE OF CONTENTS", style="Heading 1", color=BLUE_MID,
                    size=18, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=12,
                    keep_next=True, page_break=True)
        created.append(h._p)
        if not page_map:
            p = self.doc.add_paragraph()
            p.add_run("Table of contents will be generated on the second build pass.")
            return
        for text, level, bm in self.toc_entries:
            p = self.doc.add_paragraph(style="TOC 2")
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.left_indent = Cm(0.6 * (level - 1))
            pPr = p._p.get_or_add_pPr()
            tabs = OxmlElement("w:tabs")
            tab = OxmlElement("w:tab")
            tab.set(qn("w:val"), "right")
            tab.set(qn("w:leader"), "dot")
            tab.set(qn("w:pos"), "9000")
            tabs.append(tab)
            pPr.append(tabs)
            hl = OxmlElement("w:hyperlink")
            hl.set(qn("w:anchor"), bm)
            rnode = OxmlElement("w:r")
            rPr = OxmlElement("w:rPr")
            color = OxmlElement("w:color")
            color.set(qn("w:val"), "17365D" if level == 1 else "2D75B5")
            rPr.append(color)
            if level == 1:
                b = OxmlElement("w:b")
                rPr.append(b)
            rnode.append(rPr)
            tnode = OxmlElement("w:t")
            tnode.text = text
            rnode.append(tnode)
            hl.append(rnode)
            p._p.append(hl)
            p.add_run("\t" + str(page_map.get(bm, "")))
            created.append(p._p)
        for element in created:
            self.toc_marker_p._p.addprevious(element)
        self.toc_marker_p._p.getparent().remove(self.toc_marker_p._p)

    def save(self):
        self.doc.save(OUTPUT)
        mark_update_fields(OUTPUT)


# ------------------------------------------------------------------ asset creation
PREAMBLE = r"""
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta,shapes.geometric,calc,positioning,patterns}
\tikzset{
  b/.style={draw=black!70,rounded corners=2pt,fill=black!7,minimum height=0.6cm,
            text width=2.2cm,align=center,font=\scriptsize},
  d/.style={draw=black!70,rounded corners=2pt,fill=black!18,minimum height=0.6cm,
            text width=2.2cm,align=center,font=\scriptsize},
  a/.style={-{Latex[length=2mm]},thick,black!65}
}
"""

FIGURES = {
    "fig_context.tex": r"""
\begin{tikzpicture}
\node[b] (u) at (0,0) {Researcher};
\node[b] (ui) at (2.7,0) {Streamlit UI};
\node[d] (lg) at (5.4,0) {LangGraph router};
\node[b] (q) at (8.1,0.7) {Qdrant vector index};
\node[b] (g) at (8.1,-0.7) {Neo4j two-level graph};
\node[b] (o) at (10.8,0) {Cited answer};
\node[b] (t) at (5.4,-1.7) {Tavily web search};
\draw[a] (u)--(ui); \draw[a] (ui)--(lg);
\draw[a] (lg.east)--++(0.2,0)|-(q.west);
\draw[a] (lg.east)--++(0.2,0)|-(g.west);
\draw[a] (q.east)--++(0.2,0)|-(o.west);
\draw[a] (g.east)--++(0.2,0)|-(o.west);
\draw[a] (lg.south)--(t.north);
\end{tikzpicture}
""",
    "fig_ingest.tex": r"""
\begin{tikzpicture}
\node[b] (s) at (0,0) {Source};
\node[b] (gd) at (2.3,0) {Content guard};
\node[b] (ld) at (4.6,0) {Section parser};
\node[b] (ch) at (6.9,0) {1000/200 chunking};
\node[d] (em) at (9.2,0) {Cached embeddings};
\node[b] (qd) at (11.5,0.65) {Qdrant};
\node[b] (ng) at (11.5,-0.65) {Neo4j graph};
\draw[a] (s)--(gd); \draw[a] (gd)--(ld); \draw[a] (ld)--(ch); \draw[a] (ch)--(em);
\draw[a] (em.east)--++(0.15,0)|-(qd.west);
\draw[a] (em.east)--++(0.15,0)|-(ng.west);
\end{tikzpicture}
""",
    "fig_graph.tex": r"""
\begin{tikzpicture}
\node[b] (p) at (0,0.7) {Paper};
\node[b] (s) at (2.4,0.7) {Section};
\node[b] (c) at (4.8,0.7) {Chunk};
\node[b] (e1) at (2.4,-0.8) {Method};
\node[b] (e2) at (4.8,-0.8) {Finding};
\node[d] (pv) at (7.2,-0.8) {Provenance\\page + quote};
\draw[a] (p)--(s); \draw[a] (s)--(c);
\draw[a] (c.south)--++(0,-0.25)|-(e1.north);
\draw[a] (c.south)--++(0,-0.25)|-(e2.north);
\draw[a] (e1)--(e2); \draw[a] (e2)--(pv);
\node[font=\scriptsize\itshape] at (3.6,1.25) {Level 1: document hierarchy};
\node[font=\scriptsize\itshape] at (4.8,-1.35) {Level 2: semantic graph};
\end{tikzpicture}
""",
    "fig_retrieval.tex": r"""
\begin{tikzpicture}
\node[b] (q) at (0,0) {Query};
\node[b] (v) at (2.4,0.7) {Vector seeds};
\node[b] (e) at (2.4,-0.7) {Entity seeds};
\node[b] (m) at (4.9,0) {Merge, top $k$};
\node[b] (h) at (7.3,0.7) {Hierarchy expansion};
\node[b] (sim) at (7.3,-0.7) {Similarity links};
\node[b] (out) at (9.8,0) {Context $\le \max(3k,10)$};
\draw[a] (q.east)--++(0.15,0)|-(v.west);
\draw[a] (q.east)--++(0.15,0)|-(e.west);
\draw[a] (v.east)--++(0.15,0)|-(m.west);
\draw[a] (e.east)--++(0.15,0)|-(m.west);
\draw[a] (m.east)--++(0.15,0)|-(h.west);
\draw[a] (m.east)--++(0.15,0)|-(sim.west);
\draw[a] (h.east)--++(0.15,0)|-(out.west);
\draw[a] (sim.east)--++(0.15,0)|-(out.west);
\end{tikzpicture}
""",
    "fig_router.tex": r"""
\begin{tikzpicture}
\node[b] (u) at (0,0) {Question};
\node[d] (g) at (2.2,0) {Intake guard};
\node[d] (r) at (4.4,0) {Router};
\node[b] (a) at (6.6,0.8) {Retrieval agent};
\node[b] (v) at (6.6,-0.8) {Claim verification};
\node[b] (d) at (6.6,0) {Direct answer};
\node[b] (ans) at (9.3,0) {Answer};
\draw[a] (u)--(g); \draw[a] (g)--(r);
\draw[a] (r.east)--++(0.15,0)|-(a.west);
\draw[a] (r)--(d);
\draw[a] (r.east)--++(0.15,0)|-(v.west);
\draw[a] (a.east)--++(0.15,0)|-(ans.west);
\draw[a] (d.east)--++(0.15,0)|-(ans.west);
\draw[a] (v.east)--++(0.15,0)|-(ans.west);
\end{tikzpicture}
""",
    "fig_security.tex": r"""
\begin{tikzpicture}
\node[b] (i) at (0,0) {Untrusted input};
\node[b] (f) at (2.4,0.75) {File checks};
\node[b] (u) at (2.4,0) {URL checks};
\node[b] (m) at (2.4,-0.75) {Message checks};
\node[b] (c) at (5.1,0) {Research classifier};
\node[b] (s) at (7.6,0) {Scrub and fence};
\node[d] (o) at (10,0) {Safe for model};
\draw[a] (i.east)--++(0.15,0)|-(f.west);
\draw[a] (i)--(u);
\draw[a] (i.east)--++(0.15,0)|-(m.west);
\draw[a] (f.east)--++(0.15,0)|-(c.west);
\draw[a] (u.east)--++(0.15,0)|-(c.west);
\draw[a] (m.east)--++(0.15,0)|-(c.west);
\draw[a] (c)--(s); \draw[a] (s)--(o);
\end{tikzpicture}
""",
    "chart_metric_means.tex": r"""
\begin{tikzpicture}
\begin{axis}[width=13cm,height=5.2cm,ybar,bar width=14pt,
  symbolic x coords={CP,CR,CRe,AR,F},xtick=data,ymin=0,ymax=1.15,
  ytick={0,0.2,0.4,0.6,0.8,1.0},grid=major,grid style={dashed,black!15},
  nodes near coords,nodes near coords align={vertical},
  every node near coord/.append style={font=\tiny,anchor=south},
  enlarge x limits=0.25,ylabel={Mean score},xlabel={Metric},
  tick label style={font=\tiny},label style={font=\tiny}]
\addplot[fill=black!72,draw=black] coordinates {(CP,0.978)(CR,1.000)(CRe,0.576)(AR,0.991)(F,0.981)};
\draw[dashed,thick,black!55] (rel axis cs:0,0.6087) -- (rel axis cs:1,0.6087);
\end{axis}
\end{tikzpicture}
""",
    "chart_cre_trend.tex": r"""
\begin{tikzpicture}
\begin{axis}[width=13cm,height=5.2cm,xmin=1,xmax=10,ymin=0,ymax=1.05,
  xtick={1,2,...,10},ytick={0,0.2,0.4,0.6,0.8,1.0},grid=major,
  grid style={dashed,black!15},mark=*,mark size=2pt,
  xlabel={Test case},ylabel={Contextual relevancy},
  tick label style={font=\tiny},label style={font=\tiny}]
\addplot[thick,black!75] coordinates {
 (1,0.094)(2,0.269)(3,0.500)(4,0.974)(5,0.826)
 (6,0.791)(7,0.727)(8,0.441)(9,0.772)(10,0.362)};
\addplot[dashed,thick,black!55] coordinates {(1,0.7)(10,0.7)};
\end{axis}
\end{tikzpicture}
""",
    "chart_pass_rates.tex": r"""
\begin{tikzpicture}
\begin{axis}[width=13cm,height=5.2cm,ybar,bar width=14pt,
  symbolic x coords={CP,CR,CRe,AR,F},xtick=data,ymin=0,ymax=11,
  ytick={0,2,4,6,8,10},grid=major,grid style={dashed,black!15},
  nodes near coords,nodes near coords align={vertical},
  every node near coord/.append style={font=\tiny,anchor=south},
  enlarge x limits=0.25,ylabel={Cases passed (of 10)},xlabel={Metric},
  tick label style={font=\tiny},label style={font=\tiny}]
\addplot[fill=black!72,draw=black] coordinates {(CP,10)(CR,10)(CRe,5)(AR,10)(F,10)};
\end{axis}
\end{tikzpicture}
""",
}

FORMULAS = {
    "eq_chunk.tex": r"$C_j=[s+j(\ell-o),\;\min(s+(j+1)\ell,\,L)),\quad \ell=1000,\;o=200$",
    "eq_cosine.tex": r"$\operatorname{sim}(q,c)=\dfrac{q^{\top}c}{\|q\|\,\|c\|}$",
    "eq_simlink.tex": r"$\operatorname{sim}(c_i,c_j)\ge 0.78,\qquad \deg_{\mathrm{sim}}(c_i)\le 2$",
    "eq_budget.tex": r"$N_{\mathrm{ctx}}=\max(3k,10),\qquad n_v=\max(8k,40)$",
    "eq_metrics.tex": r"$P@K=\dfrac{1}{K}\sum_{i=1}^{K}r_i,\qquad R@K=\dfrac{\sum_{i=1}^{K}r_i}{|\mathcal{R}|},\qquad \mathrm{pass}=\mathbb{1}\!\left[\min_m s_m\ge 0.7\right]$",
}


def make_assets() -> dict[str, Path]:
    ASSETS.mkdir(exist_ok=True)
    if not TECTONIC.exists():
        raise RuntimeError("Tectonic not found; cannot render report assets")
    out: dict[str, Path] = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, body in {**FIGURES, **FORMULAS}.items():
            is_fig = name.startswith("fig_")
            doc = (
                "\\documentclass[border=6pt]{standalone}\n"
                + PREAMBLE
                + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
            )
            (tmp / "asset.tex").write_text(doc, encoding="utf-8")
            (tmp / "asset.pdf").unlink(missing_ok=True)
            proc = subprocess.run(
                [str(TECTONIC), "-X", "compile", "asset.tex"],
                cwd=tmp, capture_output=True, text=True, timeout=180
            )
            if proc.returncode != 0 or not (tmp / "asset.pdf").exists():
                print("FAILED", name, proc.stderr[-300:] if proc.stderr else "")
                continue
            import fitz
            pdf = fitz.open(tmp / "asset.pdf")
            pix = pdf[0].get_pixmap(dpi=200, alpha=False)
            dest = ASSETS / (name.replace(".tex", ".png"))
            pix.save(dest)
            pdf.close()
            out[name] = dest
    # Real project state machine image, copied from the repository root.
    src = ROOT / "graph.png"
    if src.exists():
        shutil.copy(src, ASSETS / "fig_state.png")
        out["fig_state.png"] = ASSETS / "fig_state.png"
    # Diagrams rendered from the mermaid blocks in README.md.
    for name in ("readme_glance", "readme_architecture", "readme_router", "readme_sequence"):
        f = ASSETS / f"{name}.png"
        if f.exists():
            out[name] = f
    return out


# ------------------------------------------------------------------------ content
def write_content(r: Report, a: dict[str, Path], page_map=None) -> None:
    # ---- front matter -------------------------------------------------------
    r.title("KONERU LAKSHMAIAH EDUCATION FOUNDATION", size=14, before=10)
    r.title("(DEEMED TO BE UNIVERSITY)", size=11)
    r.title("FINAL PROJECT REPORT", color=BLUE_DARK, size=20, before=28)
    r.title("AI RESEARCH PAPER ASSISTANT", color=BLUE_DARK, size=16, before=20)
    r.title("A HYBRID GRAPHRAG ARCHITECTURE FOR CITATION-GROUNDED,\nMULTI-DOCUMENT SCHOLARLY QUESTION ANSWERING",
            color=BLUE_DARK, size=13, line=1.2)
    r.body("")
    r.title("Submitted in partial fulfilment of the requirements for the degree of",
            size=11, color=None)
    r.title("BACHELOR OF TECHNOLOGY", size=14, before=10)
    r.title("Department of Computer Science and Engineering / Information Technology",
            size=11, before=16)
    r.title("Under the Guidance of", size=11, before=24)
    r.title("Mr. Bhavanam Venkata Suresh Reddy (Project Guide)", size=12)
    r.title("KONERU LAKSHMAIAH EDUCATION FOUNDATION", size=11, before=26)
    r.title("Vijayawada, India", size=11)
    r.title("Ch. Charitha (2300030127)   •   J. Adiseshu (2300030279)", size=11, before=30)
    r.title("S. Sambhav (2300031622)   •   B. Jyothi Naga Sravani (2300090216)", size=11)

    r.chapter("CERTIFICATE")
    r.body("This is to certify that the project work entitled “AI Research Paper Assistant: "
           "A Hybrid GraphRAG Architecture for Citation-Grounded, Multi-Document Scholarly "
           "Question Answering” was carried out by Ch. Charitha, J. Adiseshu, S. Sambhav "
           "and B. Jyothi Naga Sravani under the guidance of Mr. Bhavanam Venkata Suresh "
           "Reddy, Department of Computer Science and Engineering, Koneru Lakshmaiah "
           "Education Foundation, Vijayawada, in partial fulfilment of the requirements "
           "for the degree of Bachelor of Technology. The work reported in this document "
           "is the result of the authors' own investigation and has not been submitted "
           "elsewhere for the award of a degree.")
    r.body("")
    r.body("Project Guide                                        Head of the Department")
    r.body("______________________                              ______________________")

    r.chapter("STUDENT DECLARATION AND ORIGINALITY STATEMENT")
    r.body("We declare that the project report submitted here is our original work carried "
           "out under the supervision of the project guide. All sources consulted during "
           "the study are acknowledged in the reference list. The implementation described "
           "in this report was developed for the rag-papeer-project repository, and the "
           "results reported in Chapter 7 were produced by the evaluation pipeline included "
           "in that repository. No part of this report has been previously submitted for "
           "the award of any degree or diploma.")
    r.body("Ch. Charitha          J. Adiseshu          S. Sambhav          B. Jyothi Naga Sravani")

    r.chapter("ACKNOWLEDGEMENTS")
    r.body("The authors thank the Department of Computer Science and Engineering and the "
           "Department of Computer Science and Information Technology at Koneru Lakshmaiah "
           "Education Foundation for the computing facilities and academic support provided "
           "throughout the project. The authors are grateful to Mr. Bhavanam Venkata Suresh "
           "Reddy for continuous guidance in shaping the architecture, the evaluation "
           "protocol and the technical writing. Finally, the authors acknowledge the open "
           "source community whose libraries made the implementation possible.")

    r.chapter("STRUCTURED ABSTRACT")
    r.body("Scientific literature grows faster than any individual can read. Students and "
           "researchers need a conversational interface that can answer questions across "
           "several papers while showing where every claim came from. This report describes "
           "the AI Research Paper Assistant, a Streamlit application that ingests PDF, text, "
           "Markdown, web and arXiv sources into isolated sessions and builds two linked "
           "representations: a document hierarchy graph and a semantic graph whose entities "
           "and relations carry page and quote provenance. A LangGraph state machine routes "
           "each question to direct answering, tool-based retrieval or web-backed claim "
           "verification. Retrieval combines vector seeds, entity matches, hierarchy "
           "expansion and similarity links; the answer stage reads the complete indexed "
           "corpus. Thirteen research tools return structured analyses that separate "
           "published evidence from generated interpretation. Local guardrails screen "
           "untrusted input before any model call. A DeepEval evaluation on ten synthetic "
           "questions produced contextual precision 0.978, contextual recall 1.000, "
           "contextual relevancy 0.576, answer relevancy 0.991 and faithfulness 0.981, with "
           "five of ten cases passing at the 0.7 threshold. The results show grounded "
           "answering but imprecise context selection, which is the principal direction "
           "for future work.")
    r.body("Keywords: Retrieval-Augmented Generation; GraphRAG; Neo4j; Qdrant; LangGraph; "
           "knowledge graph; provenance; scholarly question answering; DeepEval; prompt injection.")

    r.chapter("PROJECT AT A GLANCE")
    r.table(
        ["Item", "Detail"],
        [
            ["Project title", "AI Research Paper Assistant"],
            ["Domain", "Scholarly document understanding and question answering"],
            ["Primary interface", "Streamlit web application on port 8501"],
            ["Source types", "PDF, TXT, Markdown, web URL, arXiv identifier or title"],
            ["Storage", "Qdrant Cloud vector collections and Neo4j 5.26 two-level graph"],
            ["Orchestration", "LangGraph state machine with a three-route LLM router"],
            ["Models", "gpt-5.4-mini for reasoning; gpt-5-mini for light tasks; text-embedding-3-small"],
            ["Research tools", "13 structured direct-PDF analysis tools"],
            ["Security", "Local injection detection, research classification, SSRF defence"],
            ["Evaluation", "DeepEval with five metrics over ten synthetic golden questions"],
        ],
        "Project summary")
    if "readme_glance" in a:
        r.image(a["readme_glance"],
                "System at a glance: sources, ingestion, chat, research tools and knowledge base")

    r.chapter("TEAM CONTRIBUTIONS AND RESPONSIBLE AI-TOOL DISCLOSURE")
    r.subsection("Team Contributions")
    r.body("The work was divided by subsystem while keeping a shared design review. "
           "Ch. Charitha led the literature study and the design of the two-level graph "
           "schema. J. Adiseshu implemented the LangGraph workflow, the retrieval tools "
           "and the claim-verification path. S. Sambhav implemented the document loader, "
           "chunking logic and vector-store integration, and prepared the evaluation "
           "harness. B. Jyothi Naga Sravani implemented the research intelligence studio, "
           "the knowledge base explorer and the report documentation. All four members "
           "reviewed the integrated system and the final report.")
    r.subsection("Responsible AI-Tool Disclosure")
    r.body("AI assistants were used as supplementary support for code clarification, "
           "debugging and language editing. Every architectural claim in this report was "
           "verified against the project source before inclusion. The generated research "
           "analyses are labelled as AI-generated in the application interface and are "
           "never presented as published evidence.")

    r.toc_marker()

    r.chapter("LIST OF FIGURES")
    for i, cap in enumerate([
        "System context and principal data flow",
        "Ingestion pipeline from source to dual storage",
        "Two-level GraphRAG representation",
        "Graph-expanded retrieval signals",
        "LangGraph routing and answer path",
        "LangGraph state machine as rendered by the project",
        "Guardrail chain for untrusted input",
    ], start=1):
        r.body(f"Figure {i}. {cap}")

    r.chapter("LIST OF TABLES")
    for i, cap in enumerate([
        "Project summary", "Comparison of research paper analysis approaches",
        "Functional requirements and acceptance criteria", "Technology stack and responsibility",
        "Backend module responsibilities", "Verification test cases",
        "Recorded DeepEval results", "Team contributions",
    ], start=1):
        r.body(f"Table {i}. {cap}")

    r.chapter("LIST OF ABBREVIATIONS AND ACRONYMS")
    r.table(["Abbreviation", "Expansion"], [
        ["RAG", "Retrieval-Augmented Generation"],
        ["GraphRAG", "Graph-based Retrieval-Augmented Generation"],
        ["LLM", "Large Language Model"],
        ["SSRF", "Server-Side Request Forgery"],
        ["API", "Application Programming Interface"],
        ["UI", "User Interface"],
        ["PDF", "Portable Document Format"],
        ["HTTP", "Hypertext Transfer Protocol"],
        ["JSON", "JavaScript Object Notation"],
    ], "Abbreviations used in this report")

    # ---- chapter 1 ----------------------------------------------------------
    r.chapter("CHAPTER 1 – PROJECT CONTEXT, PROBLEM DEFINITION AND OBJECTIVES")
    r.section("1.1 Project Identity")
    r.body("The project is an academic research tool named AI Research Paper Assistant. "
           "It is implemented in the repository rag-papeer-project and is delivered as a "
           "local web application. A researcher loads papers into a session and then asks "
           "questions in natural language. The system answers from the indexed documents, "
           "from live web results, or from general knowledge, depending on the question. "
           "The project belongs to the domain of software systems engineering because it "
           "combines a stateful conversational workflow, two persistence backends and a "
           "deterministic safety layer behind one interface.")
    r.section("1.2 Problem Context and Source")
    r.body("A literature review begins with questions that no single document answers. A "
           "student must locate the original method, find its evaluation table, identify "
           "later work that revisited it and compare results reported under different "
           "baselines. Repository search returns documents rather than comparisons. A "
           "general chat assistant can produce a fluent but unsupported summary because it "
           "answers from model parameters. Naive vector retrieval returns isolated chunks "
           "and loses the surrounding experimental setup that gives a finding meaning. "
           "Finally, a claim that was valid at publication time may have been superseded "
           "by newer work, which a corpus-only system cannot detect.")
    r.section("1.3 Problem Statement and Project Goal")
    r.subsection("Problem Statement")
    r.body("How can a conversational system answer questions across a collection of "
           "research papers so that every paper-related claim is traceable to a specific "
           "document, section and page, while also supporting cross-paper comparison, "
           "temporal claim verification and safe handling of untrusted documents?")
    r.subsection("Project Goal")
    r.body("The goal is to design and implement a research assistant that combines "
           "hierarchical document representation, semantic extraction with provenance, "
           "agentic retrieval, evidence-separated analysis tools and local guardrails, "
           "and to evaluate the implemented system with repeatable metrics.")
    r.section("1.4 Scope, Assumptions and Constraints")
    r.body("The system accepts PDF, text, Markdown, web and arXiv sources up to 20 MB and "
           "200 pages. It assumes an operator who configures OpenAI, Tavily and Qdrant "
           "credentials and runs Neo4j locally through Docker. It assumes a trusted "
           "single-operator deployment: there is no login system and no per-user rate "
           "limiter. The shared global corpus is intentionally searchable, so session "
           "separation organises data and conversation context but is not an access-control "
           "boundary.")
    r.section("1.5 Project Objectives and Measurable Criteria")
    r.numbered(1, "Ingest multiple source types into hierarchical chunks with page and section metadata.")
    r.numbered(2, "Build a two-level graph in which every semantic relation stores a verbatim source quote.")
    r.numbered(3, "Implement a routed question workflow with graph, vector and web tools.")
    r.numbered(4, "Provide thirteen research tools whose evidence fields are separated from generated analysis.")
    r.numbered(5, "Screen untrusted content locally before any model call or index write.")
    r.numbered(6, "Evaluate the system with five DeepEval metrics and report the measured values.")
    r.section("1.6 Feasibility, Ethics and Safety Boundary")
    r.body("The design is technically feasible with the available libraries and services, "
           "all of which are declared in pyproject.toml and requirements.txt. Ethically, "
           "the system must not present generated interpretations as published evidence, "
           "must not expose one operator's documents to another through caches, and must "
           "not treat document text as instructions. The safety boundary is a local "
           "heuristic layer that reduces risk without replacing provider safeguards or "
           "human review.")
    r.section("1.7 Report Organisation")
    r.body("Chapter 2 reviews related work and the identified gap. Chapter 3 states "
           "requirements and acceptance criteria. Chapter 4 presents architecture and data "
           "design. Chapter 5 describes implementation. Chapter 6 covers verification, "
           "security and reliability. Chapter 7 reports the evaluation. Chapter 8 concludes "
           "with impact, limitations and future work.")
    r.section("1.8 Stakeholder Analysis")
    r.body("Three stakeholder groups shape the design. Students and early-career researchers "
           "need explanations in familiar language and a fast route from a question to the "
           "passage that supports the answer. Literature reviewers need cross-paper "
           "comparison, contradiction detection and a clear distinction between a published "
           "finding and an AI-generated interpretation. The system operator needs "
           "reproducible configuration, visible ingestion state and an evaluation view that "
           "can be regenerated. These groups share one interface but place different demands "
           "on provenance, depth and operational transparency.")
    r.table(["Stakeholder", "Primary need", "Design response"], [
        ["Student", "Understand a paper quickly", "Section-aware chunks and grounded answers with paper attribution"],
        ["Literature reviewer", "Compare methods across papers", "Shared corpus, research tools and evidence separation"],
        ["Operator", "Reproducible and debuggable system", "Environment template, logs, state inspector and evaluation view"],
    ], "Stakeholder needs and design responses")
    r.section("1.9 Project Schedule and Milestones")
    r.body("The work proceeded in five overlapping phases. The first phase established the "
           "problem statement and the literature base. The second implemented ingestion, "
           "chunking and dual storage. The third implemented routing, retrieval and answer "
           "generation. The fourth added the research tools, guardrails and explorer. The "
           "fifth built the evaluation harness and prepared this report. Each phase ended "
           "with a review gate at which the working system was demonstrated before the next "
           "phase began.")
    r.section("1.10 Risk Register")
    r.table(["Risk", "Likelihood", "Mitigation"], [
        ["Provider API changes", "Medium", "Pin model names in configuration and keep modules isolated"],
        ["Neo4j unavailable", "Medium", "Qdrant fallback and independent write guards"],
        ["Poor PDF extraction", "Medium", "Font heuristics, OCR route and per-paper completeness view"],
        ["Untrusted document injection", "Low-Medium", "Pre-index screening and fenced context"],
        ["Small evaluation sample", "High", "Report results as a historical snapshot with explicit caveats"],
    ], "Principal project risks and mitigations")
    r.section("1.11 Novelty and Contribution")
    r.body("The contribution is not a new retrieval algorithm. It is the integration of "
           "structure-preserving graphs, provenance-annotated extraction, routed retrieval, "
           "evidence-separated analysis and local guards into one reproducible workflow. "
           "Each part is implementable with existing libraries, but the combination addresses "
           "a practical gap: a student can ask a question across many papers and inspect the "
           "chain from answer to claim to passage to page. The evaluation contributes a "
           "measured negative result that identifies context selection as the limiting "
           "factor rather than generation.")
    r.section("1.12 Deliverable Map")
    r.table(["Artefact", "Description"], [
        ["app.py", "Streamlit interface and session bootstrap"],
        ["backend package", "Eleven modules implementing ingestion, storage, retrieval and tools"],
        ["evaluate.py", "Golden generation, RAG invocation and DeepEval scoring"],
        ["goldens.json and eval_results.json", "Cached evaluation inputs and outputs"],
        ["docker-compose.yml and Dockerfile", "Neo4j service and containerised application"],
        ["start and stop scripts", "Windows automation with logging"],
        ["README.md", "Architecture and operations guide"],
        ["This report", "Requirements, design, implementation and evaluation record"],
    ], "Principal deliverables of the project")

    # ---- chapter 2 ----------------------------------------------------------
    r.chapter("CHAPTER 2 – LITERATURE REVIEW, EXISTING SOLUTIONS AND IDENTIFIED GAP")
    r.section("2.1 Review Approach and Evidence Boundary")
    r.body("The review is organised by capability rather than chronology. It covers "
           "retrieval-augmented generation, dense and lexical retrieval, graph-based "
           "retrieval, scientific language models, citation evaluation and prompt "
           "injection. Claims about the implemented system are drawn from the project "
           "source, while background claims are drawn from the references listed at the end.")
    r.section("2.2 Retrieval-Augmented Generation and Language Models")
    r.body("Lewis et al. introduced RAG by coupling a parametric generator with a "
           "non-parametric memory accessed by a dense retriever. The pattern reduces "
           "fabrication and allows the knowledge source to be replaced without retraining. "
           "The project follows this pattern but extends the memory from a flat passage "
           "index to two linked structures. Sentence-BERT supplies efficient sentence "
           "embeddings, and SciBERT demonstrates that domain-specific pretraining helps "
           "scientific terminology, which motivated the choice of a five-category semantic "
           "entity schema.")
    r.section("2.3 Knowledge Graphs and GraphRAG")
    r.body("GraphRAG builds an entity graph and computes community summaries for "
           "query-focused summarisation. It is effective for corpus-wide themes but does "
           "not preserve the original section order of a paper. The project instead keeps "
           "a document hierarchy as Level 1 and a provenance-carrying semantic graph as "
           "Level 2. Because every Level 2 edge points back to a chunk, a graph-derived "
           "fact can be checked against the source page.")
    r.section("2.4 Citation Grounding and Multi-Document Analysis")
    r.body("The ALCE benchmark showed that strong models still produce citations that do "
           "not support their claims. This project addresses the same risk through "
           "structured provenance, a prompt contract that requires paper attribution, and "
           "a research-tool output schema with separate evidence and analysis fields. It "
           "does not implement automatic claim-level entailment scoring; that limitation "
           "is recorded explicitly in Chapter 8.")
    r.section("2.5 Comparison of Existing Approaches")
    r.table(
        ["Approach", "Strength", "Limitation", "Response in this project"],
        [
            ["Vector-only RAG", "Fast paraphrase search", "Loses document context", "Hierarchy expansion"],
            ["Lexical search", "Exact technical terms", "No semantic match", "Entity token seeds"],
            ["Single-level GraphRAG", "Entity neighbourhoods", "No paper structure", "Two-level graph"],
            ["Corpus-only assistant", "Simple to build", "Cannot detect superseded claims", "Web verification path"],
            ["Free-form analysis tool", "Flexible prose", "Mixes evidence and opinion", "Structured evidence tiers"],
        ],
        "Comparison of research paper analysis approaches")
    r.section("2.6 Identified Research Gap and Positioning")
    r.body("The reviewed approaches provide strong individual components, but an accessible "
           "workflow that combines multi-source ingestion, hierarchy-preserving graphs, "
           "provenance-annotated extraction, routed retrieval, claim verification, "
           "evidence-separated tools and local safety screening is not present in the "
           "literature examined. The project positions itself as an integrated, reproducible "
           "implementation of that workflow. The gap statement is deliberately bounded to "
           "the sources reviewed; it does not claim that no existing tool offers any of "
           "these features.")
    r.section("2.7 Detailed Evidence from the Foundations")
    r.body("The RAG foundation established that a generator conditioned on retrieved memory "
           "can be updated by changing the memory rather than the model. In this project the "
           "memory is not a flat list: it is a hierarchy of sections and chunks plus a "
           "semantic layer. This choice changes what retrieval returns. A vector match can be "
           "expanded into its parent section, so an answer about an experiment arrives with "
           "the setup that makes the result interpretable.")
    r.body("Dense retrieval and lexical retrieval fail in complementary ways. Embeddings "
           "handle paraphrase but blur rare tokens; lexical matching preserves exact "
           "terminology but cannot match different wording. The project addresses both "
           "through two seed sets in the same graph tool: vector seeds from the Neo4j index "
           "and entity seeds from token matching against canonical entity names. The full "
           "text indexes exist in the schema but are not queried; the system therefore does "
           "not claim lexical retrieval or rank fusion as implemented features.")
    r.section("2.8 Tool Ecosystem Survey")
    r.body("LangGraph supplies cycles, conditional edges and durable checkpoints, which a "
           "linear chain cannot provide. Streamlit supplies session state and component "
           "embedding without a frontend build. Qdrant supplies managed cosine search with "
           "payload filters; Neo4j supplies graph traversal, provenance and a vector index "
           "in one store. Tavily supplies concise search results suitable for context "
           "blocks. DeepEval supplies five RAG metrics with an LLM judge. Each selection is "
           "visible in the architecture and is justified by a concrete requirement rather "
           "than by popularity.")
    r.section("2.9 Consequences of the Gap for Design")
    r.body("The identified gap leads to three design commitments. First, provenance must be "
           "stored with extraction rather than reconstructed later, because a fact without "
           "its source quote cannot be audited. Second, routing must be explicit because "
           "paper questions, verification requests and general questions need different "
           "strategies. Third, analysis output must be structured so that evidence and "
           "interpretation cannot be conflated. These commitments are realised in Chapters "
           "4 and 5 and evaluated in Chapter 7.")
    r.section("2.10 Evaluation Theory and Security Literature")
    r.body("RAG evaluation separates retrieval quality from answer quality. Retrieval measures "
           "ask whether the relevant passages were found and placed early. Answer measures "
           "ask whether the response addresses the question and remains grounded in the "
           "provided context. The project uses five DeepEval metrics that cover both groups. "
           "Because the metrics are judged by a language model, the evaluation is a "
           "model-based estimate rather than an exact computation; this is why a manually "
           "annotated subset is recommended as future work.")
    r.body("Citation accuracy has a separate literature. The ALCE benchmark showed that "
           "models frequently attach citations that do not entail the claim. The project "
           "responds by carrying provenance through extraction and by requiring paper "
           "attribution in the answer prompt, but it does not implement automatic entailment "
           "scoring. This distinction is important because a system can be faithful to "
           "retrieved text while the retrieved text itself is only loosely relevant to the "
           "question, which is exactly the pattern observed in Chapter 7.")
    r.body("Security research on LLM-integrated applications identifies indirect prompt "
           "injection as a threat in which instructions embedded in external data alter "
           "behaviour. The project addresses this with local pattern screening, hidden "
           "character removal, base64 inspection, research-document classification and "
           "fenced context. Server-side request forgery is addressed by scheme and port "
           "allowlists, DNS resolution and redirect re-screening. These controls are "
           "defensive layers, not proof of safety.")
    r.section("2.11 Comparative Matrix of Graph Approaches")
    r.table(["Dimension", "Single-level entity graph", "Two-level graph in this project"], [
        ["Document structure", "Not represented", "Sections and reading order preserved"],
        ["Entity provenance", "Often node-level", "Edge-level with verbatim quote and page"],
        ["Context expansion", "Neighbour entities", "Sequential, sibling, shared entity and similarity"],
        ["Auditability", "Graph traversal only", "Graph edge traces to source chunk and page"],
        ["Cost", "One extraction pass", "Hierarchy plus extraction pass"],
    ], "Comparison of graph representations")

    # ---- chapter 3 ----------------------------------------------------------
    r.chapter("CHAPTER 3 – REQUIREMENTS SPECIFICATION AND ACCEPTANCE CRITERIA")
    r.section("3.1 Actors and Use Cases")
    r.body("The primary actor is a student or researcher who uploads papers and asks "
           "questions. The secondary actor is an operator who configures credentials, "
           "starts the Neo4j container and monitors the evaluation view. Use cases include "
           "uploading a source, asking a paper question, verifying whether a claim is "
           "current, running a research intelligence tool, exploring the graph and "
           "reviewing evaluation results.")
    r.section("3.2 Functional Requirements")
    r.table(
        ["ID", "Requirement", "Acceptance criterion"],
        [
            ["FR1", "Accept PDF, TXT and Markdown uploads", "File is indexed and appears in the loaded-paper list"],
            ["FR2", "Accept URL and arXiv sources", "Source is fetched through guarded read and stored with a snapshot"],
            ["FR3", "Preserve section hierarchy", "Each chunk carries section path, page and chunk index"],
            ["FR4", "Build two-level graph", "Hierarchy and semantic nodes are visible in the explorer"],
            ["FR5", "Route questions", "Router emits retrieve, verify_claim or direct_answer"],
            ["FR6", "Retrieve with tools", "Graph, vector and web tools return documents to the agent"],
            ["FR7", "Verify claims", "Verdict lists at most three papers with links"],
            ["FR8", "Run 13 research tools", "Each result separates evidence from generated analysis"],
            ["FR9", "Export tool output", "Markdown export reproduces evidence and tables"],
            ["FR10", "Evaluate the system", "DeepEval writes five metric scores per test case"],
        ],
        "Functional requirements and acceptance criteria")
    r.section("3.3 Quality Requirements and Acceptance Criteria")
    r.body("The application must stream answer tokens, restore sessions after restart, "
           "keep the embedded chunk dimension fixed at 1536, and degrade gracefully when "
           "Neo4j is unavailable. The evaluation threshold is 0.7 for all five metrics. "
           "The graph explorer must reject write operations. The research-tool interface "
           "must not place generated text in an evidence field.")
    r.section("3.4 Security and Privacy Requirements")
    r.body("All uploads, URLs, arXiv queries and chat messages must pass local screening "
           "before use. Private network addresses must be rejected before fetch. Document "
           "text must be fenced and declared untrusted. Cached embeddings must be isolated "
           "from untrusted instruction content by scrubbing before embedding.")
    r.section("3.5 Deployment and Operational Requirements")
    r.body("The system must start on Windows through start.ps1, which launches the Neo4j "
           "container, creates the environment file when missing, installs dependencies "
           "with uv and starts Streamlit. A Dockerfile supports remote deployment. Health "
           "checks, logs and a checkpointer database support operational recovery.")
    r.section("3.6 User Stories")
    r.bullet("As a student, I want to upload a paper and ask what its limitations are so that I do not read the whole document manually.")
    r.bullet("As a reviewer, I want a comparison of methods across five papers with evidence for each cell.")
    r.bullet("As a researcher, I want to verify whether an older claim has been superseded and receive links to newer work.")
    r.bullet("As an operator, I want to see which papers are indexed and which stage of ingestion failed.")
    r.bullet("As a user, I want off-topic questions answered without polluting my paper conversation.")
    r.section("3.7 Measurable Quality Requirements")
    r.table(["Quality", "Target", "How it is checked"], [
        ["Grounding", "Faithfulness at or above 0.7", "DeepEval faithfulness metric"],
        ["Relevance", "Answer relevancy at or above 0.7", "DeepEval answer relevancy"],
        ["Context quality", "Contextual relevancy at or above 0.7", "DeepEval contextual relevancy"],
        ["Determinism of IDs", "Same paper and source produce same paper ID", "SHA1-based identifier"],
        ["Vector dimension", "1536 in both stores", "Collection and index configuration"],
        ["Streaming", "Answer tokens appear incrementally", "Message-mode graph streaming"],
        ["Recovery", "Session restored after restart", "JSON records and SQLite checkpoints"],
    ], "Quality requirements and measurement methods")
    r.section("3.8 Requirements Traceability")
    r.body("Each functional requirement maps to a concrete module. Requirements FR1 and FR2 "
           "are satisfied by paper_loader.py and session_store.py. FR3 and FR4 are satisfied "
           "by paper_loader.py and graph_store.py. FR5 through FR7 are satisfied by "
           "rag_graph.py with vector_store.py and graph_store.py as tool backends. FR8 and "
           "FR9 are satisfied by research_tools.py. FR10 is satisfied by evaluate.py. This "
           "traceability makes it possible to identify the module affected when a requirement "
           "changes.")
    r.section("3.9 Data Requirements and Retention")
    r.table(["Data", "Stored in", "Retention", "Sensitivity"], [
        ["Uploaded source bytes", "session_data sources folder", "Until source removed", "Potentially sensitive"],
        ["Text snapshot", "session_data sources folder", "Until source removed", "Potentially sensitive"],
        ["Embeddings", "Qdrant and embedding cache", "Until collection cleared", "Derived from source"],
        ["Graph nodes and edges", "Neo4j", "Until session removed", "Derived from source"],
        ["Conversation state", "SQLite checkpoints", "Until database cleared", "User queries"],
        ["Session metadata", "sessions.json", "Application lifetime", "Display names"],
        ["Evaluation results", "eval_results.json", "Until regenerated", "Questions and answers"],
    ], "Data categories and retention boundaries")
    r.section("3.10 Acceptance Test Catalogue")
    r.table(["ID", "Scenario", "Acceptance condition"], [
        ["A1", "Upload a valid paper", "Loaded list shows title and chunk count"],
        ["A2", "Ask a paper question", "Answer names at least one supporting paper"],
        ["A3", "Ask a verification question", "Verdict includes source links"],
        ["A4", "Run a research tool", "Evidence and analysis fields both populated"],
        ["A5", "Open graph explorer", "Nodes and provenance table render"],
        ["A6", "Restart application", "Previous session and chats restored"],
        ["A7", "Run evaluation", "eval_results.json contains five scores per case"],
    ], "Acceptance scenarios for the delivered system")
    r.section("3.11 Environment Dependencies")
    r.table(["Dependency", "Purpose", "Required"], [
        ["Python 3.12", "Runtime for the application", "Yes"],
        ["uv", "Dependency and environment manager", "Yes"],
        ["OpenAI API key", "LLM inference and embeddings", "Yes"],
        ["Tavily API key", "Web search and claim verification", "Yes"],
        ["Qdrant Cloud", "Managed vector collections", "Yes"],
        ["Neo4j 5.26", "Two-level graph and provenance store", "Yes for graph features"],
        ["Docker Compose", "Local Neo4j container", "Recommended"],
        ["DeepEval", "Evaluation metrics and judges", "For evaluation only"],
        ["PyMuPDF", "PDF parsing and table extraction", "Yes"],
    ], "Runtime dependencies and their purposes")

    # ---- chapter 4 ----------------------------------------------------------
    r.chapter("CHAPTER 4 – SYSTEM ARCHITECTURE, DATA DESIGN AND ENGINEERING DECISIONS")
    r.section("4.1 Architecture Alternatives")
    r.body("Three designs were considered. A single vector index is simple but loses "
           "document structure. A single-level entity graph supports thematic retrieval "
           "but not section context. A two-level graph with a complementary vector index "
           "preserves structure and semantic relations while keeping fast similarity "
           "search. The third design was adopted and is implemented.")
    r.section("4.2 System Context and Component Architecture")
    if "readme_architecture" in a:
        r.image(a["readme_architecture"],
                "High-level component architecture rendered from the project README")
    if "fig_context.tex" in a:
        r.image(a["fig_context.tex"], "System context and principal data flow")
    r.body("The interface layer never writes directly to a store. The orchestration layer "
           "owns all state transitions. Storage backends are substitutable, and external "
           "intelligence is reached only through tools or guarded prompts. This separation "
           "makes it possible to disable Neo4j and continue with Qdrant alone.")
    r.section("4.3 Document Ingestion and Dual Storage")
    if "fig_ingest.tex" in a:
        r.image(a["fig_ingest.tex"], "Ingestion pipeline from source to dual storage")
    r.body("Ingestion screens a source before saving it, extracts section boundaries before "
           "splitting text, embeds through a shared cache, and writes to Qdrant and Neo4j "
           "in independent guarded operations. A failure in one store does not discard the "
           "other. The same 1536-dimensional vector is used in both stores, so similarity "
           "behaviour is consistent.")
    r.section("4.4 Two-Level Knowledge Representation")
    if "fig_graph.tex" in a:
        r.image(a["fig_graph.tex"], "Two-level GraphRAG representation")
    r.body("Level 1 connects Paper, Section, Subsection and Chunk nodes with HAS_SECTION, "
           "HAS_CHUNK and NEXT_CHUNK edges. Level 2 connects five entity types through "
           "MENTIONS and RELATED_TO edges. Each provenance-bearing edge stores the paper "
           "title, section path, page, chunk index and a verbatim evidence quote. "
           "Deterministic identifiers make ingestion idempotent through MERGE operations.")
    r.section("4.5 Hybrid Retrieval Signals")
    if "fig_retrieval.tex" in a:
        r.image(a["fig_retrieval.tex"], "Graph-expanded retrieval signals")
    r.body("Hybrid retrieval in this project means combining vector similarity, entity "
           "token matching, hierarchy adjacency and similarity links in one graph tool. "
           "It does not mean combining BM25 with reciprocal rank fusion; those methods are "
           "not implemented. The project also does not use a cross-encoder reranker. "
           "Ranking comes from vector scores, entity match counts and similarity scores, "
           "followed by an advisory LLM relevancy decision.")
    r.body("The cosine similarity used by both stores is:")
    if "eq_cosine.tex" in a:
        r.image(a["eq_cosine.tex"], "Cosine similarity over query and chunk embeddings", 9)
    r.body("Similarity links are created only above a configured threshold and are capped "
           "per chunk to avoid a dense link web:")
    if "eq_simlink.tex" in a:
        r.image(a["eq_simlink.tex"], "Similarity-link criterion and degree cap", 11)
    r.body("The context budget is bounded by:")
    if "eq_budget.tex" in a:
        r.image(a["eq_budget.tex"], "Context and vector seed budgets", 11)
    r.section("4.6 Answer Generation and Evidence Contract")
    r.body("The answer node reads the full indexed corpus under a character cap and treats "
           "retrieved passages as hints. The prompt requires paper attribution and an "
           "explicit statement when evidence is insufficient. The thirteen research tools "
           "use a structured schema with separate fields for published evidence, tables, "
           "AI analysis, AI ideas and grounding notes, so generated material cannot be "
           "silently presented as published evidence.")
    r.section("4.7 Architecture Decisions and Trade-offs")
    r.body("Neo4j is queried through its HTTP transaction endpoint because the Bolt driver "
           "is not a declared dependency. The relevancy decision is recorded but does not "
           "block an answer, because the full corpus remains available. Conversation state "
           "is stored twice, once as JSON for display and once as LangGraph checkpoints "
           "for execution. Each choice improves robustness at the cost of some duplication.")
    r.section("4.8 Deployment Architecture")
    r.body("The local deployment runs the Streamlit application on the host and Neo4j in a "
           "container with fixed ports 7476 for HTTP and 7689 for Bolt. Qdrant Cloud and the "
           "model APIs are reached over HTTPS with credentials from the environment file. "
           "A containerised profile is declared in docker-compose.yml for hosts that require "
           "a single deployment unit. Because credentials are process-wide, the deployment "
           "model is single-operator.")
    r.section("4.9 State Lifecycle and Data Flow")
    r.body("A source record is created first, then a text snapshot, then vectors in Qdrant "
           "and hierarchy plus entities in Neo4j. A query creates a session-scoped checkpoint "
           "thread. Retrieved documents are appended to graph state by tool commands, never "
           "by the tools writing directly to storage. The answer node reads the corpus and "
           "the accumulated state, then writes the final response to the checkpoint. This "
           "separation keeps persistence deterministic.")
    r.section("4.10 Design Trade-off Summary")
    r.table(["Decision", "Benefit", "Cost"], [
        ["Two stores", "Fast vector search plus graph provenance", "Dual-write coordination"],
        ["Full-corpus answering", "Weak retrieval cannot block an answer", "Context cap and prompt length"],
        ["HTTP graph access", "No extra driver dependency", "Bolt fallback only for secure schemes"],
        ["Shared global corpus", "Cross-session research awareness", "Answer isolation not guaranteed"],
        ["Advisory relevancy", "No false negative gate", "Relevancy cannot trigger rewrite"],
        ["Structured tools", "Evidence and analysis separated", "Prompt and schema maintenance"],
    ], "Principal architectural trade-offs")
    r.section("4.11 Detailed Graph Schema")
    r.table(["Element", "Key properties", "Purpose"], [
        ["Session", "id", "Root scope for all stored research data"],
        ["Paper", "id, session_id, title, source, doc_type, num_chunks", "Document identity and metadata"],
        ["Section", "id, paper_id, title, path, level, order, page", "Document hierarchy position"],
        ["Chunk", "id, paper_id, section_id, text, embedding, chunk_index, page", "Retrieval unit and vector carrier"],
        ["Entity", "id, canonical_name, name, type, description", "Level 2 semantic node"],
        ["HAS_SECTION", "Paper to Section", "Top level of document hierarchy"],
        ["HAS_CHUNK", "Section to Chunk", "Section ownership of chunks"],
        ["NEXT_CHUNK", "Chunk to Chunk", "Reading order"],
        ["SIMILAR_TO", "score", "Semantic similarity above threshold"],
        ["MENTIONS", "paper_title, section_path, page, quote", "Chunk to entity evidence"],
        ["RELATED_TO", "relation, description, quote", "Entity to entity relation"],
    ], "Graph node labels and relationship properties")
    r.section("4.12 Ingestion Lifecycle States")
    r.body("The interface exposes a visible lifecycle for every uploaded paper. The states "
           "move from uploaded through validating, extracting, OCR where needed, structure "
           "analysis, chunking, embedding, knowledge extraction, graph building, indexing and "
           "summarising to completed or failed. A user can therefore see which part of a "
           "paper is missing before relying on an answer. Completeness tracking is a "
           "practical requirement because a failed PDF page silently reduces recall.")
    r.section("4.13 Retrieval Cost and Boundaries")
    r.body("Retrieval cost is bounded at three points. The vector seed pool is capped at "
           "max(8k,40), the entity seed count at max(2, floor(k/2)), and the final context at "
           "max(3k,10) chunks. Graph expansion traverses at most six sequential neighbours, "
           "four section siblings, four shared-entity chunks and three similarity links per "
           "seed. These bounds prevent an unbounded graph walk from filling the context "
           "window and are visible in the graph state inspector as retrieval reasons.")
    r.section("4.14 Caching and Deduplication")
    r.body("The embedding layer uses a BLAKE2b key encoder over a local file store with query "
           "caching enabled. Identical text therefore produces one provider call across all "
           "sessions and restarts. Paper identifiers are derived from a SHA1 of the title and "
           "source, and chunk identifiers combine that paper identifier with a chunk index. "
           "Neo4j writes use MERGE so repeated ingestion updates rather than duplicates the "
           "graph. These mechanisms keep the two representations consistent.")
    r.section("4.15 Index Structures and Query Patterns")
    r.table(["Index", "Store", "Query pattern"], [
        ["Cosine vector index", "Qdrant", "Similarity search over session and global collections"],
        ["Cosine vector index", "Neo4j", "Approximate nearest neighbours over session chunks"],
        ["Full-text chunk index", "Neo4j", "Created in schema but not queried by the application"],
        ["Full-text entity index", "Neo4j", "Created in schema but not queried by the application"],
        ["Uniqueness constraints", "Neo4j", "Idempotent MERGE of session, paper, section, chunk and entity"],
    ], "Indexes created by the system and their actual use")
    r.body("The query patterns are deliberately narrow. Vector search is used for seeds, "
           "entity matching uses token containment against canonical names, and structural "
           "expansion uses fixed traversals. This keeps behaviour predictable and auditable. "
           "The presence of unused full-text indexes is recorded so that the schema is not "
           "mistaken for evidence that lexical retrieval is implemented.")

    # ---- chapter 5 ----------------------------------------------------------
    r.chapter("CHAPTER 5 – IMPLEMENTATION")
    r.section("5.1 Implementation Overview")
    r.body("The application is a Python 3.12 project served by Streamlit. Dependencies are "
           "managed with uv and recorded in uv.lock. The backend is a package of eleven "
           "modules. Docker Compose starts Neo4j 5.26 on ports 7476 and 7689, while Qdrant "
           "Cloud and the model APIs are external services configured through the environment file.")
    r.section("5.2 Routing and State Machine")
    if "readme_router" in a:
        r.image(a["readme_router"],
                "LangGraph routing state machine rendered from the project README")
    if "fig_router.tex" in a:
        r.image(a["fig_router.tex"], "LangGraph routing and answer path")
    r.body("The LangGraph state carries messages, session identifier, query, route, retrieved "
           "documents, attempt counter, relevancy decision, claim verdict and final answer. "
           "An intake guard screens the question before the router runs. The retrieval "
           "agent may call tools for at most three turns; at the cap it is replaced by a "
           "plain model call so that no orphaned tool call reaches the checkpoint. The "
           "verification route bypasses the retrieval agent and issues two Tavily searches.")
    if "fig_state.png" in a:
        r.image(a["fig_state.png"], "LangGraph state machine as rendered by the project", 9)
    r.body("The rendered graph shows a query_rewrite node that exists in source but has no "
           "reachable incoming edge. The rewrite loop described in early documentation is "
           "therefore inert at runtime; this report records the executed behaviour.")
    r.section("5.3 Document Ingestion and Preprocessing")
    r.body("The loader detects sections from PDF font sizes, Markdown headings, numbered "
           "text headings or HTML headings. Each section is split with 1000-character chunks "
           "and 200-character overlap. The chunking rule is:")
    if "eq_chunk.tex" in a:
        r.image(a["eq_chunk.tex"], "Section-bounded chunking with overlap", 12)
    r.body("Every chunk receives a paper identifier, section path, page, chunk index and "
           "deterministic chunk identifier. Titles can be restamped after a temporary upload "
           "receives its real filename.")
    r.section("5.4 Knowledge Graph Construction")
    r.body("Entity extraction batches three chunks and runs with five workers. A structured "
           "prompt limits output to Concept, Method, Dataset, Metric and Finding entities "
           "with at most twelve entities and twelve relations per batch. Names are "
           "canonicalised, relation labels are normalised to upper snake case, and every "
           "extracted item must cite chunk indices and a verbatim quote.")
    r.section("5.5 Hybrid Retrieval and Answer Generation")
    r.body("The graph retriever runs a Neo4j vector index query with an in-memory cosine "
           "fallback, matches query tokens against entity names, merges the seeds, expands "
           "through four neighbourhoods and appends supporting triples. Qdrant search and "
           "full-corpus text provide fallbacks. The answer node receives the full corpus "
           "with retrieved passages prepended as hints.")
    if "readme_sequence" in a:
        r.image(a["readme_sequence"],
                "Question turn sequence rendered from the project README")
    r.section("5.6 Research Intelligence Studio")
    r.body("Thirteen tools cover summarisation, abstract extraction, methodology, algorithms "
           "and equations, datasets and metrics, results, contributions and limitations, "
           "research gaps, comparison, synthesis, contradictions, literature-review drafts "
           "and research questions. Each tool binds a structured output model and receives "
           "either extracted page text or, when clean and small enough, the raw PDF as a "
           "file block.")
    r.section("5.7 Backend Module Responsibilities")
    r.table(
        ["Module", "Responsibility"],
        [
            ["app.py", "Streamlit interface, sessions, three tabs, explorer"],
            ["rag_graph.py", "LangGraph state, tools, routing, checkpointer"],
            ["graph_store.py", "Neo4j schema, ingestion, retrieval, explorer queries"],
            ["vector_store.py", "Qdrant collections, cached embeddings, corpus loading"],
            ["paper_loader.py", "Source loaders, section parsers, hierarchical chunks"],
            ["content_guard.py", "Screening, scrubbing, fenced untrusted text, guarded fetch"],
            ["research_tools.py", "Thirteen research tool specifications and execution"],
            ["models.py", "Pydantic schemas for routing and extraction"],
            ["session_store.py", "Session records, sources, chats and tool history"],
            ["library.py", "Source indexing and shared corpus maintenance"],
            ["btw_handler.py", "Off-topic side channel, excluded from history"],
            ["graph_visualizer.py", "Interactive HTML5 canvas graph viewer"],
        ],
        "Backend module responsibilities")
    r.section("5.8 Structured Contracts and Validation")
    r.body("All model-produced decisions are constrained by Pydantic schemas. The router can "
           "only emit one of three route values. The relevancy judge returns a boolean and a "
           "reason. Claim verification returns a verdict, a summary and at most three papers. "
           "Knowledge extraction returns entities and relations limited to five categories, "
           "each with chunk indices and a verbatim quote. Structured output turns what would "
           "otherwise be brittle prompt parsing into a typed interface.")
    r.section("5.9 Configuration Management")
    r.body("Configuration is split into code constants and environment values. Chunk size, "
           "overlap, similarity threshold and attempt caps are code constants because "
           "changing them changes the behaviour being evaluated. Provider keys, endpoints "
           "and Neo4j credentials are environment values because they differ between "
           "deployments. The .env.example file documents every required variable without "
           "exposing secrets. The embedding cache namespace is the model name, so changing "
           "models naturally produces a new cache space.")
    r.section("5.10 Development and Debugging Workflow")
    r.body("The graph state inspector exposes a compact JSON view of each turn after the "
           "answer is produced. Messages and documents are truncated for display so that "
           "the view remains readable. This inspector, together with the ingestion "
           "lifecycle labels and the knowledge base statistics, is the principal debugging "
           "surface. Developers can see which retriever contributed a chunk, which page it "
           "came from and whether the graph or vector store supplied it.")
    r.section("5.11 Research Tool Catalogue")
    r.table(["Tool", "Question it answers"], [
        ["Paper summarisation", "What is this paper about?"],
        ["Abstract extraction", "What does the abstract claim?"],
        ["Methodology explanation", "How was the study conducted?"],
        ["Algorithm and equation explanation", "What does this algorithm or equation do?"],
        ["Dataset and metric extraction", "Which datasets and metrics were used?"],
        ["Results analysis", "What do the reported results show?"],
        ["Contributions, limitations and future work", "What did the authors claim and concede?"],
        ["Research gap discovery", "What questions remain open?"],
        ["Paper comparison", "How do these papers differ?"],
        ["Cross-paper synthesis", "What is the combined picture?"],
        ["Contradiction detection", "Where do the sources disagree?"],
        ["Literature review draft", "What does a draft review section look like?"],
        ["Research question generation", "What could be studied next?"],
    ], "The thirteen research intelligence tools")
    r.body("Each tool binds the same structured output contract. The evidence tier in the "
           "result indicates whether the output came from saved sources, the knowledge graph "
           "or both. Evidence items carry a paper title, a section or page, a verbatim quote "
           "and a statement. AI analysis and ideas occupy separate fields, and grounding "
           "notes explain the limits of the evidence base.")
    r.section("5.12 Deployment Operations")
    r.body("The Windows start script raises the Neo4j service, waits for its endpoint, "
           "creates the environment file when missing, installs dependencies when the "
           "virtual environment is absent and starts Streamlit with logging. The stop "
           "script verifies process ownership before terminating anything. A Dockerfile "
           "packages the application with Python 3.12 and a layered dependency install, and "
           "the Compose file declares both a Neo4j service and an optional application "
           "profile. This arrangement supports local development and a straightforward "
           "remote deployment without changing application code.")
    r.section("5.13 Research Tool Execution Pipeline")
    r.body("A tool run begins with source selection. Saved sources are converted into fenced "
           "text sections that include page numbers and detected tables. When the knowledge "
           "base option is enabled, hierarchy and provenance records are appended. When raw "
           "PDF attachment is enabled and the documents pass screening, the file bytes are "
           "attached as a file block. The structured model then fills the evidence, table, "
           "analysis, idea and grounding fields. If the file block fails, the run retries in "
           "text-only mode so that the user still receives a result.")

    # ---- chapter 6 ----------------------------------------------------------
    r.chapter("CHAPTER 6 – VERIFICATION, VALIDATION, SECURITY AND OPERATIONAL RELIABILITY")
    r.section("6.1 Verification and Validation Approach")
    r.body("Verification asks whether the implementation matches its specification; "
           "validation asks whether the system answers useful questions. The project uses "
           "local unit tests for the content guard, mocked two-level ingestion and "
           "retrieval tests, a DeepEval harness for answer quality, and a manual review of "
           "the knowledge base explorer. Tests that require provider credentials are mocked "
           "so that the suite can run locally.")
    r.section("6.2 Guardrail Chain and Threat Analysis")
    if "fig_security.tex" in a:
        r.image(a["fig_security.tex"], "Guardrail chain for untrusted input")
    r.table(
        ["Threat", "Implemented control", "Residual risk"],
        [
            ["Prompt injection", "Pattern screening, scrubbing, fenced context, no tools in answer state", "Novel phrasing can pass"],
            ["SSRF", "Scheme and port allowlist, DNS resolution, redirect re-screening", "DNS rebinding is not modelled"],
            ["Non-research upload", "Research-structure classifier and rejection", "Unusual short papers may be rejected"],
            ["Oversized input", "20 MB, 200 pages, 8000 character limits", "None within the stated limits"],
            ["Graph mutation via UI", "Read-only Cypher keyword filter", "Keyword filter is not a parser"],
            ["Cross-session leakage", "Session collections and graph scopes", "Global corpus is intentionally shared"],
        ],
        "Threat analysis and implemented controls")
    r.section("6.3 Robustness and Failure Handling")
    r.body("Retrieval degrades from graph to vector to full corpus. Web fetch falls back to "
           "a stored text snapshot. Research tools retry in text-only mode when a raw PDF "
           "file block fails. Graph writes and vector writes capture errors independently. "
           "The agent disables tools at the attempt cap to keep checkpoint history valid.")
    r.section("6.4 Operational Reliability")
    r.body("Streamlit caches the compiled graph with a token derived from source modification "
           "times. SQLite checkpoints and JSON records restore sessions after restart. "
           "Start and stop scripts log every step and stop only processes belonging to this "
           "project. Evaluation runs with three concurrent workers and a five-second "
           "throttle to stay within provider limits.")
    r.section("6.5 Residual Risks")
    r.body("The system has no authentication, no per-user rate limiting and no automated "
           "claim entailment validator. The shared global corpus is deliberately searched "
           "with the session corpus, so answer-level isolation is not guaranteed. The "
           "relevancy decision is advisory and the query-rewrite node is unreachable. "
           "These items are carried forward as limitations.")
    r.section("6.6 Detailed Verification Catalogue")
    r.table(["ID", "Check", "Method", "Expected result"], [
        ["V1", "Upload valid PDF", "Manual and unit test", "Chunks with page and section metadata"],
        ["V2", "Upload non-research file", "Content guard test", "Rejected before storage"],
        ["V3", "Upload injection text", "Pattern test", "Rejected or neutralised"],
        ["V4", "Fetch private URL", "SSRF test", "Rejected before network call"],
        ["V5", "Index into Neo4j", "Mocked graph test", "Hierarchy and entities created"],
        ["V6", "Query with empty graph", "Fallback test", "Qdrant or full corpus used"],
        ["V7", "Three tool calls", "Agent loop test", "Tools disabled at cap"],
        ["V8", "Read-only Cypher", "Keyword filter test", "Write query rejected"],
        ["V9", "Restart session", "Persistence test", "Chat restored from records"],
        ["V10", "Run evaluation", "DeepEval harness", "Five scores per case written"],
    ], "Verification catalogue with expected results")
    r.section("6.7 Security Review Observations")
    r.body("The guard chain is deterministic and runs before any provider call. However, it "
           "is a heuristic layer. Pattern lists cannot enumerate every override phrase, and "
           "a research-looking document with a novel instruction could pass. The URL guard "
           "resolves DNS before fetch, which mitigates but does not eliminate rebinding. "
           "The read-only console blocks write keywords but is not a Cypher parser. These "
           "observations define the boundary of the trust model rather than a claim of "
           "complete protection.")
    r.section("6.8 Operational Runbook")
    r.bullet("Start the Neo4j container with docker compose up -d neo4j and confirm the browser endpoint responds.")
    r.bullet("Copy .env.example to .env and fill provider keys; never commit the filled file.")
    r.bullet("Run uv sync, then start the application with the project start script.")
    r.bullet("Check the ingestion lifecycle labels in the sidebar before relying on answers from a new paper.")
    r.bullet("Inspect the graph state JSON when an answer is unexpected; it shows retrieval reasons.")
    r.bullet("Delete goldens.json and rerun evaluate.py when a fresh evaluation is required.")
    r.bullet("Review logs for failed jobs and restart only processes owned by this project.")
    r.section("6.9 Failure Mode and Effect Analysis")
    r.table(["Failure mode", "Effect", "Detection", "Recovery"], [
        ["PDF page extraction fails", "Missing chunks from that page", "Completeness labels", "OCR route and re-index"],
        ["Vector index unreachable", "No vector seeds", "Tool fallback chain", "Graph or full corpus"],
        ["Graph extraction malformed", "Missing entity nodes", "Provenance view", "Re-run extraction batch"],
        ["Provider rate limit", "Failed model call", "Error in chat and logs", "Retry after interval"],
        ["Checkpoint corruption", "Conversation history lost", "State inspector", "Rebuild from JSON record"],
        ["Injection text indexed", "Untrusted content in context", "Screen and fence", "Remove source and re-index"],
    ], "Failure modes, detection and recovery")
    r.section("6.10 Security Test Scenarios")
    r.table(["Scenario", "Input", "Expected behaviour"], [
        ["Role override", "Message containing an instruction to ignore prior rules", "Rejected before routing"],
        ["Hidden payload", "Zero-width characters in text", "Characters removed before screening"],
        ["Encoded payload", "Long base64 span in a document", "Decoded and screened again"],
        ["Private fetch", "URL resolving to a loopback address", "Rejected before request"],
        ["Redirect bypass", "Public URL redirecting to private host", "Redirect rejected"],
        ["Non-research upload", "Invoice or CV", "Rejected by structure classifier"],
        ["Graph write attempt", "Cypher containing a write keyword", "Rejected by console filter"],
    ], "Security scenarios exercised during verification")
    r.section("6.11 Validation Summary")
    r.table(["Validation goal", "Evidence", "Outcome"], [
        ["Answers are grounded", "Faithfulness metric", "Mean 0.981, all cases passed"],
        ["Answers address questions", "Answer relevancy", "Mean 0.991, all cases passed"],
        ["Evidence is recoverable", "Contextual recall", "1.000 in all cases"],
        ["Context is precise", "Contextual relevancy", "Mean 0.576, five cases failed"],
        ["System survives backend loss", "Fallback chain review", "Graph to vector to corpus"],
        ["Sessions survive restart", "Persistence review", "JSON and SQLite records"],
    ], "Validation goals and recorded evidence")

    # ---- chapter 7 ----------------------------------------------------------
    r.chapter("CHAPTER 7 – RESULTS, PERFORMANCE EVALUATION AND DISCUSSION")
    r.section("7.1 Evaluation Environment and Protocol")
    r.body("The evaluation pipeline in evaluate.py generates ten synthetic question and "
           "answer pairs from documents/Openclaw_Research_Report.pdf using DeepEval's "
           "synthesizer with five contexts and two questions per context. Each question is "
           "answered in a fresh session after the document is ingested. Five metrics are "
           "computed with gpt-5.4-mini as judge at a threshold of 0.7. A case passes only "
           "when all five metrics pass.")
    r.section("7.2 Metric Definitions")
    r.body("Contextual precision measures whether relevant chunks are ranked ahead of "
           "irrelevant chunks. Contextual recall measures whether the retrieved context "
           "covers the expected answer. Contextual relevancy asks whether the context is "
           "pertinent to the question and expected answer. Answer relevancy asks whether "
           "the response addresses the question. Faithfulness asks whether the response "
           "is grounded in the retrieved context.")
    r.body("The common precision and recall diagnostics are:")
    if "eq_metrics.tex" in a:
        r.image(a["eq_metrics.tex"], "Precision, recall and case-level pass criterion", 13)
    r.section("7.3 Recorded Results")
    r.table(
        ["Metric", "Mean", "Minimum", "Maximum", "Passed"],
        [
            ["Contextual Precision", "0.978", "0.833", "1.000", "10/10"],
            ["Contextual Recall", "1.000", "1.000", "1.000", "10/10"],
            ["Contextual Relevancy", "0.576", "0.094", "0.974", "5/10"],
            ["Answer Relevancy", "0.991", "0.909", "1.000", "10/10"],
            ["Faithfulness", "0.981", "0.938", "1.000", "10/10"],
        ],
        "Recorded DeepEval results over ten cases")
    r.body("Five of ten cases passed overall, and forty-five of fifty metric evaluations "
           "passed. Every failure belonged to contextual relevancy. The weak cases retrieved "
           "chunks that were topically related to the report but not pertinent to the "
           "specific question, which is consistent with a deliberate breadth-first context "
           "expansion stage.")
    r.section("7.4 Discussion")
    r.body("The measured strength is grounded answering: faithfulness and answer relevancy "
           "were near one and no case was judged ungrounded. The measured weakness is "
           "context selection rather than generation. Because the answer stage reads the "
           "complete corpus, irrelevant retrieved chunks do not necessarily damage the "
           "answer, but they do reduce the contextual relevancy score. The evaluation does "
           "not decide the correct expansion setting because the goldens do not encode a "
           "preferred answer length.")
    r.section("7.5 Threats to Validity")
    r.bullet("Ten synthetic questions from one document provide little statistical power.")
    r.bullet("The judge model is the same model used to generate answers.")
    r.bullet("The application received the question with an appended clause while the judge saw the bare question.")
    r.bullet("No latency, token count or cost values were recorded, so no efficiency claim is made.")
    r.bullet("Retrieved contexts were not persisted with the scores, limiting per-chunk auditing.")
    r.bullet("The run was single, unseeded and asynchronous with three workers.")
    r.section("7.6 Visual Analysis of the Recorded Results")
    r.body("Figure 7.1 shows the mean score of each metric. Four bars cluster near one, while "
           "contextual relevancy is visibly below the 0.7 threshold drawn as a dashed line. "
           "The chart uses only the values stored in eval_results.json; no estimated or "
           "interpolated values are shown.")
    if "chart_metric_means.tex" in a:
        r.image(a["chart_metric_means.tex"], "Mean DeepEval scores with the 0.7 pass threshold")
    r.body("Figure 7.2 traces contextual relevancy across the ten cases. The spread is wide, "
           "from 0.094 to 0.974, and five cases fall below the threshold. The cases below the "
           "line are those where the retrieved context contained material related to the "
           "report but not to the specific question.")
    if "chart_cre_trend.tex" in a:
        r.image(a["chart_cre_trend.tex"], "Contextual relevancy by test case")
    r.body("Figure 7.3 shows how many cases passed each metric. Four metrics passed in all "
           "ten cases; contextual relevancy passed in five. Because a case succeeds only "
           "when all five metrics pass, this single metric determines the overall pass rate.")
    if "chart_pass_rates.tex" in a:
        r.image(a["chart_pass_rates.tex"], "Cases passing each metric out of ten")
    r.section("7.7 Per-Case Interpretation")
    r.body("The lowest contextual relevancy score, 0.094, occurred for the question about "
           "supported model backends. The retrieved context contained relevant material but "
           "also unrelated configuration detail, which reduced the judge's relevance score. "
           "The memory question scored 0.269 and the credential-storage comparison 0.362 for "
           "the same reason. These are breadth effects of the expansion stage rather than "
           "failures of answer generation.")
    r.body("The highest contextual relevancy, 0.974, occurred for the broad security-risk "
           "question, where nearly all retrieved passages were pertinent. Intermediate "
           "scores around 0.73 to 0.83 occurred for questions whose answers were spread "
           "across several sections. The pattern suggests that contextual relevancy depends "
           "on how narrowly the question constrains the relevant context.")
    r.section("7.8 Interpretation for Stakeholders")
    r.body("For a student, the measured faithfulness of 0.981 means that answers were almost "
           "always grounded in retrieved text. For a reviewer, the contextual recall of "
           "1.000 means the expected information was present in the context, although not "
           "always alone. For an operator, the contextual relevancy result is a signal to "
           "tune expansion rather than to replace the generator. These interpretations are "
           "bounded by the small synthetic test collection and must not be generalised to "
           "other corpora without a new evaluation.")
    r.section("7.9 Per-Case Score Record")
    r.table(["Case", "CP", "CR", "CRe", "AR", "F", "Result"], [
        ["1 Model backends", "0.833", "1.000", "0.094", "1.000", "1.000", "Fail"],
        ["2 Session memory", "0.950", "1.000", "0.269", "1.000", "1.000", "Fail"],
        ["3 Channels", "1.000", "1.000", "0.500", "0.909", "1.000", "Fail"],
        ["4 Risks", "1.000", "1.000", "0.974", "1.000", "0.938", "Pass"],
        ["5 Local versus VPS", "1.000", "1.000", "0.826", "1.000", "0.964", "Pass"],
        ["6 Delegated execution", "1.000", "1.000", "0.791", "1.000", "0.944", "Pass"],
        ["7 Overall rating", "1.000", "1.000", "0.727", "1.000", "1.000", "Pass"],
        ["8 Agentic workflows", "1.000", "1.000", "0.441", "1.000", "1.000", "Fail"],
        ["9 Enterprise viability", "1.000", "1.000", "0.772", "1.000", "0.963", "Pass"],
        ["10 Credential storage", "1.000", "1.000", "0.362", "1.000", "1.000", "Fail"],
    ], "Per-case DeepEval scores recorded in eval_results.json")
    r.body("The table shows that contextual precision, contextual recall, answer relevancy "
           "and faithfulness were stable across all ten cases. Variability was confined to "
           "contextual relevancy, and the five failing cases are exactly the five with a "
           "contextual relevancy score below 0.7.")
    r.section("7.10 Baseline Discussion")
    r.body("A dense-only baseline would retrieve the top vector chunks and answer from them. "
           "The implemented system adds entity seeds, hierarchy expansion, similarity links "
           "and a web fallback before the answer stage. The recorded evaluation does not "
           "contain an ablation that isolates each component, so this report makes no "
           "comparative claim about the contribution of each signal. An ablation is included "
           "in the future work: vector only, vector plus entity seeds, plus hierarchy, plus "
           "similarity links, measured against the same goldens with latency recorded.")
    r.section("7.11 Evaluation Artifact Schema")
    r.table(["Artifact", "Fields", "Purpose"], [
        ["goldens.json", "input, expected_output", "Cached synthetic test questions"],
        ["eval_results.json", "input, actual_output, success, metrics", "Persisted evaluation run"],
        ["metric entry", "name, score, passed, reason", "Score and judge explanation"],
        ["metrics", "Five named metrics", "Context precision, recall, relevancy, answer relevancy, faithfulness"],
    ], "Structure of the evaluation artifacts")
    r.section("7.12 Interplay Between Metrics")
    r.body("The five metrics are not independent. Contextual recall and contextual relevancy "
           "both depend on the retrieved set, but they reward different properties. A broad "
           "context raises recall because more expected information is present, while the "
           "same breadth lowers relevancy because more items are not pertinent. The recorded "
           "results show exactly this tension: recall is 1.000 in every case, while "
           "relevancy averages 0.576. The system is therefore not failing to find evidence; "
           "it is including too much alongside the evidence.")
    r.body("Answer relevancy and faithfulness depend on generation rather than on the "
           "retrieved set alone. Both remained high, which indicates that the answer node "
           "can identify the useful passages inside a mixed context. This explains why an "
           "overall pass rate of 0.50 does not imply that half the answers were poor. The "
           "pass criterion is strict: a case fails when any one metric falls below 0.7, even "
           "when the answer itself is correct and grounded.")
    r.table(["Metric pair", "Observed relation", "Interpretation"], [
        ["Recall and relevancy", "1.000 versus 0.576", "Breadth helps coverage but dilutes precision"],
        ["Relevancy and pass rate", "Five failures, five low relevancy", "One metric controls case success"],
        ["Faithfulness and answer relevancy", "0.981 and 0.991", "Generation is the stable component"],
    ], "Relationships among the recorded metrics")
    r.section("7.13 Limits of the Presented Graphs")
    r.body("The three graphs in this chapter are derived from ten scored cases. They show the "
           "distribution of judge scores, not a measured property of the software such as "
           "throughput or latency. No timing data was captured during the run, so no latency "
           "curve is presented. A reader who wants efficiency figures must run a new "
           "experiment that records time and token use per stage; the current artifacts do "
           "not support such a claim.")

    # ---- chapter 8 ----------------------------------------------------------
    r.chapter("CHAPTER 8 – OUTCOMES, IMPACT, SUSTAINABILITY AND FEASIBILITY")
    r.section("8.1 Planned Versus Achieved Outcome")
    r.body("The planned outcome was a conversational tool that answers across research "
           "papers with traceable evidence. The achieved system delivers the ingestion "
           "pipeline, two-level graph, routed retrieval, claim verification, thirteen "
           "research tools, guardrails and evaluation harness. Components that remain "
           "unimplemented are recorded in Section 8.5 rather than presented as achievements.")
    r.section("8.2 Stakeholder Benefits")
    r.body("Students gain a faster route from a question to the passage that supports an "
           "answer. Researchers gain a cross-paper comparison interface and a separate "
           "claim-verification path. Literature reviewers gain structured analyses in "
           "which published evidence is visibly separated from generated interpretation.")
    r.section("8.3 Sustainable Development Goal Relevance")
    r.body("The project supports quality education by improving access to scientific "
           "literature, and responsible consumption and production by reducing duplicated "
           "reading effort. It also supports industry, innovation and infrastructure "
           "through a reproducible open-source implementation.")
    r.section("8.4 Ethics, Privacy and Responsible AI")
    r.body("Generated analysis is labelled as such. Retrieved text is fenced as untrusted "
           "data. The system does not authenticate users and is intended for a single "
           "operator, which must be considered before any wider deployment. Personal or "
           "restricted documents should not be uploaded to configurations that use external "
           "model APIs without an appropriate data agreement.")
    r.section("8.5 Limitations")
    r.bullet("No BM25, reciprocal rank fusion or neural reranker is implemented.")
    r.bullet("Automatic claim-level citation entailment validation is not implemented.")
    r.bullet("The query-rewrite node is unreachable in the compiled graph.")
    r.bullet("The global corpus is intentionally shared, so answer-level session isolation is not guaranteed.")
    r.bullet("There is no authentication and no user-facing rate limiting.")
    r.bullet("Evaluation uses a small synthetic single-document collection.")
    r.section("8.6 Future Work")
    r.body("Future work includes a second-stage relevance scorer measured against the same "
           "goldens, persistence of retrieval contexts for per-chunk auditing, a "
           "human-authored multi-document test collection with an independent judge, a "
           "scientific-domain embedding model, and an adaptive controller that activates "
           "the rewrite node when the recorded relevancy decision is negative.")
    r.section("8.7 Final Statement")
    r.body("The project demonstrates that a two-level graph, provenance-carrying extraction "
           "and a routed workflow can make scholarly question answering both useful and "
           "auditable. The measured results identify context selection, not generation, as "
           "the next improvement target.")
    r.section("8.8 Fulfilment of Objectives")
    r.body("All six objectives stated in Chapter 1 were addressed. Multi-source ingestion, "
           "hierarchical chunks with provenance and dual storage were implemented and are "
           "visible through the interface. The two-level graph and routed workflow were "
           "implemented with the stated fallbacks. Thirteen structured tools and a guardrail "
           "chain were delivered. The evaluation harness produced five metric scores for "
           "each of ten cases, and the results were reported with their limitations.")
    r.section("8.9 Lessons Learned")
    r.bullet("Provenance must be captured during extraction; reconstructing it later is unreliable.")
    r.bullet("A stored but unreachable node is not a feature; the graph state must be inspected at runtime.")
    r.bullet("Advisory signals are useful for observability but should not be described as gates.")
    r.bullet("A small synthetic evaluation is a starting point, not evidence of production performance.")
    r.bullet("Two stores improve robustness but require explicit delegation of write failures.")
    r.section("8.10 Closure")
    r.body("The project is closed at the level of a working prototype with a reproducible "
           "evaluation. It is not presented as a production multi-tenant service. The "
           "limitations recorded in this chapter define the boundary between what was "
           "demonstrated and what would require further engineering.")
    r.section("8.11 Scalability and Sustainability")
    r.body("Scalability is limited by three resources: the embedding cache on local disk, "
           "the full-corpus character cap in the answer node and the number of chunks held in "
           "Neo4j. The first is mitigated by content hashing, the second by summarisation and "
           "selection strategies, and the third by session scoping. A larger deployment would "
           "require a shared cache, hierarchical summarisation and a bounded answer context. "
           "Sustainability is supported by caching, which avoids re-embedding identical text, "
           "and by deterministic identifiers, which prevent duplicate graph nodes.")
    r.section("8.12 Recommendations")
    r.bullet("Introduce a second-stage relevance scorer and measure it against the existing golden set.")
    r.bullet("Persist retrieval contexts with evaluation scores so that relevance failures can be audited per chunk.")
    r.bullet("Add an authentication layer and per-user throttling before any multi-user deployment.")
    r.bullet("Replace the shared global corpus search with an explicit opt-in when strict isolation is required.")
    r.bullet("Build a human-authored multi-document benchmark with an independent judge and confidence intervals.")
    r.section("8.13 Knowledge Transfer and Reuse")
    r.body("The implementation is organised so that a component can be replaced without "
           "rewriting the workflow. A different vector database requires a new wrapper with "
           "the same search and add methods. A different graph store requires a new "
           "implementation of the retrieval and explorer queries. A different answer model "
           "requires a configuration change. This modularity is the practical value of the "
           "architecture for a project that may be extended after the academic submission.")

    r.chapter("REFERENCES")
    refs = [
        "Lewis, P., Perez, E., Piktus, A., et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems 33, 2020, pp. 9459–9474.",
        "Gao, T., Yen, H., Yu, J., Chen, D. Enabling Large Language Models to Generate Text with Citations. Proceedings of EMNLP, 2023.",
        "Greshake, K., Abdelnabi, S., Mishra, S., et al. Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. Proceedings of AISec, 2023.",
        "Karpukhin, V., Oguz, B., Min, S., et al. Dense Passage Retrieval for Open-Domain Question Answering. Proceedings of EMNLP, 2020, pp. 6769–6781.",
        "Reimers, N., Gurevych, I. Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks. Proceedings of EMNLP-IJCNLP, 2019, pp. 3982–3992.",
        "Robertson, S., Zaragoza, H. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4), 2009, pp. 333–389.",
        "Cormack, G. V., Clarke, C. L. A., Buettcher, S. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. Proceedings of SIGIR, 2009, pp. 758–759.",
        "Beltagy, I., Lo, K., Cohan, A. SciBERT: A Pretrained Language Model for Scientific Text. Proceedings of EMNLP-IJCNLP, 2019, pp. 3615–3620.",
        "Edge, D., Trinh, H., Cheng, N., et al. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130, 2024.",
        "Es, S., James, J., Espinosa-Anke, L., Schockmel, R. RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217, 2023.",
        "Confident AI. DeepEval: The Open-Source LLM Evaluation Framework. https://github.com/confident-ai/deepeval, accessed 2026.",
        "OWASP. Server Side Request Forgery Prevention Cheat Sheet. https://cheatsheetseries.owasp.org, accessed 2026.",
        "LangChain AI. LangGraph Documentation. https://langchain-ai.github.io/langgraph/, accessed 2026.",
        "Neo4j. Neo4j Graph Database Documentation. https://neo4j.com/docs/, accessed 2026.",
        "Qdrant. Qdrant Documentation. https://qdrant.tech/documentation/, accessed 2026.",
    ]
    for i, ref in enumerate(refs, start=1):
        r._p(f"[{i}] {ref}", style="Body Text", align=WD_ALIGN_PARAGRAPH.JUSTIFY,
             line=1.15, after=3)

    r.chapter("APPENDIX A – CONFIGURATION CONSTANTS")
    r.table(["Constant", "Value", "Location"], [
        ["CHUNK_SIZE", "1000", "paper_loader.py"],
        ["CHUNK_OVERLAP", "200", "paper_loader.py"],
        ["EMBEDDING_DIM", "1536", "vector_store.py"],
        ["SIMILARITY_THRESHOLD", "0.78", "graph_store.py"],
        ["MAX_SIMILAR_LINKS_PER_CHUNK", "2", "graph_store.py"],
        ["EXTRACTION_BATCH_SIZE", "3", "graph_store.py"],
        ["EXTRACTION_MAX_WORKERS", "5", "graph_store.py"],
        ["MAX_RETRIEVAL_ATTEMPTS", "3", "rag_graph.py"],
        ["MAX_UPLOAD_BYTES", "20971520", "content_guard.py"],
        ["MAX_PDF_PAGES", "200", "content_guard.py"],
        ["MAX_USER_CHARS", "8000", "content_guard.py"],
        ["MIN_RESEARCH_CHARS", "800", "content_guard.py"],
        ["METRIC_THRESHOLD", "0.7", "evaluate.py"],
    ], "Configuration constants used by the implementation")

    r.chapter("APPENDIX B – ROUTE AND TOOL CONTRACTS")
    r.table(["Contract", "Fields", "Constraint"], [
        ["RouterDecision", "route", "retrieve, verify_claim or direct_answer"],
        ["RelevancyDecision", "is_relevant, reason", "Boolean and string"],
        ["ClaimVerificationResult", "is_superseded, verdict_summary, superseding_papers", "At most three papers"],
        ["ExtractedEntity", "name, entity_type, description, evidence_quote, chunk_indices", "Five semantic types"],
        ["ExtractedRelation", "source, target, relation, description, evidence_quote, chunk_index", "Upper snake case"],
        ["ResearchToolOutput", "evidence, tables, analysis, ideas, notes", "Evidence and AI content separated"],
    ], "Structured contracts enforced by Pydantic models")

    r.chapter("APPENDIX C – EVALUATION PROTOCOL")
    r.body("The evaluation protocol is reproducible from the repository. Delete goldens.json "
           "to regenerate questions, then run evaluate.py. The script caches synthetic "
           "goldens, answers each question in a unique session, judges five metrics and "
           "writes eval_results.json. The evaluation tab in the application reads that file "
           "and displays per-case scores. The results in Chapter 7 are the values recorded "
           "in the repository at the time of writing and are a historical snapshot.")

    r.chapter("APPENDIX D – EXAMPLE QUERIES AND WORKFLOWS")
    r.body("The following examples illustrate how the router selects a strategy. A question "
           "such as “What limitations does the paper report?” is routed to retrieve, and the "
           "graph tool searches the two-level graph before falling back to vector or full "
           "corpus text. A request such as “Verify the claim that encoder-decoder models are "
           "the best approach for translation” is routed to verify_claim and produces a "
            "structured verdict with links. A general question such as “What is a "
            "retrieval-augmented system?” may be routed to direct_answer.")
    r.table(["Query type", "Route", "Tools used", "Output"], [
        ["Paper content", "retrieve", "Graph, vector, optionally web", "Grounded answer with paper attribution"],
        ["Temporal claim", "verify_claim", "Two Tavily searches", "Verdict and up to three papers"],
        ["General knowledge", "direct_answer", "None", "Short answer without retrieval"],
        ["Off-topic", "Side channel", "Tavily if needed", "Answer excluded from session history"],
    ], "Example query routes")
    r.chapter("APPENDIX E – DEPLOYMENT CHECKLIST")
    r.bullet("Confirm Python 3.12 and uv are installed.")
    r.bullet("Start the Neo4j container and verify the HTTP endpoint.")
    r.bullet("Create .env from .env.example and fill OpenAI, Tavily and Qdrant values.")
    r.bullet("Install dependencies with uv sync.")
    r.bullet("Start the Streamlit application and confirm the health endpoint.")
    r.bullet("Upload one test paper and confirm the loaded-paper list and graph statistics.")
    r.bullet("Run the test suite and the evaluation script before presenting the system.")
    r.chapter("APPENDIX F – GLOSSARY")
    r.table(["Term", "Meaning"], [
        ["Chunk", "Section-bounded text segment with metadata and embedding"],
        ["Provenance", "Paper, section, page and quote attached to an extracted fact"],
        ["Level 1 graph", "Document hierarchy of paper, sections and chunks"],
        ["Level 2 graph", "Semantic entities and relations with provenance"],
        ["Retrieval reason", "Label describing why a chunk was returned"],
        ["Checkpoint", "LangGraph thread state stored in SQLite"],
        ["Global corpus", "Shared index that aggregates papers across sessions"],
        ["Golden", "Synthetic question and expected answer used in evaluation"],
    ], "Glossary of project terms")
    r.chapter("APPENDIX G – REPRODUCIBILITY CHECKLIST")
    r.bullet("Record the Python version, dependency lock file and model identifiers used for the run.")
    r.bullet("Confirm that the embedding dimension is 1536 in both Qdrant and Neo4j.")
    r.bullet("Use a fresh session identifier for each evaluation question.")
    r.bullet("State whether goldens were regenerated or reused from the cached file.")
    r.bullet("Report all five metric means, ranges and pass counts, not only the best metric.")
    r.bullet("Disclose that the judge and generator share a model family.")
    r.bullet("Keep the evaluation artifacts alongside the report for later comparison.")
    r.chapter("APPENDIX H – DETAILED MODULE REFERENCE")
    r.body("This appendix records the principal functions and constants that define the "
           "behaviour described in the main chapters. It is intended as a map for a reader "
           "who wants to locate an implementation detail in the repository.")
    r.table(["Function or constant", "Module", "Role"], [
        ["CHUNK_SIZE", "paper_loader.py", "Maximum characters per chunk"],
        ["CHUNK_OVERLAP", "paper_loader.py", "Characters repeated across chunk boundaries"],
        ["load_document", "paper_loader.py", "Dispatches by source kind"],
        ["_build_hierarchical_chunks", "paper_loader.py", "Creates section-aware chunks"],
        ["EMBEDDING_DIM", "vector_store.py", "Fixed vector dimension of 1536"],
        ["get_collection_name", "vector_store.py", "Builds a session collection name"],
        ["add_paper", "vector_store.py", "Writes vectors and attempts graph write"],
        ["search_indexed", "vector_store.py", "Searches global and session collections"],
        ["ensure_schema", "graph_store.py", "Creates constraints and indexes"],
        ["_cosine_sim_matrix", "graph_store.py", "Row-wise cosine similarity"],
        ["_link_similar_chunks", "graph_store.py", "Creates SIMILAR_TO edges"],
        ["_extract_kg_for_docs", "graph_store.py", "Extracts entities and relations"],
        ["graph_retrieve", "graph_store.py", "Seeds and expands retrieval context"],
        ["MAX_RETRIEVAL_ATTEMPTS", "rag_graph.py", "Caps tool-calling rounds at three"],
        ["build_graph", "rag_graph.py", "Compiles the LangGraph state machine"],
        ["verify_claim_node", "rag_graph.py", "Runs two web searches and structured verdict"],
        ["generate_answer_node", "rag_graph.py", "Produces the final answer"],
        ["RESEARCH_TOOLS", "research_tools.py", "The thirteen tool specifications"],
        ["run_research_tool", "research_tools.py", "Executes one structured tool"],
        ["screen_upload_bytes", "content_guard.py", "Screens file type, size and PDF header"],
        ["assess_research_document", "content_guard.py", "Classifies a document as research"],
        ["guarded_read", "content_guard.py", "Fetches a URL with redirect screening"],
        ["add_source", "session_store.py", "Saves a source and descriptor"],
        ["save_chats", "session_store.py", "Persists the visible chat history"],
    ], "Principal functions and constants in the implementation")
    r.populate_toc(page_map)


def compute_page_map(pdf_path: Path, entries) -> dict[str, int | str]:
    import fitz
    pdf = fitz.open(pdf_path)
    mapping: dict[str, int | str] = {}
    # TOC pages are identified by their dense dot leaders, not by heading text
    # (the TOC itself contains the names of later headings).
    skip = {i for i, p in enumerate(pdf)
            if len(__import__("re").findall(r"\.{6,}", p.get_text())) > 5}
    for text, level, bm in entries:
        key = text[:24]
        found = ""
        for i, page in enumerate(pdf):
            if i in skip:
                continue
            page_text = " ".join(page.get_text().split())
            if key in page_text:
                found = i + 1
                break
        mapping[bm] = found
    pdf.close()
    return mapping


def main() -> None:
    print("Rendering compact figures and formula images...")
    assets = make_assets()
    print(f"  rendered {len(assets)} assets")

    # Iterative passes: build, render, re-measure page numbers until stable.
    soffice = Path("C:/Program Files/LibreOffice/program/soffice.exe")
    mapping: dict[str, int | str] = {}
    if soffice.exists():
        import tempfile
        for iteration in range(4):
            report = Report()
            write_content(report, assets, mapping or None)
            report.save()
            with tempfile.TemporaryDirectory() as td:
                subprocess.run([str(soffice), "--headless", "--convert-to", "pdf",
                                "--outdir", td, str(OUTPUT)],
                               capture_output=True, text=True, timeout=300)
                pdfs = list(Path(td).glob("*.pdf"))
                if not pdfs:
                    break
                new_map = compute_page_map(pdfs[0], report.toc_entries)
            print(f"  pass {iteration + 1}: {sum(1 for v in new_map.values() if v)} headings located")
            if new_map == mapping:
                break
            mapping = new_map
    else:
        report = Report()
        write_content(report, assets, None)
        report.save()
    print(f"wrote {OUTPUT.name} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
