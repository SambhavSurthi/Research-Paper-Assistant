# Capstone Report — Version 3

`Capstone-Report-v3.docx` is the curated, project-grounded capstone report.
It replaces the earlier versions with expanded content, the README diagrams and
the Chapter 7 evaluation graphs. The v1 file remains the formatting template.

## What changed in v3

- **Length:** 52 A4 pages, measured by rendering with LibreOffice; within the
  strict 50–60 page requirement.
- **Expanded content:** Every chapter now contains additional project-grounded
  analysis: stakeholder and risk analysis, detailed literature evidence,
  data-retention and acceptance catalogues, full graph schema, deployment
  operations, a 13-tool catalogue, failure-mode and security test tables,
  per-case result tables, metric interpretation, scalability and
  recommendations. Appendices A–H cover configuration, contracts, evaluation
  protocol, example workflows, deployment, glossary, reproducibility and a
  module reference.
- **README diagrams:** The four high-level diagrams from `README.md` were
  rendered locally with Mermaid CLI and inserted: system at a glance,
  high-level component architecture, LangGraph routing state machine, and the
  turn sequence diagram. Source `.mmd` files and rendered PNGs are in `assets/`.
- **Chapter 7 graphs:** Three charts generated strictly from `eval_results.json`:
  mean metric scores with the 0.7 threshold, contextual relevancy by test case,
  and pass counts per metric. No invented latency or cost data is shown.
- **Index:** A static, clickable table of contents with 125 entries, dot leaders,
  correct page numbers and internal bookmarks. It is built iteratively so page
  numbers remain correct after every content edit.
- **Format:** v1 styles, fonts, blue heading colours, margins, header and
  page-numbered footer are preserved. Times New Roman body at 12 pt, justified,
  1.15 line spacing. Every chapter, front-matter item and appendix starts on a
  new page.

## Rebuild

```powershell
cd reports
& ..\.venv\Scripts\python.exe build_report.py
```

Requirements: project virtual environment with `python-docx`, `Pillow`,
`PyMuPDF`; Tectonic at `%LOCALAPPDATA%\tectonic_msvc\tectonic.exe` for the
compact figures and formulae; LibreOffice at
`C:\Program Files\LibreOffice\program\soffice.exe` for page measurement; and
Mermaid CLI (`mmdc`) for the README diagrams. The four `readme_*.png` files in
`assets/` are reused on subsequent builds.

The build writes `Capstone-Report-v3.docx` and a verification PDF in
`pdf_check/`. If the previous output file is open in Word, close it first or
change `OUTPUT` in `build_report.py`.
