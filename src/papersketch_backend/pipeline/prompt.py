from __future__ import annotations

from papersketch_backend.document.models import StructuredDocument


def _lang_name(lang: str) -> str:
    l = (lang or "en").strip().lower()
    if l in ("ch", "zh", "cn", "chinese"):
        return "Chinese"
    return "English"


def _format_chunks(doc: StructuredDocument, max_context_chars: int) -> str:
    lines: list[str] = []
    char_count = 0
    current_path: tuple[str, ...] = ()

    for chunk in doc.chunks:
        if chunk.chunk_type == "caption":
            continue

        path = tuple(chunk.section_path)
        if path != current_path:
            header = " > ".join(path) if path else "Preamble"
            lines.append(f"\n### {header}\n")
            current_path = path

        piece = f"{chunk.text}\n"
        if char_count + len(piece) > max_context_chars:
            lines.append("\n... (content truncated for length) ...\n")
            break
        lines.append(piece)
        char_count += len(piece)

    return "\n".join(lines).strip()


def _figure_markdown(doc: StructuredDocument, max_figures: int = 6) -> list[str]:
    urls: list[str] = []
    for fig in doc.figures:
        if fig.crop_url:
            urls.append(fig.crop_url)
        if len(urls) >= max_figures:
            break

    lines: list[str] = []
    for idx, url in enumerate(urls, start=1):
        lines.append(f"![figure={idx}]({url})")
    return lines


def build_prompt(
    *,
    url: str,
    lang: str = "en",
    document: StructuredDocument,
    max_context_chars: int = 24_000,
) -> str:
    """
    Build the shared prompt for markdown PaperSketch generation.

    The prompt is now driven entirely by the structured extraction pipeline:
    body chunks provide text context and detected figure crops provide
    embeddable image URLs.
    """
    language = _lang_name(lang)
    chunks_block = _format_chunks(document, max_context_chars)
    figure_urls = _figure_markdown(document)

    figures_block = ""
    if figure_urls:
        figures_md = "\n".join(figure_urls)
        figures_block = f"""
AVAILABLE FIGURES (use ONLY these URLs if you include images):
{figures_md}
"""

    text_block = ""
    if chunks_block:
        text_block = f"""
PAPER CONTENT (structured extraction; may be truncated):
\"\"\"\n{chunks_block}\n\"\"\"
"""

    prompt = f"""PAPER URL:
{(url or '').strip()}
{figures_block}
{text_block}
You are an expert research assistant. Summarize the paper concisely in {language}.

OUTPUT FORMAT (strict):
- Return ONLY markdown.
- Use these exact section headers (markdown H2):
  ## Research Background
  ## Research Methodology
  ## Experimental Results
  ## Main Contributions
  ## Limitations
  ## Future Work

STYLE:
- Prefer bullet points.
- Be specific (datasets, metrics, main claims, key ablations), but keep it short.
- Do NOT invent details. If missing, write "Not specified in the provided text."
- If you include figures, insert them near the most relevant section using markdown image syntax:
  ![figure](URL)
- Use at most 2–6 figures.

CONTENT PRIORITIES:
1) What problem is solved and why it matters
2) What method is proposed (core idea + components)
3) What experiments were run and what the key results are
4) Contributions (3–5 bullets)
5) Limitations (2–4 bullets)
6) Future work (2–4 bullets)
"""
    return prompt.strip() + "\n"
