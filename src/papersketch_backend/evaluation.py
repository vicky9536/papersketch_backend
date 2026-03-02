from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any, Optional

from papersketch_backend.document.models import StructuredDocument
from papersketch_backend.llm.registry import resolve


_REQUIRED_HEADERS = [
    "## Research Background",
    "## Research Methodology",
    "## Experimental Results",
    "## Main Contributions",
    "## Limitations",
    "## Future Work",
]

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
_STOPWORDS = {
    "this", "that", "with", "from", "they", "their", "there", "which", "using",
    "into", "were", "have", "has", "been", "such", "than", "then", "also",
    "over", "under", "more", "most", "some", "many", "each", "paper", "model",
    "method", "methods", "results", "result", "based", "used", "show", "shows",
    "use", "between", "where", "when", "what", "while", "these", "those",
    "because", "about", "after", "before", "without", "within", "across",
    "through", "being", "does", "done", "make", "made", "only", "provided",
    "figure", "table", "section",
}
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _normalize_words(text: str) -> list[str]:
    return [
        word.lower()
        for word in _WORD_RE.findall(text)
        if word.lower() not in _STOPWORDS
    ]


def _top_keywords(doc: StructuredDocument, limit: int = 20) -> list[str]:
    counts: Counter[str] = Counter()
    for chunk in doc.chunks:
        if chunk.chunk_type == "caption":
            continue
        counts.update(_normalize_words(chunk.text))
    return [word for word, _ in counts.most_common(limit)]


def _score_headers(text: str) -> float:
    if not text.strip():
        return 0.0
    hits = sum(1 for header in _REQUIRED_HEADERS if header in text)
    return hits / len(_REQUIRED_HEADERS)


def _score_keyword_coverage(text: str, keywords: list[str]) -> float:
    if not text.strip() or not keywords:
        return 0.0
    words = set(_normalize_words(text))
    hits = sum(1 for word in keywords if word in words)
    return hits / len(keywords)


def _score_length(text: str) -> float:
    length = len(text.strip())
    if length < 300:
        return length / 300
    if length <= 3500:
        return 1.0
    if length >= 7000:
        return 0.0
    return max(0.0, 1.0 - ((length - 3500) / 3500))


def _score_figure_usage(text: str, doc: StructuredDocument) -> float:
    figure_urls = [f.crop_url for f in doc.figures if f.crop_url]
    if not figure_urls:
        return 1.0
    used = sum(1 for url in figure_urls if url in text)
    return min(1.0, used / min(len(figure_urls), 3))


def _evaluate_heuristic(text: str, doc: StructuredDocument, keywords: list[str]) -> dict[str, Any]:
    structure = _score_headers(text)
    coverage = _score_keyword_coverage(text, keywords)
    length = _score_length(text)
    figures = _score_figure_usage(text, doc)

    total = (
        (structure * 0.35)
        + (coverage * 0.40)
        + (length * 0.15)
        + (figures * 0.10)
    )

    return {
        "structure_score": round(structure, 3),
        "coverage_score": round(coverage, 3),
        "length_score": round(length, 3),
        "figure_score": round(figures, 3),
        "overall_score": round(total * 100, 1),
    }


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n or n <= 0:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f1_from_counters(pred: Counter[Any], gold: Counter[Any]) -> float:
    if not pred or not gold:
        return 0.0
    overlap = sum((pred & gold).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(gold.values())
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for tok_a in a:
        prev = 0
        for j, tok_b in enumerate(b, start=1):
            cur = dp[j]
            if tok_a == tok_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def _reference_metrics(text: str, reference_summary: str) -> Optional[dict[str, Any]]:
    if not text.strip() or not reference_summary.strip():
        return None

    pred_tokens = _normalize_words(text)
    ref_tokens = _normalize_words(reference_summary)
    if not pred_tokens or not ref_tokens:
        return None

    rouge1 = _f1_from_counters(_ngrams(pred_tokens, 1), _ngrams(ref_tokens, 1))
    rouge2 = _f1_from_counters(_ngrams(pred_tokens, 2), _ngrams(ref_tokens, 2))

    lcs = _lcs_length(pred_tokens, ref_tokens)
    prec_l = lcs / len(pred_tokens)
    rec_l = lcs / len(ref_tokens)
    rouge_l = 0.0 if (prec_l + rec_l) == 0 else (2 * prec_l * rec_l / (prec_l + rec_l))

    token_overlap = len(set(pred_tokens) & set(ref_tokens))
    precision = token_overlap / len(set(pred_tokens))
    recall = token_overlap / len(set(ref_tokens))
    token_f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))

    aggregate = (rouge1 + rouge2 + rouge_l + token_f1) / 4

    return {
        "rouge1_f1": round(rouge1, 3),
        "rouge2_f1": round(rouge2, 3),
        "rougeL_f1": round(rouge_l, 3),
        "token_f1": round(token_f1, 3),
        "reference_score": round(aggregate * 100, 1),
    }


def _paper_context(doc: StructuredDocument, max_chars: int = 8000) -> str:
    lines: list[str] = []
    if doc.title:
        lines.append(f"Title: {doc.title}")

    if doc.sections:
        lines.append("Sections:")
        for sec in doc.sections[:12]:
            lines.append(f"- {sec.section_id}: {sec.title}")

    lines.append("Key source chunks:")
    char_count = 0
    for chunk in doc.chunks:
        if chunk.chunk_type == "caption":
            continue
        snippet = chunk.text.strip().replace("\n", " ")
        piece = f"[{chunk.chunk_id}] {snippet}\n"
        if char_count + len(piece) > max_chars:
            lines.append("... (truncated)")
            break
        lines.append(piece)
        char_count += len(piece)

    return "\n".join(lines)


def _judge_prompt(doc: StructuredDocument, results: list[dict[str, Any]]) -> str:
    candidates: list[str] = []
    for row in results:
        if row.get("status") != "OK":
            continue
        candidates.append(
            f"MODEL: {row.get('model')}\n"
            f"SUMMARY:\n{row.get('paperSketch', '').strip()}\n"
        )

    return f"""You are evaluating academic-paper summaries from multiple models.

Score each model on a 1-5 scale for:
- factuality
- coverage
- clarity
- usefulness

Use the source paper context below as ground truth.
Return ONLY valid JSON in this exact shape:
{{
  "scores": [
    {{
      "model": "openai:gpt-4o-mini",
      "factuality": 4,
      "coverage": 4,
      "clarity": 5,
      "usefulness": 4,
      "overall": 4.25,
      "reason": "short reason"
    }}
  ]
}}

SOURCE PAPER CONTEXT
{_paper_context(doc)}

CANDIDATE SUMMARIES
{chr(10).join(candidates)}
"""


def _parse_judge_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    m = _JSON_FENCE_RE.search(raw)
    if m:
        raw = m.group(1).strip()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Judge returned a non-object JSON payload.")
    return data


def _judge_results(
    *,
    doc: StructuredDocument,
    results: list[dict[str, Any]],
    judge_model: Optional[str],
) -> dict[str, Any]:
    if not judge_model:
        return {"enabled": False, "model": None, "ranking": [], "error": None}

    ok_results = [row for row in results if row.get("status") == "OK" and row.get("paperSketch")]
    if not ok_results:
        return {"enabled": False, "model": judge_model, "ranking": [], "error": "No successful model outputs to judge."}

    try:
        client, provider, model_name = resolve(judge_model)
        prompt = _judge_prompt(doc, ok_results)
        t0 = time.perf_counter()
        resp = client.summarize(prompt=prompt, model=model_name)
        judge_ms = int((time.perf_counter() - t0) * 1000)
        data = _parse_judge_json(resp.text)
    except Exception as exc:
        return {
            "enabled": True,
            "model": judge_model,
            "ranking": [],
            "error": str(exc),
        }

    score_rows = data.get("scores", [])
    if not isinstance(score_rows, list):
        return {
            "enabled": True,
            "model": judge_model,
            "ranking": [],
            "error": "Judge response missing 'scores' list.",
        }

    by_model: dict[str, dict[str, Any]] = {}
    ranking: list[dict[str, Any]] = []
    for item in score_rows:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model", "")).strip()
        if not model:
            continue
        judge_info = {
            "factuality": float(item.get("factuality", 0)),
            "coverage": float(item.get("coverage", 0)),
            "clarity": float(item.get("clarity", 0)),
            "usefulness": float(item.get("usefulness", 0)),
            "overall_score": round(float(item.get("overall", 0)) * 20, 1),
            "reason": str(item.get("reason", "")).strip(),
            "judge_model": f"{provider}:{model_name}",
            "judge_ms": judge_ms,
        }
        by_model[model] = judge_info
        ranking.append(
            {
                "model": model,
                "overall_score": judge_info["overall_score"],
                "reason": judge_info["reason"],
            }
        )

    ranking.sort(key=lambda item: item["overall_score"], reverse=True)
    for idx, item in enumerate(ranking, start=1):
        item["rank"] = idx

    return {
        "enabled": True,
        "model": f"{provider}:{model_name}",
        "ranking": ranking,
        "scores_by_model": by_model,
        "error": None,
    }


def evaluate_compare_results(
    *,
    doc: StructuredDocument,
    results: list[dict[str, Any]],
    reference_summary: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> dict[str, Any]:
    """
    Combined evaluator for multi-model markdown PaperSketch outputs.

    Includes:
    - heuristic rubric (format/coverage/length/figure use)
    - reference-based metrics when a gold summary is provided
    - optional LLM-judge scoring and ranking
    """
    keywords = _top_keywords(doc)
    heuristic_ranking: list[dict[str, Any]] = []

    for row in results:
        heuristic = (
            _evaluate_heuristic(row.get("paperSketch", ""), doc, keywords)
            if row.get("status") == "OK"
            else {
                "structure_score": 0.0,
                "coverage_score": 0.0,
                "length_score": 0.0,
                "figure_score": 0.0,
                "overall_score": 0.0,
            }
        )
        reference = (
            _reference_metrics(row.get("paperSketch", ""), reference_summary or "")
            if row.get("status") == "OK"
            else None
        )

        row["evaluation"] = {
            "overall_score": heuristic["overall_score"],
            "heuristic": heuristic,
            "reference": reference,
            "judge": None,
        }

        if row.get("status") == "OK":
            heuristic_ranking.append(
                {
                    "model": row.get("model"),
                    "modelInfo": row.get("modelInfo"),
                    "overall_score": heuristic["overall_score"],
                }
            )

    heuristic_ranking.sort(key=lambda item: item["overall_score"], reverse=True)
    for idx, item in enumerate(heuristic_ranking, start=1):
        item["rank"] = idx

    judge = _judge_results(doc=doc, results=results, judge_model=judge_model)
    if judge.get("scores_by_model"):
        by_model = judge["scores_by_model"]
        for row in results:
            judge_info = by_model.get(row.get("model"))
            if judge_info:
                row["evaluation"]["judge"] = judge_info

    reference_enabled = bool((reference_summary or "").strip())
    display_ranking = judge["ranking"] if judge.get("ranking") else heuristic_ranking
    display_source = "judge" if judge.get("ranking") else "heuristic"

    return {
        "keywords_used": keywords,
        "heuristic_ranking": heuristic_ranking,
        "reference_enabled": reference_enabled,
        "judge": judge,
        "display_ranking": display_ranking,
        "display_ranking_source": display_source,
    }
