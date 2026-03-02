from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from papersketch_backend.document.models import BBox, Chunk, Figure, StructuredDocument
from papersketch_backend.evaluation import evaluate_compare_results


class EvaluationTests(unittest.TestCase):
    def test_compare_results_are_ranked(self) -> None:
        doc = StructuredDocument(
            pdf_sha256="abc",
            title="Test Paper",
            chunks=[
                Chunk(
                    chunk_id="aaaabbbbcccc",
                    text="Transformer architecture improves benchmark accuracy on ImageNet.",
                    section_path=["Method"],
                    page=0,
                    token_count=10,
                )
            ],
            figures=[
                Figure(
                    label="Figure 1",
                    bbox=BBox(0, 0, 10, 10, 0),
                    crop_url="http://127.0.0.1:8001/static/abc/fig_1.png",
                )
            ],
        )

        strong = {
            "model": "openai:test",
            "modelInfo": "openai:test",
            "status": "OK",
            "paperSketch": "\n".join(
                [
                    "## Research Background",
                    "Transformer architecture benchmark accuracy.",
                    "## Research Methodology",
                    "Uses a transformer architecture.",
                    "## Experimental Results",
                    "Improves ImageNet accuracy.",
                    "## Main Contributions",
                    "- Better benchmark accuracy.",
                    "## Limitations",
                    "- Not specified.",
                    "## Future Work",
                    "- Extend evaluation.",
                    "![figure](http://127.0.0.1:8001/static/abc/fig_1.png)",
                ]
            ),
        }
        weak = {
            "model": "gemini:test",
            "modelInfo": "gemini:test",
            "status": "OK",
            "paperSketch": "Short vague summary.",
        }

        out = evaluate_compare_results(
            doc=doc,
            results=[weak, strong],
            reference_summary="Transformer architecture improves benchmark accuracy on ImageNet.",
        )

        self.assertEqual(out["heuristic_ranking"][0]["model"], "openai:test")
        self.assertGreater(
            strong["evaluation"]["heuristic"]["overall_score"],
            weak["evaluation"]["heuristic"]["overall_score"],
        )
        self.assertIsNotNone(strong["evaluation"]["reference"])
        self.assertGreater(strong["evaluation"]["reference"]["reference_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
