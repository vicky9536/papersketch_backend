# PaperSketch Backend

**A multi-LLM paper summarization and figure extraction backend with fair model comparison**

## Overview

PaperSketch Backend is a research-oriented backend system that takes a paper PDF URL (e.g. arXiv), extracts the paper content and key figures, and produces a concise **PaperSketch-style markdown summary** using large language models (LLMs).

The system is designed with **clean separation of responsibilities**:

* **Backend**: preprocessing, figure extraction, LLM inference, and evaluation metrics
* **Connector / UI**: rendering the returned markdown and images into a visual sketch

A core design goal is **fair comparison across LLMs**.
To achieve this, the backend preprocesses each paper **once**, then runs multiple LLMs on the **same extracted text, figures, and prompt**, measuring **LLM-only latency and token usage**.

---

## Key Features

* 📄 PDF ingestion from public URLs (arXiv supported)
* 🧠 Multi-LLM support:

  * OpenAI
  * Google Gemini
  * DeepSeek (OpenAI-compatible API)
* 🖼 Automatic figure page detection and rendering
* 🧪 Fair LLM comparison:

  * shared preprocessing
  * per-model inference timing
  * per-model token usage
* 🌐 REST API (FastAPI)
* 🖥 Flask UI for interactive comparison and inspection
* 🔌 Connector-friendly output (markdown + image URLs)

---

## High-Level Architecture

```
Paper URL
   │
   ▼
[ Preprocess Once ]
   ├─ download PDF
   ├─ extract text
   ├─ detect figure pages
   ├─ render page images
   ▼
Shared Context
   │
   ├─ LLM A (OpenAI)
   ├─ LLM B (Gemini)
   ├─ LLM C (DeepSeek)
   ▼
Markdown PaperSketch + Metrics
```

**Important**:
The backend does **not** render the final sketch UI.
It returns **markdown and image URLs**, which are rendered by:

* a ChatGPT connector, or
* the provided Flask comparison UI.

---

## Project Structure

```
papersketch-backend/
├─ src/papersketch_backend/
│  ├─ settings.py              # Model configuration helpers
│  ├─ cache.py                 # Structured document cache
│  ├─ api/
│  │  ├─ app.py                 # FastAPI app factory + route registration
│  │  ├─ deps.py                # Shared API dependencies (auth)
│  │  └─ routes/
│  │     ├─ papersketch.py      # Single-model API
│  │     ├─ compare.py          # Multi-model comparison API
│  │     └─ structure.py        # Structured extraction/summary APIs
│  ├─ ui/
│  │  ├─ app.py                 # Flask comparison UI
│  │  └─ templates/
│  │     ├─ index.html
│  │     └─ result.html
│  ├─ pipeline/
│  │  ├─ preprocess.py          # Shared preprocessing (run once)
│  │  ├─ summarize.py           # LLM-only inference
│  │  ├─ run.py                 # Single-model orchestration
│  │  └─ prompt.py              # Prompt construction
│  ├─ llm/
│  │  ├─ base.py                # LLM interface
│  │  ├─ registry.py            # Model routing
│  │  ├─ openai_client.py
│  │  ├─ gemini_client.py
│  │  └─ deepseek_client.py
│  ├─ document/
│  │  ├─ fetch.py               # Secure PDF download
│  │  ├─ models.py              # Structured document dataclasses
│  │  ├─ text_blocks.py         # BBox-aware block extraction
│  │  ├─ layout.py              # Layout region detection
│  │  ├─ reading_order.py       # Two-column reading order
│  │  ├─ sections.py            # Section tree builder
│  │  ├─ figure_extraction.py   # Structured figure/table extraction
│  │  └─ chunking.py            # Chunk generation
│  └─ knowledge/                # Structured LLM outputs
│
├─ static/                      # Runtime-rendered figure images (not committed)
├─ cache/                       # Runtime document cache (not committed)
├─ README.md
```

---

## API Endpoints

### `GET /api/v1/papersketch_url`

Generate a PaperSketch using **one LLM**.

**Query parameters**

* `url` – paper PDF URL (required)
* `lang` – output language (`en`, `zh`)
* `model` – model spec (e.g. `openai:gpt-4o`)
* preprocessing controls (`max_pages`, `max_chars`, etc.)

**Response**

```json
{
  "paperSketch": "...markdown...",
  "modelInfo": "openai:gpt-4o",
  "preprocess_ms": 2400,
  "llm_ms": 1800,
  "latency_ms": 4200,
  "usage": {
    "prompt_tokens": 8200,
    "output_tokens": 520,
    "total_tokens": 8720
  },
  "meta": { ... }
}
```

---

### `GET /api/v1/papersketch_compare`

Compare **multiple LLMs fairly** on the same paper.

* Preprocessing is executed **once**
* Each LLM is timed independently

**Response highlights**

```json
{
  "shared": {
    "preprocess_ms": 2400,
    "figure_pages": [1, 3, 7]
  },
  "results": [
    {
      "model": "openai:gpt-4o",
      "llm_ms": 1800,
      "latency_ms": 4200,
      "usage": { ... }
    },
    {
      "model": "gemini:gemini-3-flash-preview",
      "llm_ms": 1200,
      "latency_ms": 3600
    }
  ]
}
```

---

## Evaluation Methodology (Important)

This project explicitly separates **preprocessing cost** from **LLM inference cost**.

### Why?

* PDF download, text extraction, and figure detection are **model-independent**
* Repeating them per model would inflate latency and bias comparisons

### Method

1. Preprocess the paper **once**
2. Fix:

   * extracted text
   * figure candidates
   * prompt template
3. Measure per-model:

   * LLM inference time (`llm_ms`)
   * token usage (when available)

---

## Supported LLM Providers

| Provider | Example model spec       |
| -------- | ------------------------ |
| OpenAI   | `openai:gpt-4o`     |
| Gemini   | `gemini:gemini-3-flash-preview`  |
| DeepSeek | `deepseek:deepseek-chat` |

Adding a new provider requires:

1. Implementing `LLMClient`
2. Registering it in `llm/registry.py`

---

## Flask Comparison UI

The Flask UI is a **developer / evaluation tool**.

Features:

* paper URL input
* model selection (single or ALL)
* latency & token comparison table
* raw markdown output inspection
* figure image preview

Run:

```bash
PYTHONPATH=src python -m papersketch_backend.ui.app
```

Open:

```
http://127.0.0.1:5000
```

---

## Setup Instructions

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn flask requests pymupdf openai google-genai
```

### 3. Set environment variables

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="..."

export BASE_URL="http://127.0.0.1:8001"
```

---

## Run the Backend

```bash
python -m uvicorn --app-dir src papersketch_backend.api.app:app --port 8001 --reload
```

Docs:

```
http://127.0.0.1:8001/docs
```

---

## Design Philosophy

* **Backend = analysis + assets**
* **Frontend / connector = rendering**
* **LLMs are evaluated fairly**
* **Architecture favors clarity over shortcuts**
