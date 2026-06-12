---
description: Document the latest ML concept into my Notion "ML Journey" database
argument-hint: [optional concept name]
---

Document an ML/fastai concept into my Notion **ML Journey** database.

Target data source: `6703affa-3621-448a-bfc5-432db79d6167`
(database: https://app.notion.com/p/c9744eb3f8df4ebbb1a9672d443a5000)

Concept to document: $ARGUMENTS
If that is empty, use the most recent concept(s) we discussed in this conversation.

Create ONE new page in that data source (use the `notion-create-pages` tool with parent `data_source_id`) with these properties:
- **Concept** (title): the concept name
- **Topics** (multi-select, pick the best fits, JSON array string): any of `fastai`, `PyTorch & Tensors`, `Audio & DSP`, `Voice Cloning`, `Math`, `Deep Learning`
- **Status**: `Understood` (use `Needs review` if I said I'm still shaky on it)
- **Date**: today's date (set the `date:Date:start` property)
- **Source**: where it came from (e.g. "fastbook Ch.4 — 04_mnist_basics")

Page body — same plain-English teaching style we use (concise, skimmable, with real tiny examples):
1. **One-line idea** — the concept in a single sentence.
2. **Plain-English explanation** — with a tiny concrete example / real numbers if it helps.
3. **Why it matters** — tie it to voice cloning / the bigger goal where relevant.
4. **Code** — the minimal snippet + a plain-English read of it.
5. **Analogy** — a JS / engineer analogy if one fits.

When done, reply with the new page's URL and a one-line confirmation. Do NOT include the concept title inside the page body (it's already the page title).
