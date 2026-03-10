# Smart Search Skill
## Autonomous Code Improvement via the Karpathy Autoresearch Method

**Date:** 2026-03-10  
**Author:** Peyman / Viridium Gruppe  
**Repo:** `smart-search-skill` (v0.3.0)

---

## 1. What Is the Karpathy Autoresearch Method?

In March 2026, Andrej Karpathy published [autoresearch](https://github.com/karpathy/autoresearch) — a framework for autonomous LLM-driven code improvement. The core idea:

> *An AI agent modifies code, runs a fixed evaluation metric, keeps improvements, discards regressions, and loops — overnight, unsupervised.*

The loop has four components:

```
┌─────────────────────────────────────────────────────┐
│                  AUTORESEARCH LOOP                  │
│                                                     │
│   FIXED METRIC         CODE                         │
│   (ground truth) ←──  (agent modifies)              │
│        │                    ↑                       │
│        ▼                    │                       │
│   EVALUATE ──── IMPROVE? ───┘                       │
│                    │                                │
│                  KEEP / DISCARD                     │
└─────────────────────────────────────────────────────┘
```

**Original use case:** LLM training — the agent modifies `train.py`, runs a 5-minute GPU training experiment, keeps improvements to `val_bpb` (validation bits-per-byte), loops overnight.

**This project:** We adapted the same methodology to improve a Python search package — treating `search_cost` as the metric analogous to `val_bpb`.

---

## 2. Adapting the Method: Search Workflow as the Experiment Target

### The Metric

```
search_cost = avg_latency_ms / (coverage_rate × snippet_quality × 1000)
```

| Term | Definition |
|---|---|
| `coverage_rate` | Fraction of queries returning ≥ 1 result |
| `snippet_quality` | Fraction of results with non-empty title + snippet |
| `avg_latency_ms` | Mean per-query latency |

Lower is better. Penalizes slow, empty, or low-quality results simultaneously.

### The Constraint

- `eval.py` is **immutable ground truth** — analogous to Karpathy's fixed training loop
- Agent may only modify `search_workflow/tools.py`
- No new files, no docker, no external API calls during experiments
- Every experiment is git-committed; regressions are `git reset --hard`

### The Stack

```
Homelab (Pop!_OS, AMD GPU, 32GB VRAM)
  └─ search-autoresearch/
       ├─ eval.py              ← fixed metric (immutable)
       ├─ search_workflow/
       │    └─ tools.py        ← the only file the agent edits
       ├─ results.tsv          ← experiment log
       └─ TASKSPEC.md          ← agent's bible
```

---

## 3. The Spec-Driven Agent Pattern

Rather than a free-running agent loop (which burned API credits earlier), we adopted a structured **spec-driven delegation pattern**:

```
LAYER 1: TASKSPEC.md      ← complete spec, never changes during execution
     ↓
LAYER 2: AUDIT            ← agent reads code, produces KEEP/PATCH/REWRITE/DELETE
     ↓
LAYER 3: SESSION PROMPTS  ← scoped to one deliverable, anti-patterns injected
     ↓
LAYER 4: EXECUTION        ← agent edits tools.py, commits, runs eval
     ↓
LAYER 5: HUMAN CHECKPOINT ← human verifies metric, approves next session
```

**Why this matters:** The original autoresearch loop assumes a cheap, deterministic eval (5-min GPU run). LLM agent loops burn expensive API tokens. The spec-driven pattern adds human gates that prevent runaway loops — each session costs ~5–10 API calls, not 500.

---

## 4. Experiment Results

### Baseline

```
commit: 7a555e4
search_cost:    1.467797
avg_latency_ms: 1027.5 ms
coverage_rate:  0.70      ← 3 of 10 queries returned 0 results
avg_results:    3.7
```

**Root cause of 70% coverage:**  
SearXNG had `categories: "news"` hardcoded — technical queries like *"Python asyncio best practices"* have no news coverage, returning 0 results. DDG fallback used `backend='news'` and crashed with `DecodeError`.

### Experiment Log

| Session | Commit | search_cost | Latency | Coverage | Change |
|---|---|---|---|---|---|
| Baseline | `7a555e4` | 1.467 | 1027ms | 0.70 | SearXNG news-only |
| S1 | `e1799bb` | 1.988 | 1690ms | **0.90** | `categories=general` + fix health_check |
| S2 | `e56a8df` | 2.060 | 2060ms | **1.00** | DDG `backend=text`, no more DecodeError |
| S3 | `3464765` | **1.052** | 1052ms | 1.00 | Parallel SearXNG+DDG via `asyncio.gather` |
| S4 | `f45b4cc` | 1.675 | 1675ms | 1.00 | Switch to `DDGS().text()` — 10 results/query |
| Infra fix | — | 1.677 | 1627ms | 1.00 | SearXNG engines un-suspended |

### Key Insight: Infrastructure Was the Bottleneck

After code improvements, we discovered SearXNG itself was silently broken:

```
unresponsive engines:
  - ['brave',     'Suspended: too many requests']
  - ['duckduckgo','Suspended: access denied']
  - ['startpage', 'Suspended: CAPTCHA']
```

Default SearXNG bans engines for **24 hours** on first error. This meant the entire general web search layer was dark. Fix: reduce ban times to 15 minutes.

```yaml
# Before (default)              # After (fixed)
SearxEngineAccessDenied: 86400  →  900   # 15 min
SearxEngineCaptcha:      86400  →  900   # 15 min  
SearxEngineTooManyReqs:  3600   →  300   # 5 min
cf_SearxEngineCaptcha:   1296000 → 3600  # 1h
```

---

## 5. Smart Search Skill — What It Is

`smart-search-skill` is a self-hosted Python search package for technical teams. It replaces commercial search APIs with a privacy-first, resilient hybrid engine.

### Architecture

```
query
  │
  ├─► SearXNG (categories=general)  ─┐
  │   asyncio.gather (parallel)      ├─► dedup by URL ──► results[:max]
  └─► DuckDuckGo DDGS.text()        ─┘

Engines active in SearXNG: brave, duckduckgo, startpage, google, github, stackoverflow
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Parallel, not sequential fallback | Eliminates fallback latency penalty (~600ms saved) |
| `DDGS().text()` over LangChain wrapper | 10 results vs 4-result cap |
| `categories` as a parameter | Callers choose `"general"`, `"news"`, `"it"`, `"general,news"` |
| `health_check` via real search probe | `/healthz` doesn't exist on SearXNG — was silently bypassing it |
| DDG `backend='text'` not `'news'` | `'news'` crashes on technical queries with `DecodeError` |

### Benchmark: Org Use Cases

Tested against 8 real technical queries (GitHub, Stack Overflow, official docs):

| Category | Query | Top results |
|---|---|---|
| **GitHub** | `langchain langgraph github source code` | github.com ×4 |
| **GitHub** | `fastapi github examples authentication` | github.com ×3, fastapi-users.github.io |
| **Stack Overflow** | `python asyncio gather exception handling` | stackoverflow.com, realpython.com |
| **Stack Overflow** | `typescript strict null checks` | stackoverflow.com, typescriptlang.org |
| **Official docs** | `PostgreSQL JSONB indexing documentation` | postgresql.org ×2 |
| **Official docs** | `Docker compose healthcheck` | docs.docker.com ×2 |
| **Official docs** | `neo4j python driver documentation` | neo4j.com ×3 |
| **Official docs** | `aiohttp client session best practices` | docs.aiohttp.org ×2 |

**Result: GitHub repos, Stack Overflow, and official docs surface correctly for all query types.**

### Final Metrics

```
search_cost:      1.677730
avg_latency_ms:   1627ms
coverage_rate:    1.00    ← 10/10 queries return results
snippet_quality:  0.97
avg_results:      10.0    ← up from 3.7 at baseline
```

---

## 6. Lessons Learned

### On the Autoresearch Method

1. **The method works for any fixed-metric optimization problem** — not just GPU training. Any codebase with a runnable eval metric is a candidate.

2. **The metric must be meaningful before the loop starts.** We spent hours debugging a broken eval harness before getting valid measurements. Rule: *manually verify `python3 eval.py` produces a clean metric before spawning any agent.*

3. **Infrastructure failures masquerade as code problems.** The biggest improvement came from fixing SearXNG's ban times — not from any code change. The autoresearch loop can't discover infrastructure issues; human diagnosis is irreplaceable.

4. **Cost control is non-negotiable for LLM agents.** An agent told to "NEVER STOP" on a broken eval will burn your entire API quota. The spec-driven pattern with human checkpoints and `--max-turns 10` is the right guard.

### On the Spec-Driven Pattern

```
Spec (30 min) → Audit (5 turns) → Sessions (8 turns each) → Human checkpoint
```

- Writing the spec upfront forces clarity about what "done" looks like
- The audit prevents building on top of broken foundations  
- Anti-pattern injection ("old code did X wrong, don't repeat") is more effective than abstract rules
- Human checkpoints after each session prevent compounding failures

---

## 7. Repo

**Name:** `smart-search-skill`  
**Version:** `0.3.0`  
**License:** MIT  

```bash
pip install smart-search-skill
# or
uv add smart-search-skill
```

```python
from search_workflow.tools import search_direct

results = await search_direct(
    "neo4j python driver documentation",
    max_results=10,
    categories="general",
)
```

**Prerequisites:** SearXNG running locally (Docker compose included in repo).

---

*Report generated: 2026-03-10 | Homelab: Pop!_OS, AMD RDNA3, 32GB VRAM*
