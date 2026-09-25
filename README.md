# HackJudge AI

An LLM-augmented static analysis pipeline for automated hackathon submission triage, combining heuristic commit-history forensics with generative scoring and cross-repository plagiarism detection via lexical hashing and dense vector similarity.

---

## Abstract

HackJudge AI ingests a public GitHub repository URL and produces a structured, multi-dimensional evaluation artifact suitable for hackathon adjudication. The system decomposes the evaluation problem into three orthogonal subsystems: **static metadata extraction** (GitHub REST API v3), **generative reasoning** (Gemini 3.5 Flash-Lite via the `google-genai` SDK), and **similarity detection** (SHA-256 content hashing + dense embedding cosine similarity). The frontend is implemented as a single-page Streamlit application with session-scoped in-memory state, custom CSS injection for non-default theming, and a tab-based leaderboard aggregation view.

---

## System Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌───────────────┐
│  Streamlit  │─────▶│  github_fetcher   │─────▶│  GitHub REST   │
│    (app.py) │      │      .py          │      │    API v3      │
└──────┬──────┘      └──────────────────┘      └───────────────┘
       │
       ├──────────────▶ similarity.py ──────▶ Gemini Embedding API
       │                (hashing + cosine)      (gemini-embedding-001)
       │
       └──────────────▶ summarizer.py ───────▶ Gemini Generative API
                         (structured scoring)    (gemini-3.5-flash-lite)
```

### Module responsibilities

| Module | Responsibility | External I/O |
|---|---|---|
| `app.py` | Presentation layer, session-state orchestration, UI rendering | None (delegates) |
| `github_fetcher.py` | Repository tree traversal, blob retrieval, commit-log parsing | GitHub REST API v3 (`/git/trees`, `/commits`) |
| `summarizer.py` | Prompt construction, structured JSON generation via constrained decoding | Gemini `generateContent` |
| `similarity.py` | Content normalization, cryptographic hashing, embedding generation, vector similarity | Gemini `embedContent` |

---

## Core Subsystems

### 1. Repository Ingestion (`github_fetcher.py`)

Repository state is resolved via a two-phase branch-resolution strategy against `main` and `master` refs, using the recursive Git Trees API (`GET /repos/{owner}/{repo}/git/trees/{branch}?recursive=1`) to avoid N+1 request patterns during directory traversal. Blob content is fetched independently via `raw.githubusercontent.com` rather than the Contents API, trading atomicity for reduced payload overhead (no base64 encoding round-trip).

File inclusion is governed by an allowlist filter over two predicates:
- **Filename allowlist**: `{readme.md, package.json, requirements.txt}`
- **Extension allowlist**: `{.py, .js, .ts}`

Decoding uses `errors="replace"` to guarantee UTF-8 well-formedness against binary blob contamination without raising on malformed byte sequences.

### 2. Commit Forensics Engine

Commit metadata is paginated (100/page, 3-page ceiling = 300-commit sampling window) via `GET /repos/{owner}/{repo}/commits?sha={branch}`. From the raw commit graph, the following derived quality signals are computed:

- **Trivial message ratio**: lexical matching of first-line commit messages against a curated low-information stopword set (`{fix, update, wip, ...}`), case-folded and punctuation-stripped
- **Temporal dispersion**: `active_days` (cardinality of distinct commit dates) vs. `span_days` (wall-clock delta between first and last commit)
- **Commit density**: `commits_per_active_day`, a proxy for burst-commit behavior
- **Last-day concentration coefficient**: `last_day_commit_pct`, the modal-day commit count normalized by total commit count — a high-recall heuristic for detecting single-session "dump" submissions that violate the iterative-development assumption implicit in hackathon judging criteria
- **Author cardinality**: deduplicated by commit author email/name, used as a proxy for team participation breadth

These signals are **not** fed to the LLM as raw features for it to interpret opaquely — they are pre-aggregated into a structured summary block and injected into the system prompt as grounding context, constraining the model's `commit_quality` score to be defensible against objective, auditable metrics rather than pure stylistic inference over commit message prose.

### 3. Generative Scoring Pipeline (`summarizer.py`)

Structured output is enforced via `response_mime_type="application/json"` on the `GenerateContentConfig`, eliminating brittle regex/markdown-fence post-processing that plagues unconstrained LLM output parsing. The system prompt encodes a fixed five-dimensional rubric:

```
{originality, technical_execution, completeness, commit_quality, documentation}
```

Each dimension resolves to a `{score: int[1,10], reason: str}` tuple, plus a top-level `overall_score` (weighted aggregate) and a `red_flags: list[str]` field for anomaly surfacing. The prompt explicitly biases the score distribution toward the 4–7 range to counteract the well-documented sycophancy/leniency bias of instruction-tuned LLMs when acting as evaluators (cf. RLHF-induced positivity skew in autorater literature).

### 4. Plagiarism & Originality Detection (`similarity.py`)

A two-tier similarity detection strategy trades off recall vs. semantic robustness:

**Tier 1 — Lexical hashing (exact/near-exact match, O(1) comparison):**
Source files are normalized via comment-stripping (regex-based, language-agnostic across `#`, `//`, `/* */` delimiters), whitespace collapsing, and case-folding, then SHA-256 hashed per-file. Cross-repository overlap is computed as:

```
overlap_pct = |hash(A) ∩ hash(B)| / min(|hash(A)|, |hash(B)|)
```

The `min()` denominator ensures a small repository fully subsumed into a larger one still triggers a high overlap coefficient — asymmetric plagiarism (partial copy into a larger novel codebase) is a common evasion pattern that a symmetric Jaccard-style denominator would under-detect.

**Tier 2 — Dense embedding similarity (semantic/paraphrase-robust):**
Concatenated README + source text (truncated to 20K characters to bound embedding API token cost) is projected into a dense vector space via `gemini-embedding-001`. Cross-repository semantic distance is computed as standard cosine similarity:

```
sim(u, v) = (u · v) / (‖u‖₂ · ‖v‖₂)
```

This tier catches renamed-variable / restructured / translated derivative works that evade hash-based detection entirely, at the cost of requiring an additional embedding API round-trip per judged repository. A dual-threshold gate (`hash_pct > 20 OR embed_pct > 85`) triggers a similarity warning, balancing false-positive suppression against recall on paraphrased plagiarism.

Fingerprints are accumulated in `st.session_state.fingerprints` as an in-memory, session-scoped corpus — each newly judged repository is compared against the full prior fingerprint set before being appended, giving O(n) pairwise comparisons per judged repo and O(n²) total across a full judging session (acceptable at hackathon-scale submission volumes of low hundreds).

### 5. Temporal Provenance Verification

A lightweight anti-fraud heuristic compares the repository's earliest commit timestamp against an organizer-supplied hackathon start date:

```python
if first_commit_date < hackathon_start:
    flag_as_pre_existing_work()
```

This addresses a specific and common integrity violation class: teams submitting substantially complete pre-existing projects as fresh hackathon work. The check is intentionally naive (single-commit boundary comparison) rather than distributional, prioritizing interpretability and zero false-negative risk on the clearest violation pattern over sensitivity to more subtle timeline manipulation (e.g., rebased/squashed history, which is a known limitation — see below).

---

## State Management & Frontend Architecture

The Streamlit application uses `st.session_state` as the sole persistence layer — there is **no backing database**; all state (leaderboard entries, fingerprint corpus) is ephemeral and scoped to the browser session's WebSocket connection lifetime. This is an intentional simplicity/durability tradeoff appropriate for single-session, single-judge hackathon evaluation workflows, but precludes multi-judge concurrent aggregation without an external state layer (see Roadmap).

UI theming bypasses Streamlit's default component styling via raw CSS injection (`st.markdown(..., unsafe_allow_html=True)`), implementing a custom design token system (CSS custom properties for gradient stops, typography scale via `Space Grotesk` / `JetBrains Mono` webfonts) rather than relying on Streamlit's constrained theming API.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.11 |
| Frontend framework | Streamlit ≥1.57.0 |
| LLM inference | `google-genai` SDK ≥2.6.0 (Gemini 3.5 Flash-Lite, `gemini-embedding-001`) |
| HTTP client | `requests` ≥2.34.2 |
| Numerical computing | `numpy` ≥1.26.0 (cosine similarity, vector ops) |
| Data manipulation | `pandas` ≥2.2.0 (commit timeline aggregation for `st.bar_chart`) |
| Hashing | `hashlib` (SHA-256, stdlib) |

---

## Known Limitations & Threat Model Gaps

- **Squash/rebase evasion**: temporal provenance verification relies on commit timestamp integrity, which is trivially defeated by `git commit --amend` or history rewriting prior to submission.
- **Rate-limit ceiling**: unauthenticated GitHub API access caps at 60 req/hr; each judged repository consumes ~3–5 requests (tree, N raw blobs, commits), practically limiting unauthenticated throughput to ~12–15 repos/hour without a PAT.
- **Embedding truncation**: the 20K-character truncation window may cause information loss on large monorepos, biasing similarity scores toward whatever content survives truncation (typically top-of-tree files).
- **No persistent corpus**: cross-session plagiarism detection is impossible without externalizing the fingerprint store (see Roadmap).
- **LLM non-determinism**: despite `response_mime_type` constraint, score outputs are not guaranteed bit-identical across repeated invocations on identical input (temperature is not pinned to 0 in the current `GenerateContentConfig`).

---

## Roadmap

- [ ] Externalize fingerprint corpus to a persistent store (SQLite/Postgres) for cross-session plagiarism detection at organizer scale
- [ ] GitHub PAT-authenticated requests to raise rate ceiling to 5,000 req/hr
- [ ] Pin `temperature=0` for score reproducibility
- [ ] CSV batch-judging mode for organizer-scale submission processing
- [ ] Network-graph visualization of pairwise similarity clusters (force-directed layout)
- [ ] README-to-implementation consistency scoring (claimed vs. actually-imported dependency cross-check)

