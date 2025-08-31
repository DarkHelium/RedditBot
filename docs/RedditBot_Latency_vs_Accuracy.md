# RedditBot — Latency vs. Accuracy Tradeoff (Tech Note)

**Owner:** Anav Madan  
**Repo:** `DarkHelium/RedditBot`  
**Context:** Chrome extension UI (popup) + analysis script; aims to predict success of Reddit posts using factors like *title wording, post length, media presence, and sentiment*. The UX goal is **snappy, in-context guidance** before/while a user posts.

---

## 1) Product Goals & Latency Budgets
- **Primary UX:** Feedback **inline, near-instant** as the user drafts a post / hovers a link.  
- **Targets:** `P50 ≤ 300 ms`, `P95 ≤ 1.0 s` for an initial score; high-accuracy refinement may arrive **asynchronously** within `≤ 3–5 s`.
- **SLOs:** 99.9% success (no hard failures), graceful fallbacks, deterministic outputs for identical inputs (versioned features/models).

**User-visible states**
1. *Instant* → lightweight score + hints (no spinner).  
2. *Improve* → “Refining…” badge; refined score & rationale within a few seconds.  
3. *Offline/Rate-limited* → cached guidance + non-blocking warning.

---

## 2) Options Overview

| Option | Description | Latency | Accuracy | Cost | Ops Complexity | Privacy |
|---|---|---:|:--:|---:|---:|:--:|
| **A. Heuristics (Rules)** | Pure JS rules on features (length, caps, emojis, time-of-day, media flag, basic sentiment lexicon) | **~5–20 ms** | Low–Med | **$** | **★** | Local |
| **B. Lightweight ML** | Logistic/Linear model or Gradient Boosted Trees on engineered features; small WASM bundle or local Node | **~20–80 ms** | Med | **$$** | **★★** | Local |
| **C. LLM API Only** | Call provider for semantic scoring/explanations | **0.7–3.0 s** | **High** | **$$$** | **★** | Remote |
| **D. **Hybrid Two‑Pass** (Recommended)** | A→B for instant score; C optional for refinement + rationale | **Instant + Async** | **High** | **$$$** | **★★** | Split |

> **Why Hybrid?** Users get instant guidance (A/B) and richer, more accurate insights (C) **without blocking** the UI. Costs are controlled via sampling/throttles.

---

## 3) Features & Signals (current + near‑term)
- **Textual:** title length, type/token ratio, sentiment (lexicon or tiny model), question/exclamation usage, presence of numbers/‘guide’ verbs, *click‑bait* patterns.  
- **Media:** image/video flag, link domain whitelist/blacklist.  
- **Context:** subreddit baseline (historical median upvotes/engagement bucket), time-of-day/day-of-week.  
- **Derived:** normalized title length, normalized sentiment, media × subreddit interaction, posting-time z‑score.

> Keep features **stable & versioned** (e.g., `features_v3`) to make caches reproducible.

---

## 4) Architecture Paths

### A) Client‑Only (Rules / Tiny Model)
- **Where:** Chrome extension popup/content script.  
- **Pros:** Zero server bill; fastest; works offline.  
- **Cons:** Limited semantic accuracy; bundle size constraints; private weights are extractable.

### B) Local/Edge Service (Node microservice)
- **Where:** Localhost dev or edge/region serverless (e.g., Vercel/Cloudflare/AWS).  
- **Pros:** Small models, richer features, central A/B, analytics.  
- **Cons:** Adds infra; handle quotas, cold starts.

### C) LLM Provider (Refinement)
- **Where:** Provider API.  
- **Pros:** Best semantic judgments + rationale generation.  
- **Cons:** Latency, cost, rate limits; privacy review required.

**Recommended:** **Client A/B for instant**, queue optional **C** for high‑value contexts (e.g., user clicks “Get Deep Review” or confidence < threshold).

---

## 5) Caching & Throttling

- **Key:** `cache_key = hash(subreddit, title_norm, media_flag, version)`  
- **Policy:** `stale‑while‑revalidate` (SWR) → show cached score immediately; refresh in background.  
- **TTLs:** 24h default; 1h for hot subreddits; manual bust on version/model changes.  
- **Rate Limits:** Per‑origin + per‑user with exponential backoff.  
- **Deduping:** In‑flight promise map to prevent thundering herds.

---

## 6) Model Choices & Prompts

- **B (Lightweight ML):** Start with **Logistic Regression** (fast, stable) → consider **XGBoost** if non‑linearity helps. Export weights to WASM or JSON for deterministic client eval.  
- **C (LLM):** Prompt yields **score [0–100]** + **three evidence bullets**. Use few‑shot examples for subreddit‑aware tone.  
- **Determinism:** Fix temperature/seed; clamp outputs by confidence; emit `(score, confidence, rationale[], version)`.

---

## 7) Measurement Plan

- **Dataset:** 5–10k historical posts across 10 subreddits; label with normalized engagement (e.g., top‑decile = “win”).  
- **Metrics:** AUC/PR for accuracy; `P50/P95` latency for each path; cost/post.  
- **Ablations:** Rules vs Rules+Sentiment vs ML; ML vs ML+Context; Hybrid with/without LLM refine.  
- **Acceptance:** `AUC ≥ 0.70` for A/B instant model; hybrid refine increases precision@top‑K by `≥ +7%` with `≤ 1.5 s` extra P95 latency.

---

## 8) Rollout & Guardrails

1. **Phase 1 (Instant‑Only)**: Ship A/B with SWR cache. Telemetry: latency, confidence, click‑through on “Deep Review”.  
2. **Phase 2 (Hybrid)**: Enable LLM refine for low‑confidence cases or on‑demand. Add cost caps and daily budget.  
3. **Phase 3 (Tuning)**: Subreddit‑specific thresholds; prompt and feature updates via versioned configs.  
4. **Privacy & Compliance:** No PII; redact text on logs; add “opt‑out” toggle.

---

## 9) Recommendation

Adopt **Hybrid Two‑Pass**: deliver **instant** rule/ML guidance, then **optional LLM refinement** under budget and latency caps. This preserves UX speed, scales transparently, and aligns with a newsroom‑grade bar for **explainability** (evidence bullets) and **operational reliability** (SLOs, SWR caches, rate limits).

