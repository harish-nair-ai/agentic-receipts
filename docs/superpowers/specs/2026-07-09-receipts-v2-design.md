# Receipts v2 — A Calibrated, Uncertainty-Aware Fact-Checker for AI-Agent Claims

**Status:** Design (in review) — 2026-07-09
**Author:** Harish Nair S (with Claude)
**Scope:** One implementation cycle, two phases. Ships on a feature branch → PR (not `main`). No PyPI publish this cycle.

> **Product definition:** Receipts is an automated, calibrated, uncertainty-aware **fact-checker for AI-agent completion claims**. When an agent says "done," Receipts treats each claim like a FEVER fact-verification problem — retrieve evidence, decide **SUPPORTED / REFUTED / NOT-ENOUGH-INFO**, attach a calibrated score and a specific natural-language critique.

---

## 1. Context & the problem (with real numbers)

Receipts v1 is a zero-config Claude Code Stop hook: it extracts completion claims from a session, matches them against deterministic evidence (test exit codes, file writes), and asks an **independent** LLM judge (maker ≠ checker; Gemini by default) for a **discrete** verdict per fuzzy claim. Output is a Rich terminal receipt card + local `receipts stats`.

**The problem is real, large-scale, measured, and getting relatively worse.** The 20,574-session study *"How Coding Agents Fail Their Users"* (arXiv 2605.29442; 16,118 validated misalignment episodes, 93% precision) found:

- **"Inaccurate Self-Reporting" = 22.58% of all misalignment episodes** — agents prematurely claiming success/completion without verification (the 3rd-largest failure category).
- **False success claims are *rising* in relative share** — "training improvements address code correctness more effectively than behavioral adherence or honest self-assessment." Models get better at coding, **not** at honestly reporting done.
- **91.49% of resolutions required a human to intervene** — the human is currently the only verifier. That's the labor Receipts automates.

The market thesis (2026 landscape): *"The core bottleneck is no longer code generation speed. It is verification capacity."* Anthropic itself now recommends our exact pattern — a deterministic hook for hard claims plus a separate model pass for ambiguous ones — but as **DIY guidance, not a zero-config product.** That gap is the opening.

> **Note (v1 correctness debt):** the README's "~47% of the time" figure is unsourced. Replace it during implementation with the citable **22.58%** (S7, arXiv 2605.29442) or the 42%-false-positive case study. Tracked in §14.

---

## 2. Evidence base & technique fusion

A real product mixes multiple validated techniques. Every layer below is independently grounded (full citations in §16).

| Layer | Source(s) | Contribution |
|---|---|---|
| Calibrated continuous score (logprob expectation over an A–T 20-level scale), K repeated passes, criteria decomposition, PPT best-of-N | **2607.05391** (base paper) + released code | The scoring engine and inference-time scaling dimensions |
| Probabilistic (expectation) scoring **beats discrete** on judge sensitivity | RewardBench 2 result via LLM-judge surveys | Justifies replacing the discrete verdict with a calibrated score |
| **Claim-as-fact-verification**: retrieve→verify, three labels SUPPORTED / REFUTED / **NOT-ENOUGH-INFO** | **FEVER** (1803.05355), NLI/SummaC/FactCC faithfulness lineage | Rigorous backbone; NEI = our honest abstention; groundedness = our evidence layer |
| **Independent critic > self-review** (catches bugs humans miss, preferred >80%) | **CriticGPT** — "LLM Critics Help Catch LLM Bugs" (2407.00215) | Empirical validation of maker ≠ checker; motivates a NL critique on every receipt |
| **Self-critique collapses; external verification gains** | 2310.01798, 2509.17995 | Why the checker must be external; why Phase-2 external ranking of fixes works |
| **Uncertainty-aware verification, abstention, cost cascades** | Overconfidence-in-judge (2508.06225), abstention (2510.24020), cascades (2506.11887), confidence-gated verification (2602.18447) | Variance = confidence signal → adaptive K + abstain-over-guess (meta-honesty) |
| **Weak-verifier ensemble** shrinks the generation–verification gap | 2506.18203 | Principled combination of criteria + deterministic evidence |

**Dropped from the base paper** (validated as weak for us): the RL reward-shaping section ships **zero** code; "TurboAgent" is a FastAPI reverse proxy (`ANTHROPIC_BASE_URL`) that **multiplies inference cost by N** — not a real hook/plugin. We take neither.

---

## 3. Positioning: the AI lie detector — that never lies

Voice and narrative: **the lie detector for AI coding agents** (rides the documented, rising "agents falsely claim done" pain). But the credibility of a lie detector is destroyed the moment it makes a false accusation. So the defining product property is **meta-honesty**:

- Three-way verdict with **real abstention** — when evidence is insufficient, Receipts says **NOT-ENOUGH-INFO**, never a forced accusation. Grounded in FEVER's NEI label + the abstention literature.
- Every REFUTED verdict is backed by a calibrated score **and** a specific critique pointing at the contradicting evidence — a *defensible* accusation (CriticGPT-style), not a vibe.

**Growth = utility + a show-don't-tell launch, not a share button.** The research says the edge is solving a real, measured, expensive problem; virality (if any) is organic — the terminal card is good enough that people screenshot it themselves. We do **not** build a share/badge feature (see §4, §10).

---

## 4. Goals / Non-Goals

### Goals
- Replace the discrete judge with a **calibrated continuous score** per claim and a **three-way calibrated verdict** (SUPPORTED / REFUTED / NOT-ENOUGH-INFO), backed by the technique stack in §2.
- **Meta-honesty:** abstain instead of guessing when confidence is low; every REFUTED carries a specific critique.
- **Adaptive inference-time scaling:** variance-gated K (cheap on clear claims, thorough on ambiguous), criteria decomposition as a weak-verifier ensemble.
- Preserve **zero-config**: graceful fallback when calibrated logprobs are unavailable — never crash, never require setup.
- **Phase 2:** self-healing — on a REFUTED claim, generate candidate fixes, rank with PPT via the external verifier, present the winning diff, opt-in auto-apply.

### Non-Goals (YAGNI / explicitly dropped)
- No RL / DSRL-SAC / GRPO reward shaping (no released code).
- No reverse-proxy / `ANTHROPIC_BASE_URL` interception / cost-multiplying "turbo" mode.
- No React/Vite dashboard.
- **No `receipts share` (PNG) and no `receipts badge`.** Manufactured virality, no evidence of value, maintenance cost. The terminal card is the organic shareable artifact.
- **Honesty Index is out of core scope** — demoted to an optional post-launch data-moat track (§11). It does not gate or shape the architecture this cycle.
- No PyPI publish this cycle. No push to `main` — feature branch + PR only.

---

## 5. Architecture overview (science-mapped pipeline)

```text
Session transcript
  → Claim Extraction            (claims.py, unchanged)        [FEVER: claim identification]
  → Evidence Matching           (evidence.py, unchanged)      [FEVER: retrieval / groundedness]
  → Verifier core (per claim)   (verifier.py ← NEW)
       1. Deterministic entailment fast-path
            hard evidence directly contradicts/supports? → decide with certainty, no LLM
       2. Calibrated independent scorer     (scorer.py ← NEW)
            A–T logprob expectation (Eq 3.1) over criteria (Spec/Output/Error = weak-verifier ensemble)
            → continuous score + NL critique (CriticGPT-style)
       3. Uncertainty control       (adaptive K; abstain → NOT-ENOUGH-INFO on low confidence)
       → three-way calibrated verdict + score + critique
  → Receipt card                 (render.py, upgraded)
  → Local store + stats          (stats.py, extended)

Phase 2 (self-healing):
  REFUTED claim → generate N candidate fixes → external verifier scores each
       → PPT best-of-N select (pivot_tournament.py ← NEW)
       → present winning diff → [opt-in] auto-apply (retry.py ← NEW)

Optional post-launch:
  [opt-in] aggregate publish → AI Agent Honesty Index (§11)
```

`judge.py` is refactored into the new `verifier.py`/`scorer.py` path; its discrete-verdict logic survives only as the deepest fallback rung (§8.5). Maker ≠ checker is enforced at the verifier boundary: the checker model must differ from the audited agent. Default checker is an OpenAI-compatible logprob-capable model, with Vertex-Gemini support; Anthropic is allowed only when the audited agent is not Anthropic.

---

## 6. Why not just a native hook?

Receipts *is* installed as a Claude Code Stop hook — the hook is the **delivery mechanism**, not the product. "A 5-line `pytest --exit-code` hook already blocks on failing tests" is true for the deterministic ~20% (our §8.1 fast-path), and that is the *floor*, not the pitch.

| | Naive self-check hook | **Receipts** |
|---|---|---|
| Verifies… | a thing *you* remembered to script | the **claims the agent actually made**, per session (FEVER-style) |
| Fuzzy claims ("refactored", "improved error handling", "added validation") | ❌ no exit code exists | ✅ calibrated independent verifier + critique |
| No test suite / unknown command | ❌ nothing to run | ✅ claim-driven evidence matching still works |
| Independence | maker == checker (self-critique **collapses** — 2310.01798) | ✅ maker ≠ checker (CriticGPT-validated) |
| Honesty about its own limits | forced pass/fail | ✅ abstains to NOT-ENOUGH-INFO |
| Setup | write & maintain per repo | ✅ `pip install` + `receipts install`, zero-config |

The unit of value is **the gap between what the agent said and what it did** — most of which no native hook can close.

---

## 7. Phase 1 — the verifier

## 8. Components (Phase 1)

### 8.1 Deterministic entailment fast-path (`verifier.py`, NEW)
Before any LLM call: if hard evidence directly decides the claim, decide with certainty and skip the model. `pytest`/`npm test`/`cargo`/`go`/`jest` exit ≠ 0 contradicting an "all tests pass" claim → **REFUTED** (score clamped to 0). A matching diff/file-write fully supporting a "created file X" claim → **SUPPORTED** (score 100). This is the cheap, exact FEVER-retrieval case and the "floor" a native hook also covers. Only ambiguous claims reach the LLM.

### 8.2 `scorer.py` (NEW) — calibrated continuous score
For claims the fast-path can't decide, the independent checker emits a single scoring token on a fixed 20-level ordinal scale (`A`…`T`, A = fully refuted, T = fully supported). We request `top_logprobs` (≤20) and compute the calibrated score as the probability-weighted expectation (paper Eq 3.1; mirrors `fine_grained_reward.py::extract_score`):

```
score01 = Σ_i (value(tok_i) · p_i) / Σ_i p_i      # renormalized over returned candidates
value(letter) = (ord(letter) − ord('A')) / 19     # A→0.0 … T→1.0 ; p_i = exp(logprob_i)
```

Reported 0–100 per claim. The checker also returns a short **natural-language critique** (CriticGPT-style) naming the specific supporting/contradicting evidence. `G = 20` is a constant (matches released code; not user-tunable this cycle).

**Interface:**
```python
@dataclass
class ClaimVerdict:
    claim: Claim
    label: FactLabel            # SUPPORTED | REFUTED | NOT_ENOUGH_INFO
    score: float                # 0..100 calibrated
    confidence: float           # 0..1 from cross-pass agreement + logprob dispersion
    per_criterion: dict[str, float]
    critique: str               # specific NL critique / evidence pointer
    method: ScoreMethod         # LOGPROB | SAMPLED
    passes: int                 # K actually run (adaptive)

def verify_claim(claim, evidence, transcript_context, config) -> ClaimVerdict: ...
def verify_session(claims, evidence, transcript_context, config) -> SessionReport: ...
```

### 8.3 Criteria decomposition as a weak-verifier ensemble (`criteria.py` + `criteria/*.md`, NEW)
For code claims, score three sub-criteria (paper's coding triad; framed as a weak-verifier ensemble per 2506.18203):
- `specification.md` — does the change match what was asked/claimed?
- `output_match.md` — does observed output/behavior match the claim?
- `error_signal.md` — are there error signals (failing tests, tracebacks, non-zero exits) contradicting the claim?

Each criterion is an independent Markdown prompt fragment shipped in-package. `criteria.py` selects the set by `ClaimType`, runs each through the scorer, and combines (error-signal is min-biased so one strong contradiction sinks an over-optimistic claim). Non-code/generic claims use a single holistic criterion.

### 8.4 Uncertainty control: adaptive K + abstention
- **Adaptive K** (cascade / confidence-gated): run 1 pass; if cross-pass/criterion variance or logprob dispersion exceeds a threshold, escalate up to `K_max` (default 3, `RECEIPTS_SCORE_PASSES`). Cheap on clear claims, thorough on ambiguous ones. Passes run concurrently when escalated.
- **Abstain over guess:** map `confidence` (from cross-pass agreement + logprob dispersion) — when it is below `RECEIPTS_MIN_CONFIDENCE`, output **NOT_ENOUGH_INFO** regardless of the point score. This is the meta-honesty guarantee: no forced accusations.
- Score → label bands (applied only when not abstaining): `≥ 85` SUPPORTED · `< 45` REFUTED · in-between → NOT_ENOUGH_INFO. Deterministic fast-path decisions override bands.

### 8.5 Graceful fallback (zero-config guarantee)
Logprobs require Vertex Gemini or an OpenAI-compatible endpoint exposing `top_logprobs ≤ 20`; **Anthropic exposes no logprobs**. Resolution order per session:
1. Checker supports logprobs → **LOGPROB** method (calibrated expectation).
2. No logprobs (e.g. Anthropic-only key) → **SAMPLED** method: sample the scoring letter at low temperature a few times, average numeric values. Lower fidelity, labeled `sampled` on the card, still continuous.
3. No checker key → deterministic-evidence-only receipt (fast-path decides hard claims; fuzzy claims → NOT_ENOUGH_INFO). Never crash.

### 8.6 Upgraded receipt card (`render.py`)
Header shows the aggregate **Verified-Done Score: NN/100** and counts of ✅ supported / ❌ refuted / ❔ not-enough-info. Per claim: label glyph, continuous score, the critique one-liner, per-criterion mini-bars for code claims, and a `method` tag (`calibrated` vs `sampled`). Keeps the existing Rich box aesthetic; REFUTED rows are rendered as evidence (verbatim claim → contradiction) so the card is inherently screenshot-worthy without a separate share tool.

### 8.7 `stats.py`
Persist continuous `score`, `label`, `confidence`, `method`, `passes` alongside existing fields. `receipts stats` reports numeric aggregates. Backward-compatible: old records without these are treated as verdict-only.

---

## 9. Phase 2 — self-healing (headline follow-on)

Triggered on a **REFUTED** claim (not on NOT_ENOUGH_INFO — we don't "fix" what we're unsure about).

1. **Generate** N candidate fixes (default 3) from the refuting evidence + relevant transcript slice.
2. **Score** each candidate through the external verifier (§8.2–8.3). External ranking is what makes this work — self-correction alone collapses (2310.01798); the generation–verification gap is closed by an independent verifier (2506.18203).
3. **Select** best-of-N with **PPT** (`pivot_tournament.py`, NEW) — Bradley-Terry pivots `1/(1+exp(-(ra−rb)))`, `DEFAULT_PIVOTS=2`, O(Nk); ported from the reference repo. Pairwise ranking also sidesteps the known instability of pointwise scores.
4. **Present** the winning candidate as a unified diff in the receipt.
5. **Apply** only on **opt-in** (`RECEIPTS_AUTOFIX=1` or interactive confirm). Never auto-writes by default.

`retry.py` (NEW) orchestrates generate → score → PPT → present → (opt-in) apply.

---

## 10. Data model changes (`models.py`)
- Add `FactLabel` enum (`SUPPORTED`, `REFUTED`, `NOT_ENOUGH_INFO`) and `ScoreMethod` (`LOGPROB`, `SAMPLED`). Keep the existing `Verdict` for back-compat mapping.
- Add `ClaimVerdict` / `SessionReport` (dataclasses in `verifier.py`; Pydantic where persisted).
- Persist `score`, `label`, `confidence`, `method`, `passes`. Backward-compatible with v1 records.

---

## 11. Optional post-launch: AI Agent Honesty Index (out of core scope)
The one piece with a genuine **data moat** (only Receipts holds cross-agent claim-verification data) and a science-backed angle: a **longitudinal benchmark** tracking the documented "false-reporting is rising" trend, per audited agent (Claude Code vs Cursor vs Copilot vs Codex), **agent-vs-agent, never user-vs-user**. Thin Cloudflare Worker + KV + a static ranked page; `receipts publish` sends **aggregate numbers only**, keyed by audited-agent identity (a known enum, not free text), opt-in and off by default. **Not built this cycle** — listed so the data model (§10) stays forward-compatible. Revisit only after the core verifier ships and is genuinely useful.

---

## 12. Configuration
- `RECEIPTS_SCORE_PASSES` (default 3) — `K_max` for adaptive escalation.
- `RECEIPTS_MIN_CONFIDENCE` (default e.g. 0.6) — abstention threshold.
- `RECEIPTS_CHECKER_MODEL` / existing provider auto-detection — checker selection; enforce maker ≠ checker.
- `RECEIPTS_AUTOFIX` (default 0) — Phase 2 opt-in apply.
- `RECEIPTS_BLOCK` (existing) — block exit on refuted/unverified.
- Fast mode (K=1, single criterion) for latency-sensitive users.

---

## 13. Privacy & safety (hard constraints)
- **Local-first.** All verification and receipts are computed and stored locally. Nothing is transmitted unless the user runs an opt-in command.
- **maker ≠ checker** enforced in code; the audited agent's provider cannot be the checker.
- **(If the Honesty Index is ever built)** publish is **aggregate-only** — a fixed schema of numbers + audited-agent enum + checker model name + anonymous contributor id; Worker rejects any out-of-schema field; transcript/claim text/code/paths/prompts are **never** sent; explicit opt-in, off by default.
- **No PyPI publish, no push to `main`** this cycle.

---

## 14. Testing strategy
- **Scorer math:** unit-test Eq 3.1 expectation on synthetic logprob distributions (known inputs → known score), incl. renormalization and single-candidate edges.
- **Three-way labeling + abstention:** assert low-confidence inputs → NOT_ENOUGH_INFO; high-agreement supporting/contradicting → SUPPORTED/REFUTED. This is the meta-honesty guarantee — cover it hard.
- **Adaptive K:** clear claim resolves in 1 pass; high-variance claim escalates to `K_max`.
- **Deterministic fast-path:** `pytest` exit ≠ 0 → REFUTED without an LLM call; full file-write support → SUPPORTED.
- **Criteria ensemble:** selection per `ClaimType`; a strong error signal sinks an optimistic claim.
- **Fallback ladder:** logprob-capable / sampled-only / no-key configs → correct method, no crash.
- **PPT:** ranking correctness on synthetic ratings; O(Nk) pivot count.
- **README stat fix:** verify the "47%" is replaced with the cited 22.58% (arXiv 2605.29442).
- **End-to-end (per global guidance):** reproduce a real Claude Code session with a genuinely refuted "all tests pass" claim and a genuinely ambiguous "improved error handling" claim; assert REFUTED-with-critique and NOT_ENOUGH_INFO respectively, plus the Phase-2 proposed diff — before trusting unit tests alone.

---

## 15. Build plan (agent allocation)
Token-heavy implementation → **Sonnet** agents on isolated branches/worktrees; **Fable 5** reserved only for the go/no-go calibration review and un-sticking a stuck agent (high-stakes, expensive).

- **Agent A (Sonnet):** `verifier.py` (fast-path + three-way labeling + adaptive K + abstention) + `scorer.py` (Eq 3.1 math) + `FactLabel`/`ScoreMethod` model changes + fallback ladder + unit tests.
- **Agent B (Sonnet):** `criteria.py` + `criteria/*.md` triad + weak-verifier combination + tests.
- **Agent C (Sonnet):** `render.py` card upgrade (three-way + critique + per-criterion bars) + `stats.py` numeric aggregates + README "47%" fix.
- **Agent D (Sonnet, Phase 2):** `pivot_tournament.py` + `retry.py` + self-healing flow + tests.
- **Fable 5 checkpoint:** review calibration correctness, abstention thresholds, and maker ≠ checker enforcement before the PR; intervene on any stuck agent.

Integration: each agent branches from the feature branch; results merged and run through the end-to-end test before opening the PR against `harish-nair-ai/agentic-receipts` (PR, not `main`).

---

## 16. Risks & mitigations
- **Pointwise score instability** (documented). → K-averaging + logprob expectation stabilize pointwise; PPT (pairwise) for Phase-2 ranking; abstain on high variance.
- **Judge overconfidence** (2508.06225). → confidence from cross-pass agreement + logprob dispersion, not the model's self-reported confidence; abstention threshold.
- **Logprob availability varies.** → fallback ladder (§8.5); label method on card; never require setup.
- **Calibration drift across checker models.** → fixed 20-level scale + renormalization; Fable 5 go/no-go on real sessions.
- **Latency from K× and criteria×.** → adaptive K (most claims resolve in 1 pass) + concurrency + fast mode.
- **Meta-honesty regression (false accusation).** → three-way labeling + abstention tests are first-class (§14).
- **Scope creep back to RL/proxy/share-features.** → explicitly dropped in §4; do not re-introduce.

---

## 17. References
- **2607.05391** — LLM-as-a-Verifier (base paper: calibrated logprob expectation, scaling dimensions, PPT). Released code: `llm-as-a-verifier/llm-as-a-verifier`.
- **2605.29442** — How Coding Agents Fail Their Users: 20,574-session misalignment analysis (22.58% inaccurate self-reporting; rising trend).
- **1803.05355** — FEVER: Fact Extraction and VERification (SUPPORTED/REFUTED/NOT-ENOUGH-INFO; retrieve→verify). Related: FactCC, QAGS, SummaC faithfulness.
- **2407.00215** — LLM Critics Help Catch LLM Bugs (CriticGPT): independent critics beat human/self review.
- **2310.01798** — Large Language Models Cannot Self-Correct Reasoning Yet.
- **2509.17995** — Variation in Verification: self-critique collapses; external verification gains.
- **2506.18203** — Shrinking the Generation-Verification Gap with Weak Verifiers.
- **2508.06225** — Overconfidence in LLM-as-a-Judge: diagnosis + confidence-driven solution.
- **2510.24020** — Teaching LLMs to Abstain via Fine-Grained Semantic Confidence.
- **2506.11887** — Cascaded Language Models for Cost-effective Human-AI Decision-Making.
- **2602.18447** — ConfSpec: confidence-gated verification.
