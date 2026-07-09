# Receipts v2 — Calibrated Verifier + The AI Lie Detector

**Status:** Approved (design) — 2026-07-09
**Author:** Harish Nair S (with Claude)
**Scope:** One implementation cycle, two phases. Ships on a feature branch → PR (not `main`). No PyPI publish this cycle.

---

## 1. Context

Receipts v1 is "the verified-done layer for AI coding agents": a zero-config Claude Code Stop hook that extracts completion claims from a session, matches them against deterministic evidence (test exit codes, file writes), and asks an **independent** LLM judge (maker ≠ checker; Gemini by default) for a **discrete** verdict per fuzzy claim (`verified` / `unverified` / `refuted` / `skipped`). Output is a Rich terminal receipt card + local `receipts stats`.

arXiv **2607.05391** ("LLM-as-a-Verifier") contributes a genuinely useful core we do not yet have: a **calibrated continuous score** and three orthogonal **inference-time scaling dimensions** that make that score more reliable. We validated the paper against its released code (`llm-as-a-verifier/llm-as-a-verifier`):

- **Implemented & reusable:** continuous score from logprob expectation (Eq 3.1), score-granularity scale (A–T, 20 levels), repeated-evaluation averaging (K passes), criteria decomposition (coding triad: Specification / Output Match / Error-Signal Detection), and a Probabilistic Pivot Tournament (PPT) for cheap best-of-N ranking (Bradley-Terry pivots, O(Nk)).
- **Paper-only / weak for us — dropped:** the RL reward-shaping section ships **zero** code; "TurboAgent" is a FastAPI reverse proxy (via `ANTHROPIC_BASE_URL`) that **multiplies inference cost by N** — not a real hook/plugin. We take neither.

**Our unique wedge (not in the paper):** zero-config *consumer* distribution + maker ≠ checker discipline + a genuinely organic viral mechanic. The paper builds a verifier for researchers; we build a verified-done *product* for every coding-agent user — and we make the moment it fires inherently shareable.

### Positioning: the AI lie detector (gotcha-forward)

The viral object is **not** a personal score card (nobody screenshots their own report card) and **not** a user-vs-user leaderboard (vanity noise). Validated against how OpenClaw and HeyClicky actually went viral, the law is: *the shareable object is the product doing something jaw-dropping during real use, that the user captures because it makes them look interesting — never a badge we ask them to post.*

Receipts' jaw-drop moment is **catching the AI lying, at the instant of maximum surprise** — riding the hottest current narrative ("AI agents confidently claim done, then haven't"). Two viral surfaces, both organic:

1. **The "caught red-handed" receipt** — when a confident claim is refuted, the receipt is rendered as *evidence*, engineered to be screenshot-worthy. The hero cases are the claims **no script can check**: `Agent: "✅ Added input validation + improved error handling, done." → 🧾 REFUTED (score 18/100) — no try/except or validation added to the diff; the only new code is a print statement.` The calibrated scorer is what makes this a **defensible** accusation (every gotcha backed by evidence + a calibrated score), not a cheap dunk. Deterministic cases (`pytest exit 1`) are the *floor*, not the pitch — see §2.5.
2. **The AI Agent Honesty Index** — aggregate, opt-in, **agent-vs-agent** leaderboard: which coding agents most often falsely claim "done" (Claude Code vs Cursor vs Copilot vs Codex …). A controversial *industry* stat people quote-tweet — an ongoing data-journalism story and a data moat (only Receipts watches real sessions). Never user-vs-user.

Launch asset = a HeyClicky-style ~104-second clip: a real agent says "done, all green," Receipts instantly flashes ❌ REFUTED. Product copy and README carry the same lie-detector voice, kept credible by the evidence-first scoring.

### Goals
- Replace the discrete judge with a **calibrated continuous score** per claim and an aggregate **Verified-Done Score (0–100)** per session, backed by the paper's scaling dimensions — the credibility engine behind every gotcha.
- Preserve **zero-config**: if calibrated logprobs are unavailable, degrade gracefully to a sampled score — never crash, never require setup.
- Make the **refutation moment** shareable: `receipts share` (PNG "caught red-handed" receipt), `receipts badge` (SVG), and **opt-in** `receipts publish` (aggregate-only) feeding the **AI Agent Honesty Index**.
- **Phase 2:** self-healing — on a refuted claim, generate candidate fixes, rank with PPT, present the winning diff, opt-in auto-apply.

### Non-Goals (YAGNI / explicitly dropped)
- No RL / DSRL-SAC / GRPO reward shaping (no released code; out of product scope).
- No reverse-proxy / `ANTHROPIC_BASE_URL` interception / cost-multiplying "turbo" mode.
- No React/Vite dashboard. Terminal card + static Honesty Index page only.
- **No user-vs-user leaderboard, no "share your score" as the hook.** The score exists to make gotchas credible; virality comes from the refutation moment and the agent-vs-agent index.
- No PyPI publish this cycle. No push to `main` — feature branch + PR only.
- Honesty Index stores **aggregate numbers only**, never transcript content (see §7).

---

## 2. Architecture Overview

```text
Session transcript
   → Claim Extraction        (claims.py, unchanged)
   → Evidence Matching       (evidence.py, unchanged — free hard evidence)
   → Calibrated Scorer       (scorer.py  ← NEW, replaces judge.py's discrete verdict)
        ├─ Criteria decomposition   (criteria.py + criteria/*.md ← NEW)
        ├─ Continuous score via A–T logprob expectation (Eq 3.1)
        ├─ K repeated passes, averaged
        └─ Graceful sampled fallback when logprobs unavailable
   → Receipt (score card)     (render.py, upgraded)
   → Local store + stats      (stats.py, extended with numeric score)
   → [opt-in] share / badge / publish

Phase 2 (self-healing):
   Refuted claim → generate N candidate fixes → decomposed scorer on each
        → PPT best-of-N select (pivot_tournament.py ← NEW)
        → present winning diff → [opt-in] auto-apply (retry.py ← NEW)
```

Maker ≠ checker is enforced at the scorer boundary: the checker model must differ from the agent under audit (Claude Code). Default checker is an OpenAI-compatible logprob-capable model, with Vertex-Gemini support; Anthropic is allowed only when the audited agent is not Anthropic.

### 2.5 Why not just a native hook?

Receipts *is* installed as a Claude Code Stop hook — the hook is the **delivery mechanism**, not the product. The obvious objection is "a 5-line `pytest --exit-code` hook already blocks on failing tests." True — and that is the *floor*, the deterministic ~20% of claims, not the pitch. The defensibility line:

| | Naive self-check hook | **Receipts** |
|---|---|---|
| **Verifies…** | a thing *you* remembered to script | the **claims the agent actually made**, extracted per session |
| **Fuzzy claims** ("refactored", "improved error handling", "added validation") | ❌ no exit code exists | ✅ calibrated independent auditor scores them |
| **No test suite / unknown command** | ❌ nothing to run | ✅ claim-driven evidence matching still works |
| **Independence** | maker == checker (agent self-certifies — the failure mode itself) | ✅ maker ≠ checker enforced |
| **Aggregate signal** | invisible, per-project | ✅ cross-session/-agent **Honesty Index** |
| **Setup** | write & maintain per repo | ✅ `pip install` + `receipts install`, zero-config, any repo |

The unit of value is **the gap between what the agent said and what it did** — most of which (fuzzy claims, repos without tests, self-certification blind spots) no native hook can close. Receipts is the claim-auditing intelligence layer *on top of* the hook primitive.

---

## 3. Phase 1 — Calibrated Verifier + Shareable Score

### 3.1 `scorer.py` (NEW) — continuous calibrated score

Replaces `judge.py`'s discrete verdict as the primary path. `judge.py` stays as the fallback discrete judge (see 3.4).

**Score model (paper Eq 3.1).** The checker is prompted to emit a single scoring token on a fixed 20-level ordinal scale (letters `A`…`T`, `A` = worst / fully refuted, `T` = best / fully verified). We request `top_logprobs` (≤20) on that token position and compute the calibrated score as the probability-weighted expectation over the returned candidates:

```
score01 = Σ_i (value(tok_i) · p_i) / Σ_i p_i      # renormalized over returned candidates
```

where `value(letter) = (ord(letter) − ord('A')) / 19` maps A→0.0 … T→1.0, and `p_i = exp(logprob_i)`. Per-claim score is reported 0–100. This mirrors `fine_grained_reward.py::extract_score` in the reference repo.

**Interface:**
```python
@dataclass
class ClaimScore:
    claim: Claim
    score: float               # 0..100 calibrated
    verdict: Verdict           # discretized band (see 3.5) for card glyphs
    per_criterion: dict[str, float]   # criterion name -> 0..100
    method: ScoreMethod        # LOGPROB | SAMPLED (fallback)
    passes: int                # K actually run
    reasoning: str

def score_claim(claim, evidence, transcript_context, config) -> ClaimScore: ...
def score_session(claims, evidence, transcript_context, config) -> SessionScore: ...
```

`SessionScore` aggregates per-claim scores into the **Verified-Done Score (0–100)** (evidence-weighted mean; deterministically-refuted claims, e.g. `pytest` exit ≠ 0, hard-clamp their claim score toward 0 regardless of checker output).

### 3.2 Scaling dimension — repeated evaluation (K)
Run the scorer `K` times (default `K=3`, `RECEIPTS_SCORE_PASSES` override) and average `score01`. Cheap variance reduction from the paper. `K=1` when a fast/cheap mode is requested. Passes run concurrently.

### 3.3 Scaling dimension — criteria decomposition (`criteria.py` + `criteria/*.md`)
For code claims, score three sub-criteria (paper's coding triad) and combine:
- `specification.md` — does the change match what was asked / claimed?
- `output_match.md` — does observed output/behavior match the claim?
- `error_signal.md` — are there error signals (failing tests, tracebacks, non-zero exits) contradicting the claim?

Each criterion is an independent prompt fragment (Markdown, shipped in-package). `criteria.py` selects the criteria set by `ClaimType`, runs each through the scorer, and combines (min-biased for error-signal so one strong contradiction can sink an over-optimistic claim). Non-code / generic claims use a single holistic criterion. Criteria files are data, versioned in-repo, referenced by name.

### 3.4 Graceful fallback (zero-config guarantee)
Logprobs require Vertex Gemini or an OpenAI-compatible endpoint exposing `top_logprobs ≤ 20`; **Anthropic exposes no logprobs**. Resolution order per session:
1. Checker supports logprobs → **LOGPROB** method (calibrated expectation).
2. Checker has no logprobs (e.g. Anthropic-only key) → **SAMPLED** method: request the scoring letter at low temperature over a few samples and average their numeric values. Lower fidelity, clearly labeled on the card (`method: sampled`), still continuous, still zero-config.
3. No checker key at all → deterministic-evidence-only receipt (v1 behavior for hard claims; fuzzy claims marked `skipped`). Never crash.

The score granularity `G` (20) is a constant, matching the released code (not user-tunable this cycle).

### 3.5 Verdict banding (card compatibility)
Continuous score maps to glyphs for the card: `≥ 85` ✅ verified · `55–84` ⚠️ weak/unverified · `< 55` ❌ refuted. Deterministic refutation forces ❌. Bands are display-only; the stored value is the continuous score.

### 3.6 Upgraded receipt card (`render.py`)
Card shows the **Verified-Done Score: NN/100** in the header, per-claim continuous scores with glyph + one-line reason, per-criterion mini-bars for code claims, and a `method` tag (`calibrated` vs `sampled`). Keeps the existing Rich box aesthetic.

### 3.7 Sharing & distribution (gotcha-forward)
- `receipts share [session]` → renders the **"caught red-handed" receipt** to a **PNG**, optimized as a social artifact: the agent's verbatim confident claim above the ❌ REFUTED verdict + score + the specific contradicting evidence. Prioritizes refuted fuzzy claims (the shareable ones) over deterministic ones. Self-contained; nothing leaves the machine unless the user posts it. This is the organic viral unit.
- `receipts badge` → **SVG** badge (shields-style) for READMEs — but framed as an *honesty* signal (e.g. "verified-done", not a vanity number).
- `receipts publish` → **opt-in**, sends **aggregate numbers only** — keyed by **audited-agent identity** (e.g. `claude-code`, `cursor`, `copilot`) — false-done rate, verification rate, checker model name, claim/session counts. **Never** sends transcript text, claim text, code, paths, or prompts. Explicit first-run consent; off by default. Feeds the Honesty Index (§3.8).

### 3.8 The AI Agent Honesty Index (thin, static, agent-vs-agent)
Cloudflare Worker + KV store + a static HTML page ranking **coding agents by how often they falsely claim "done"** — never users. Worker accepts the aggregate payload (validated, size-capped, no free-text pass-through), aggregates per audited-agent identity across all opt-in contributors, and serves a read-only ranked "honesty leaderboard." No accounts; an anonymous local contributor id used only for dedup/rate-limiting, never displayed or ranked. This is the only server-side component and the data moat — an ongoing, quote-tweetable industry stat only Receipts can produce.

---

## 4. Phase 2 — Self-Healing (headline follow-on)

Triggered on a **refuted** claim (score below the refuted band or deterministic contradiction).

1. **Generate** N candidate fixes (default N=3) from the refuting evidence + relevant transcript slice, using the checker-side model (still maker ≠ original agent for the *evaluation*; generation model configurable).
2. **Score** each candidate through the decomposed scorer (§3.3).
3. **Select** best-of-N with **PPT** (`pivot_tournament.py`, NEW) — Bradley-Terry pivots `1/(1+exp(-(ra−rb)))`, `DEFAULT_PIVOTS=2`, O(Nk); ported from `pivot_tournament.py` in the reference repo.
4. **Present** the winning candidate as a unified diff in the receipt.
5. **Apply** only on **opt-in** (`RECEIPTS_AUTOFIX=1` or interactive confirm). Never auto-writes by default.

`retry.py` (NEW) orchestrates generate → score → PPT → present → (opt-in) apply.

---

## 5. Data Model Changes (`models.py`)
- Add `ScoreMethod` enum (`LOGPROB`, `SAMPLED`).
- Add `ClaimScore` and `SessionScore` (or dataclasses in `scorer.py` importing existing enums — keep Pydantic where persisted).
- Persist continuous `score` (0–100), `method`, and `passes` alongside existing verdict fields in the local store so `stats` and `badge` can aggregate numerically. Backward-compatible: old records without `score` are treated as verdict-only.
- Extend `EvidenceSource` only if needed for candidate-fix provenance in Phase 2.

---

## 6. Configuration
- `RECEIPTS_SCORE_PASSES` (default 3) — K repeated passes.
- `RECEIPTS_CHECKER_MODEL` / existing provider auto-detection — checker selection; enforce maker ≠ checker.
- `RECEIPTS_AUTOFIX` (default 0) — Phase 2 opt-in apply.
- `RECEIPTS_PUBLISH` / first-run consent — opt-in Honesty Index contribution.
- `RECEIPTS_BLOCK` (existing) — block exit on unverified.
- Fast mode (`K=1`, single criterion) for latency-sensitive users.

---

## 7. Privacy & Safety (hard constraints)
- **Local-first.** All scoring and receipts are computed and stored locally. Nothing is transmitted unless the user runs an opt-in command.
- **`publish` is aggregate-only.** Payload schema is a fixed set of numbers + the audited-agent identity + the checker model name + an anonymous contributor id. The Worker rejects any field outside the schema; there is no free-text field. Transcript, claim text, code, file paths, and prompts are **never** sent. The audited-agent identity is a known enum (`claude-code`, `cursor`, …), not free text.
- **Consent is explicit and off by default.** First `publish` requires an interactive opt-in; a config flag records it.
- **maker ≠ checker** is enforced in code; the audited agent's provider cannot be the checker.
- **No PyPI publish, no push to `main`** this cycle.

---

## 8. Testing Strategy
- **Scorer math:** unit-test Eq 3.1 expectation on synthetic logprob distributions (known inputs → known score), including renormalization and single-candidate edge cases.
- **Fallback ladder:** simulate logprob-capable, sampled-only, and no-key configs → assert method selection and no-crash behavior.
- **Criteria decomposition:** assert criterion selection per `ClaimType` and min-biased error-signal combination (a strong error signal sinks an optimistic claim).
- **Determinism clamp:** `pytest` exit ≠ 0 forces ❌ regardless of checker output.
- **PPT:** ranking correctness on synthetic rating vectors; O(Nk) pivot count.
- **Privacy:** unit-test the `publish` payload builder emits only whitelisted numeric/id fields (golden-schema test); Worker rejects extra fields.
- **End-to-end (per global guidance):** reproduce a real Claude Code session end-to-end — a genuinely refuted "all tests pass" claim — and assert the receipt score, card, and (Phase 2) proposed diff, before trusting unit tests alone.

---

## 9. Build Plan (agent allocation)
Token-heavy implementation is delegated to **Sonnet** agents on isolated branches/worktrees; **Fable 5** is reserved only for the go/no-go calibration review and for un-sticking a stuck agent (high-stakes, expensive).

- **Agent A (Sonnet):** `scorer.py` + Eq 3.1 math + `ScoreMethod` model changes + fallback ladder + scorer unit tests.
- **Agent B (Sonnet):** `criteria.py` + `criteria/*.md` triad + combination logic + tests.
- **Agent C (Sonnet):** `render.py` card upgrade + `share` (PNG) + `badge` (SVG) CLI.
- **Agent D (Sonnet):** Honesty Index Worker + KV + static agent-vs-agent page + `publish` opt-in client (aggregate, agent-keyed) + privacy/schema tests.
- **Agent E (Sonnet, Phase 2):** `pivot_tournament.py` + `retry.py` + self-healing flow + tests.
- **Fable 5 checkpoint:** review calibrated-score correctness and maker≠checker/privacy enforcement before the PR; intervene on any stuck agent.

Integration: each agent branches from the feature branch; results merged and run through the end-to-end test before opening the PR against `harish-nair-ai/agentic-receipts` (PR, not `main`).

---

## 10. Risks & Mitigations
- **Logprob availability varies by provider/endpoint.** → Fallback ladder (§3.4); label method on card; never require setup.
- **Calibration drift across checker models.** → Fixed 20-level scale + renormalization; Fable 5 go/no-go review on real sessions.
- **Latency from K× and criteria×.** → Concurrency; fast mode (K=1, single criterion).
- **Privacy regression in `publish`.** → Golden-schema test + Worker-side field rejection; off by default.
- **Scope creep back toward the paper's RL/proxy.** → Explicitly dropped in Non-Goals; do not re-introduce.
