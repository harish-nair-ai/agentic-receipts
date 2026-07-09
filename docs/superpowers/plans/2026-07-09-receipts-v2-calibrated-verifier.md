# Receipts v2 — Calibrated, Uncertainty-Aware Verifier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Receipts' single discrete LLM judge with a calibrated, uncertainty-aware fact-checker that gives every agent completion claim a three-way verdict (SUPPORTED / REFUTED / NOT-ENOUGH-INFO), a calibrated 0–100 score, and a specific critique — then adds opt-in self-healing for refuted claims.

**Architecture:** A per-claim verifier pipeline: deterministic entailment fast-path → independent calibrated scorer (logprob expectation over an A–T 20-level scale, paper Eq 3.1) run across a weak-verifier ensemble of criteria → uncertainty control (variance-gated adaptive K + abstention). Graceful fallback ladder (logprob → sampled → evidence-only) preserves zero-config. Phase 2 ranks candidate fixes with a Probabilistic Pivot Tournament.

**Tech Stack:** Python ≥3.10, Pydantic v2, httpx, Rich, Click, hatchling. pytest + pytest-asyncio + ruff for dev. LLM checker via OpenAI-compatible / Gemini (logprobs) or Anthropic (sampled fallback).

## Global Constraints

- Python floor: `requires-python = ">=3.10"` — use the `StrEnum` backport pattern already in `models.py`/`config.py`; do not import `enum.StrEnum` unconditionally.
- Ships on branch `feat/receipts-v2-calibrated-verifier` → **PR only, never push to `main`**.
- **No PyPI publish** this cycle. Do not run `hatch publish` / `twine`.
- **maker ≠ checker** enforced in code: the checker model's provider must differ from the audited agent's family; when it cannot, mark `checker_independent = False` and surface it — never silently self-audit.
- **Local-first:** nothing is transmitted anywhere except the user's configured checker API. No telemetry, no `receipts share`, no `receipts badge`, no Honesty Index this cycle.
- Score scale is fixed: `G = 20` levels, letters `A`…`T`, `value(letter) = (ord(letter) − ord('A')) / 19`. Not user-tunable.
- Score→label bands (when not abstaining / not fast-pathed): `score ≥ 85` → SUPPORTED · `score < 45` → REFUTED · otherwise → NOT_ENOUGH_INFO.
- Abstention: `confidence < RECEIPTS_MIN_CONFIDENCE` (default 0.6) → NOT_ENOUGH_INFO regardless of point score.
- Config env vars (exact names): `RECEIPTS_SCORE_PASSES` (default 3), `RECEIPTS_MIN_CONFIDENCE` (default 0.6), `RECEIPTS_CHECKER_MODEL`, `RECEIPTS_AUTOFIX` (default 0), plus existing `RECEIPTS_BLOCK`, `RECEIPTS_PROVIDER`, `RECEIPTS_MODEL`, `RECEIPTS_DIR`, `RECEIPTS_TIMEOUT`.
- Never crash the agent: all new pipeline code is wrapped by the existing `process_transcript` try/except; degrade to a lower fallback rung instead of raising.
- Follow existing style: `from __future__ import annotations`, Pydantic models for persisted data, plain dataclasses for in-flight compute, ruff rules `E,F,I,N,W,UP,B,SIM`, line length 100.

---

## File Structure

**New files:**
- `src/receipts/scorer.py` — calibrated scoring: Eq 3.1 math (pure functions) + LLM scoring/critique calls + `ScorePass`.
- `src/receipts/criteria.py` — criterion selection by `ClaimType`, prompt assembly, weak-verifier combination.
- `src/receipts/criteria/specification.md` — "does the change match what was asked/claimed?" prompt fragment.
- `src/receipts/criteria/output_match.md` — "does observed output/behavior match the claim?" fragment.
- `src/receipts/criteria/error_signal.md` — "are there error signals contradicting the claim?" fragment.
- `src/receipts/criteria/holistic.md` — single holistic fragment for non-code/generic claims.
- `src/receipts/verifier.py` — fast-path, adaptive-K loop, confidence aggregation, abstention, three-way labeling; `ClaimVerdict`, `SessionReport`, `verify_claim`, `verify_session`.
- `src/receipts/pivot_tournament.py` — Phase 2: Probabilistic Pivot Tournament best-of-N.
- `src/receipts/retry.py` — Phase 2: generate → score → PPT → present → opt-in apply.
- `tests/test_scorer_math.py`, `tests/test_scorer_calls.py`, `tests/test_criteria.py`, `tests/test_verifier.py`, `tests/test_models_v2.py`, `tests/test_config_v2.py`, `tests/test_render_v2.py`, `tests/test_stats_v2.py`, `tests/test_pivot_tournament.py`, `tests/test_retry.py`, `tests/test_end_to_end.py`.
- `tests/fixtures/session_refuted_and_ambiguous.jsonl` — E2E transcript fixture.

**Modified files:**
- `src/receipts/models.py` — add `FactLabel`, `ScoreMethod`; extend `VerifiedClaim` + `Receipt` (back-compat, all new fields optional).
- `src/receipts/config.py` — add `score_passes`, `min_confidence`, `checker_model`, `autofix`; `supports_logprobs` property; `check_independence`.
- `src/receipts/hook.py` — route claims through `verify_session`; map `ClaimVerdict` → `VerifiedClaim`.
- `src/receipts/render.py` — three-way card upgrade.
- `src/receipts/stats.py` — numeric aggregates over the new fields.
- `README.md` — replace unsourced "~47%" with cited "22.58%".
- `pyproject.toml` — ship `criteria/*.md` as package data.

---

## Task 1: Data model — `FactLabel`, `ScoreMethod`, extended `VerifiedClaim`/`Receipt`

**Files:**
- Modify: `src/receipts/models.py`
- Test: `tests/test_models_v2.py`

**Interfaces:**
- Consumes: existing `Claim`, `Verdict`, `Evidence`, `StrEnum` backport, `VerifiedClaim`, `Receipt`.
- Produces:
  - `FactLabel(StrEnum)`: `SUPPORTED="supported"`, `REFUTED="refuted"`, `NOT_ENOUGH_INFO="not_enough_info"`.
  - `ScoreMethod(StrEnum)`: `LOGPROB="logprob"`, `SAMPLED="sampled"`, `DETERMINISTIC="deterministic"`, `NONE="none"`. (Superset of the spec's LOGPROB|SAMPLED: fast-path decisions and evidence-only receipts need honest method tags too.)
  - `verdict_for_label(label: FactLabel) -> Verdict` mapping SUPPORTED→VERIFIED, REFUTED→REFUTED, NOT_ENOUGH_INFO→UNVERIFIED.
  - `VerifiedClaim` new optional fields: `label: FactLabel | None`, `score: float | None`, `confidence: float | None`, `per_criterion: dict[str, float]`, `critique: str`, `method: ScoreMethod | None`, `passes: int`.
  - `Receipt` new props: `supported_count`, `nei_count`, `verified_done_score` (mean of per-claim `score`, 0–100), `checker_independent: bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_v2.py
from receipts.models import (
    Claim, ClaimType, Evidence, EvidenceSource, FactLabel, Receipt, ScoreMethod,
    VerifiedClaim, Verdict, verdict_for_label,
)


def test_factlabel_and_scoremethod_values():
    assert FactLabel.SUPPORTED == "supported"
    assert FactLabel.REFUTED == "refuted"
    assert FactLabel.NOT_ENOUGH_INFO == "not_enough_info"
    assert ScoreMethod.LOGPROB == "logprob"
    assert ScoreMethod.SAMPLED == "sampled"


def test_verdict_for_label_maps_backcompat():
    assert verdict_for_label(FactLabel.SUPPORTED) == Verdict.VERIFIED
    assert verdict_for_label(FactLabel.REFUTED) == Verdict.REFUTED
    assert verdict_for_label(FactLabel.NOT_ENOUGH_INFO) == Verdict.UNVERIFIED


def test_verifiedclaim_new_fields_optional_and_backcompat():
    # Old-style construction (no new fields) still works.
    vc = VerifiedClaim(claim=Claim(text="x"), verdict=Verdict.VERIFIED)
    assert vc.label is None and vc.score is None and vc.per_criterion == {}
    # New-style construction populates the calibrated fields.
    vc2 = VerifiedClaim(
        claim=Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS),
        verdict=Verdict.REFUTED,
        label=FactLabel.REFUTED,
        score=8.0,
        confidence=0.92,
        per_criterion={"error_signal": 5.0},
        critique="pytest exited 1 with 2 failures",
        method=ScoreMethod.LOGPROB,
        passes=1,
    )
    assert vc2.label == FactLabel.REFUTED
    assert vc2.per_criterion["error_signal"] == 5.0


def test_receipt_aggregates_over_calibrated_fields():
    claims = [
        VerifiedClaim(claim=Claim(text="a"), verdict=Verdict.VERIFIED,
                      label=FactLabel.SUPPORTED, score=95.0),
        VerifiedClaim(claim=Claim(text="b"), verdict=Verdict.REFUTED,
                      label=FactLabel.REFUTED, score=5.0),
        VerifiedClaim(claim=Claim(text="c"), verdict=Verdict.UNVERIFIED,
                      label=FactLabel.NOT_ENOUGH_INFO, score=60.0),
    ]
    r = Receipt(session_id="s1", claims=claims)
    assert r.supported_count == 1
    assert r.refuted_count == 1
    assert r.nei_count == 1
    # verified_done_score = mean of scores = (95+5+60)/3 ≈ 53.33
    assert round(r.verified_done_score, 1) == 53.3
    assert r.checker_independent is True  # default


def test_receipt_verified_done_score_empty_is_100():
    r = Receipt(session_id="s2", claims=[])
    assert r.verified_done_score == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models_v2.py -v`
Expected: FAIL — `ImportError: cannot import name 'FactLabel'`.

- [ ] **Step 3: Write minimal implementation**

In `src/receipts/models.py`, after the `Verdict` class, add:

```python
class FactLabel(StrEnum):
    """FEVER-style three-way verdict for a completion claim."""

    SUPPORTED = "supported"              # evidence supports the claim
    REFUTED = "refuted"                  # evidence contradicts the claim
    NOT_ENOUGH_INFO = "not_enough_info"  # honest abstention — insufficient evidence


class ScoreMethod(StrEnum):
    """How a claim's calibrated score was produced."""

    LOGPROB = "logprob"              # probability-weighted expectation over top_logprobs
    SAMPLED = "sampled"              # averaged sampled scoring letters (no logprobs available)
    DETERMINISTIC = "deterministic"  # decided by the hard-evidence fast-path, no LLM
    NONE = "none"                    # no checker available — evidence-only / abstained


def verdict_for_label(label: FactLabel) -> Verdict:
    """Map the three-way FactLabel back to the legacy Verdict for back-compat storage."""
    return {
        FactLabel.SUPPORTED: Verdict.VERIFIED,
        FactLabel.REFUTED: Verdict.REFUTED,
        FactLabel.NOT_ENOUGH_INFO: Verdict.UNVERIFIED,
    }[label]
```

Extend `VerifiedClaim` (add fields after `reasoning`):

```python
    # --- v2 calibrated fields (all optional for back-compat with v1 records) ---
    label: FactLabel | None = Field(
        default=None, description="Three-way calibrated verdict (v2)"
    )
    score: float | None = Field(
        default=None, description="Calibrated 0-100 score for this claim (v2)"
    )
    confidence: float | None = Field(
        default=None, description="0-1 confidence from cross-pass agreement + logprob dispersion"
    )
    per_criterion: dict[str, float] = Field(
        default_factory=dict, description="Per-criterion 0-100 scores (weak-verifier ensemble)"
    )
    critique: str = Field(default="", description="Specific NL critique / evidence pointer")
    method: ScoreMethod | None = Field(default=None, description="How the score was produced")
    passes: int = Field(default=0, description="Number of scorer passes actually run (adaptive K)")
```

Extend `Receipt`: add `checker_independent` field and the new aggregate properties.

```python
    checker_independent: bool = Field(
        default=True, description="Whether checker model differed from the audited agent (maker≠checker)"
    )
```

Add these properties alongside the existing ones on `Receipt`:

```python
    @property
    def supported_count(self) -> int:
        """Number of claims labeled SUPPORTED (v2)."""
        return sum(1 for c in self.claims if c.label == FactLabel.SUPPORTED)

    @property
    def nei_count(self) -> int:
        """Number of claims labeled NOT_ENOUGH_INFO (v2)."""
        return sum(1 for c in self.claims if c.label == FactLabel.NOT_ENOUGH_INFO)

    @property
    def verified_done_score(self) -> float:
        """Aggregate calibrated Verified-Done Score (0-100): mean of per-claim scores.

        Falls back to the legacy fraction*100 when no calibrated scores are present.
        Empty session → 100.0 (nothing claimed, nothing to doubt).
        """
        scored = [c.score for c in self.claims if c.score is not None]
        if scored:
            return sum(scored) / len(scored)
        if self.total_claims == 0:
            return 100.0
        return self.score * 100.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models_v2.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/receipts/models.py tests/test_models_v2.py
git commit -m "feat(models): add FactLabel, ScoreMethod, calibrated VerifiedClaim/Receipt fields

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Config — new env vars, logprob capability, maker≠checker

**Files:**
- Modify: `src/receipts/config.py`
- Test: `tests/test_config_v2.py`

**Interfaces:**
- Consumes: existing `Config`, `JudgeProvider`, `from_env`.
- Produces:
  - `Config.score_passes: int` (env `RECEIPTS_SCORE_PASSES`, default 3, min 1).
  - `Config.min_confidence: float` (env `RECEIPTS_MIN_CONFIDENCE`, default 0.6, clamped 0..1).
  - `Config.checker_model: str` (env `RECEIPTS_CHECKER_MODEL`, else falls back to `model`).
  - `Config.autofix: bool` (env `RECEIPTS_AUTOFIX`, default False).
  - `Config.supports_logprobs -> bool` property: True for GEMINI/OPENAI, False for ANTHROPIC.
  - `AGENT_PROVIDER_FAMILY: dict[str, JudgeProvider]` and `check_independence(config, agent) -> bool`: False when checker provider == the audited agent's provider family.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_v2.py
import pytest

from receipts.config import Config, JudgeProvider, check_independence


def _cfg(provider, model="m", **kw):
    return Config(provider=provider, model=model, api_key="k", **kw)


def test_new_env_vars_parsed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-x")
    monkeypatch.setenv("RECEIPTS_SCORE_PASSES", "5")
    monkeypatch.setenv("RECEIPTS_MIN_CONFIDENCE", "0.75")
    monkeypatch.setenv("RECEIPTS_CHECKER_MODEL", "gemini-custom")
    monkeypatch.setenv("RECEIPTS_AUTOFIX", "1")
    cfg = Config.from_env()
    assert cfg.score_passes == 5
    assert cfg.min_confidence == 0.75
    assert cfg.checker_model == "gemini-custom"
    assert cfg.autofix is True


def test_defaults_when_unset(monkeypatch):
    for v in ("RECEIPTS_SCORE_PASSES", "RECEIPTS_MIN_CONFIDENCE",
              "RECEIPTS_CHECKER_MODEL", "RECEIPTS_AUTOFIX"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    cfg = Config.from_env()
    assert cfg.score_passes == 3
    assert cfg.min_confidence == 0.6
    assert cfg.autofix is False
    assert cfg.checker_model == cfg.model  # falls back to model


def test_score_passes_floor_and_confidence_clamp(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("RECEIPTS_SCORE_PASSES", "0")
    monkeypatch.setenv("RECEIPTS_MIN_CONFIDENCE", "5")
    cfg = Config.from_env()
    assert cfg.score_passes == 1       # floored to 1
    assert cfg.min_confidence == 1.0   # clamped to [0,1]


def test_supports_logprobs_by_provider():
    assert _cfg(JudgeProvider.OPENAI).supports_logprobs is True
    assert _cfg(JudgeProvider.GEMINI).supports_logprobs is True
    assert _cfg(JudgeProvider.ANTHROPIC).supports_logprobs is False


def test_check_independence():
    # Auditing Claude Code (anthropic family) with an Anthropic checker → NOT independent.
    assert check_independence(_cfg(JudgeProvider.ANTHROPIC), "claude-code") is False
    # Gemini checker auditing Claude Code → independent.
    assert check_independence(_cfg(JudgeProvider.GEMINI), "claude-code") is True
    # Unknown agent → assume independent (can't prove overlap).
    assert check_independence(_cfg(JudgeProvider.ANTHROPIC), "some-other-agent") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_v2.py -v`
Expected: FAIL — `ImportError: cannot import name 'check_independence'`.

- [ ] **Step 3: Write minimal implementation**

In `src/receipts/config.py`, add fields to `Config` (after `timeout`):

```python
    score_passes: int = Field(default=3, description="K_max for adaptive scorer escalation")
    min_confidence: float = Field(default=0.6, description="Abstention threshold (0-1)")
    checker_model_override: str = Field(
        default="", description="Explicit checker model (RECEIPTS_CHECKER_MODEL)"
    )
    autofix: bool = Field(default=False, description="Phase 2 opt-in: auto-apply winning fix")

    @property
    def checker_model(self) -> str:
        """The model used as the independent checker (falls back to `model`)."""
        return self.checker_model_override or self.model

    @property
    def supports_logprobs(self) -> bool:
        """Whether the configured provider can return top_logprobs for calibrated scoring."""
        return self.provider in (JudgeProvider.GEMINI, JudgeProvider.OPENAI)
```

In `from_env`, after `timeout = ...`, parse the new vars and pass them into `cls(...)`:

```python
        score_passes = max(1, int(os.environ.get("RECEIPTS_SCORE_PASSES", "3")))
        min_confidence = min(1.0, max(0.0, float(os.environ.get("RECEIPTS_MIN_CONFIDENCE", "0.6"))))
        checker_model_override = os.environ.get("RECEIPTS_CHECKER_MODEL", "").strip()
        autofix = os.environ.get("RECEIPTS_AUTOFIX", "0").strip() == "1"
```

And extend the final `return cls(...)` with:

```python
            score_passes=score_passes,
            min_confidence=min_confidence,
            checker_model_override=checker_model_override,
            autofix=autofix,
```

At module scope (after `API_URLS`), add the agent→family map and independence check:

```python
# Which provider family an audited agent belongs to (for maker≠checker enforcement).
AGENT_PROVIDER_FAMILY: dict[str, JudgeProvider] = {
    "claude-code": JudgeProvider.ANTHROPIC,
    "claude": JudgeProvider.ANTHROPIC,
    "codex": JudgeProvider.OPENAI,
    "cursor": JudgeProvider.OPENAI,  # unknown/mixed; treated as OpenAI-family by default
}


def check_independence(config: "Config", agent: str) -> bool:
    """True when the checker's provider differs from the audited agent's provider family.

    Unknown agents are assumed independent — we cannot prove model overlap.
    """
    family = AGENT_PROVIDER_FAMILY.get(agent.lower())
    if family is None:
        return True
    return config.provider != family
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_v2.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/receipts/config.py tests/test_config_v2.py
git commit -m "feat(config): adaptive-K, abstention, checker-model, autofix, maker≠checker

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `scorer.py` — Eq 3.1 calibrated scoring math (pure functions)

**Files:**
- Create: `src/receipts/scorer.py`
- Test: `tests/test_scorer_math.py`

**Interfaces:**
- Produces:
  - `G = 20`, `SCORE_LETTERS = "ABCDEFGHIJKLMNOPQRST"`.
  - `letter_value(letter: str) -> float` — `(ord−ord('A'))/19`, clamped to valid letters.
  - `expected_score01(logprobs: dict[str, float]) -> float | None` — Eq 3.1 renormalized expectation over valid letters; `None` if no valid candidate.
  - `peakiness(logprobs: dict[str, float]) -> float` — renormalized probability mass on the modal valid letter (0..1).
  - `sampled_score01(letters: list[str]) -> float | None` — mean of `letter_value` over valid sampled letters.
  - `sampled_peakiness(letters: list[str]) -> float` — `1 - min(1, 2*pstdev(values))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scorer_math.py
import math

import pytest

from receipts.scorer import (
    G, SCORE_LETTERS, expected_score01, letter_value, peakiness,
    sampled_peakiness, sampled_score01,
)


def test_scale_constants():
    assert G == 20
    assert SCORE_LETTERS == "ABCDEFGHIJKLMNOPQRST"
    assert len(SCORE_LETTERS) == 20


def test_letter_value_endpoints_and_mid():
    assert letter_value("A") == 0.0
    assert letter_value("T") == 1.0
    assert letter_value("K") == pytest.approx((ord("K") - ord("A")) / 19)


def test_expected_score01_single_candidate():
    # Only "T" present → expectation is exactly value(T) = 1.0 after renormalization.
    assert expected_score01({"T": math.log(0.5)}) == pytest.approx(1.0)


def test_expected_score01_two_candidates_renormalized():
    # A (0.0) with p=0.25, T (1.0) with p=0.75 → renormalized expectation = 0.75.
    lp = {"A": math.log(0.25), "T": math.log(0.75)}
    assert expected_score01(lp) == pytest.approx(0.75)


def test_expected_score01_ignores_invalid_tokens():
    # "Z", " ", "hello" are not valid A..T letters and are dropped.
    lp = {"Z": math.log(0.5), "T": math.log(0.25), "A": math.log(0.25)}
    # Renormalize over A(0.25)->0.0 and T(0.25)->1.0 → 0.5
    assert expected_score01(lp) == pytest.approx(0.5)


def test_expected_score01_none_when_no_valid_candidate():
    assert expected_score01({"Z": math.log(0.9), "5": math.log(0.1)}) is None
    assert expected_score01({}) is None


def test_peakiness_concentrated_vs_flat():
    concentrated = peakiness({"T": math.log(0.95), "S": math.log(0.05)})
    flat = peakiness({"A": math.log(0.5), "T": math.log(0.5)})
    assert concentrated > 0.9
    assert flat == pytest.approx(0.5)


def test_sampled_score01_and_peakiness():
    assert sampled_score01(["T", "T", "T"]) == pytest.approx(1.0)
    assert sampled_score01([]) is None
    # All-same letters → maximum peakiness.
    assert sampled_peakiness(["K", "K", "K"]) == pytest.approx(1.0)
    # Spread letters → lower peakiness.
    assert sampled_peakiness(["A", "T"]) < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scorer_math.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'receipts.scorer'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/receipts/scorer.py
"""Calibrated scoring — logprob expectation over an A-T 20-level ordinal scale.

Implements the base paper's Eq 3.1 (mirrors fine_grained_reward.py::extract_score):

    score01 = Σ_i value(tok_i) · p_i / Σ_i p_i      # renormalized over valid candidates
    value(letter) = (ord(letter) − ord('A')) / 19   # A→0.0 … T→1.0 ; p_i = exp(logprob_i)
"""

from __future__ import annotations

import math
from statistics import pstdev

G = 20
SCORE_LETTERS = "ABCDEFGHIJKLMNOPQRST"  # 20 ordinal levels, A = fully refuted, T = fully supported
_VALID = set(SCORE_LETTERS)


def _clean_letter(token: str) -> str | None:
    """Normalize a returned token to a single valid scoring letter, or None."""
    t = token.strip().upper()
    return t if len(t) == 1 and t in _VALID else None


def letter_value(letter: str) -> float:
    """Map a scoring letter to its ordinal value in [0, 1]."""
    t = _clean_letter(letter)
    if t is None:
        raise ValueError(f"not a valid scoring letter: {letter!r}")
    return (ord(t) - ord("A")) / (G - 1)


def _valid_probs(logprobs: dict[str, float]) -> dict[str, float]:
    """Collapse raw {token: logprob} to {letter: probability} over valid letters."""
    probs: dict[str, float] = {}
    for token, lp in logprobs.items():
        letter = _clean_letter(token)
        if letter is not None:
            probs[letter] = probs.get(letter, 0.0) + math.exp(lp)
    return probs


def expected_score01(logprobs: dict[str, float]) -> float | None:
    """Probability-weighted expectation (Eq 3.1), renormalized over valid candidates."""
    probs = _valid_probs(logprobs)
    denom = sum(probs.values())
    if denom == 0.0:
        return None
    return sum(letter_value(letter) * p for letter, p in probs.items()) / denom


def peakiness(logprobs: dict[str, float]) -> float:
    """Renormalized probability mass on the modal valid letter (0..1). 0.0 if none valid."""
    probs = _valid_probs(logprobs)
    denom = sum(probs.values())
    if denom == 0.0:
        return 0.0
    return max(probs.values()) / denom


def sampled_score01(letters: list[str]) -> float | None:
    """Mean ordinal value over valid sampled letters, or None if none valid."""
    values = [letter_value(c) for c in letters if _clean_letter(c) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def sampled_peakiness(letters: list[str]) -> float:
    """Concentration of sampled letters: 1 - min(1, 2·pstdev(values)). 1.0 for <2 samples."""
    values = [letter_value(c) for c in letters if _clean_letter(c) is not None]
    if len(values) < 2:
        return 1.0
    return 1.0 - min(1.0, 2.0 * pstdev(values))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scorer_math.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/receipts/scorer.py tests/test_scorer_math.py
git commit -m "feat(scorer): Eq 3.1 calibrated logprob expectation math

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `scorer.py` — LLM scoring & critique calls (`ScorePass`, `score_pass`, `generate_critique`)

**Files:**
- Modify: `src/receipts/scorer.py`
- Test: `tests/test_scorer_calls.py`

**Interfaces:**
- Consumes: `expected_score01`, `peakiness`, `sampled_score01`, `sampled_peakiness`, `SCORE_LETTERS`; `Config`, `JudgeProvider`, `ScoreMethod`.
- Produces:
  - `@dataclass ScorePass{ score01: float; peakiness: float; method: ScoreMethod; letter: str }`.
  - `score_pass(prompt: str, config: Config) -> ScorePass | None` — one calibrated scoring call; returns `None` on hard failure (caller drops to a lower rung).
  - `generate_critique(claim_text, evidence_text, transcript_context, label, config) -> str` — one short NL critique call; `""` on failure.
  - Internal `_score_logprob_openai`, `_score_logprob_gemini`, `_score_sampled`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scorer_calls.py
import math
from unittest.mock import patch

from receipts.config import Config, JudgeProvider
from receipts.models import ScoreMethod
from receipts.scorer import ScorePass, generate_critique, score_pass


def _cfg(provider, model="m"):
    return Config(provider=provider, model=model, api_key="k")


def test_score_pass_logprob_openai_parses_top_logprobs():
    fake = ScorePass(score01=0.75, peakiness=0.9, method=ScoreMethod.LOGPROB, letter="T")
    with patch("receipts.scorer._score_logprob_openai", return_value=fake) as m:
        out = score_pass("prompt", _cfg(JudgeProvider.OPENAI))
    assert out is fake
    m.assert_called_once()


def test_score_pass_uses_sampled_for_anthropic():
    fake = ScorePass(score01=0.5, peakiness=1.0, method=ScoreMethod.SAMPLED, letter="K")
    with patch("receipts.scorer._score_sampled", return_value=fake) as m:
        out = score_pass("prompt", _cfg(JudgeProvider.ANTHROPIC))
    assert out.method == ScoreMethod.SAMPLED
    m.assert_called_once()


def test_score_pass_returns_none_on_exception():
    with patch("receipts.scorer._score_logprob_openai", side_effect=RuntimeError("boom")):
        assert score_pass("prompt", _cfg(JudgeProvider.OPENAI)) is None


def test_generate_critique_returns_empty_on_exception():
    with patch("receipts.scorer._critique_call", side_effect=RuntimeError("net")):
        assert generate_critique("c", "e", "ctx", "refuted", _cfg(JudgeProvider.OPENAI)) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scorer_calls.py -v`
Expected: FAIL — `ImportError: cannot import name 'ScorePass'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/receipts/scorer.py`:

```python
from dataclasses import dataclass

import httpx

from receipts.config import Config, JudgeProvider
from receipts.models import ScoreMethod

_SAMPLED_N = 5  # sampled-fallback: number of scoring letters to draw per pass

_SCORE_INSTRUCTION = (
    "You are an independent auditor scoring whether an AI coding agent's claim is "
    "supported by the transcript evidence. Respond with EXACTLY ONE letter on the "
    f"scale {SCORE_LETTERS[0]}-{SCORE_LETTERS[-1]}, where {SCORE_LETTERS[0]} means the "
    "claim is fully REFUTED by the evidence and "
    f"{SCORE_LETTERS[-1]} means it is fully SUPPORTED. Output only the single letter."
)


@dataclass
class ScorePass:
    """One calibrated scoring pass over a single criterion."""

    score01: float          # 0..1 calibrated score
    peakiness: float        # 0..1 distribution concentration (confidence proxy)
    method: ScoreMethod     # LOGPROB or SAMPLED
    letter: str             # modal / representative scoring letter


def score_pass(prompt: str, config: Config) -> ScorePass | None:
    """Run one calibrated scoring call. Returns None on failure (caller degrades)."""
    full = f"{_SCORE_INSTRUCTION}\n\n{prompt}\n\nScore (single letter only):"
    try:
        if config.supports_logprobs and config.provider == JudgeProvider.OPENAI:
            return _score_logprob_openai(full, config)
        if config.supports_logprobs and config.provider == JudgeProvider.GEMINI:
            return _score_logprob_gemini(full, config)
        return _score_sampled(full, config)
    except Exception:
        return None


def _score_logprob_openai(prompt: str, config: Config) -> ScorePass | None:
    url = f"{config.api_url}/chat/completions"
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": config.checker_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
    }
    with httpx.Client(timeout=config.timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["logprobs"]["content"][0]
    logprobs = {alt["token"]: alt["logprob"] for alt in content["top_logprobs"]}
    return _pass_from_logprobs(logprobs, content.get("token", ""))


def _score_logprob_gemini(prompt: str, config: Config) -> ScorePass | None:
    url = f"{config.api_url}/models/{config.checker_model}:generateContent?key={config.api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1,
            "responseLogprobs": True,
            "logprobs": 20,
        },
    }
    with httpx.Client(timeout=config.timeout) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    # Gemini returns logprobsResult.topCandidates[0].candidates: [{token, logProbability}]
    top = data["candidates"][0]["logprobsResult"]["topCandidates"][0]["candidates"]
    logprobs = {c["token"]: c["logProbability"] for c in top}
    chosen = data["candidates"][0]["logprobsResult"]["chosenCandidates"][0]["token"]
    return _pass_from_logprobs(logprobs, chosen)


def _pass_from_logprobs(logprobs: dict[str, float], chosen: str) -> ScorePass | None:
    score01 = expected_score01(logprobs)
    if score01 is None:
        return None
    return ScorePass(
        score01=score01,
        peakiness=peakiness(logprobs),
        method=ScoreMethod.LOGPROB,
        letter=(chosen.strip().upper()[:1] or "?"),
    )


def _score_sampled(prompt: str, config: Config) -> ScorePass | None:
    """No logprobs available: sample the scoring letter a few times and average."""
    letters: list[str] = []
    for _ in range(_SAMPLED_N):
        letter = _sample_one_letter(prompt, config)
        if letter:
            letters.append(letter)
    score01 = sampled_score01(letters)
    if score01 is None:
        return None
    return ScorePass(
        score01=score01,
        peakiness=sampled_peakiness(letters),
        method=ScoreMethod.SAMPLED,
        letter=(letters[0] if letters else "?"),
    )


def _sample_one_letter(prompt: str, config: Config) -> str:
    """One low-temperature sample of a single scoring letter (Anthropic path)."""
    url = f"{config.api_url}/messages"
    headers = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.checker_model,
        "max_tokens": 1,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
    }
    with httpx.Client(timeout=config.timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    text = data["content"][0]["text"].strip().upper()
    return text[:1] if text else ""


def generate_critique(
    claim_text: str, evidence_text: str, transcript_context: str, label: str, config: Config
) -> str:
    """One short natural-language critique naming the deciding evidence. '' on failure."""
    prompt = (
        "You are an independent auditor. In ONE sentence, state the specific transcript "
        f"evidence that makes the following claim '{label}'. Name the concrete signal "
        "(command, exit code, file, or its absence). Do not hedge.\n\n"
        f"CLAIM: {claim_text}\n\nDETERMINISTIC EVIDENCE:\n{evidence_text}\n\n"
        f"TRANSCRIPT:\n{transcript_context[:2000]}\n\nOne-sentence critique:"
    )
    try:
        return _critique_call(prompt, config).strip()
    except Exception:
        return ""


def _critique_call(prompt: str, config: Config) -> str:
    """Provider-agnostic short text completion for the critique."""
    if config.provider == JudgeProvider.OPENAI:
        url = f"{config.api_url}/chat/completions"
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": config.checker_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 120,
        }
        with httpx.Client(timeout=config.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    if config.provider == JudgeProvider.GEMINI:
        url = f"{config.api_url}/models/{config.checker_model}:generateContent?key={config.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 120},
        }
        with httpx.Client(timeout=config.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    # Anthropic
    url = f"{config.api_url}/messages"
    headers = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.checker_model,
        "max_tokens": 120,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    with httpx.Client(timeout=config.timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scorer_calls.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/receipts/scorer.py tests/test_scorer_calls.py
git commit -m "feat(scorer): logprob + sampled scoring calls and NL critique

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `criteria.py` + `criteria/*.md` — weak-verifier ensemble

**Files:**
- Create: `src/receipts/criteria.py`
- Create: `src/receipts/criteria/specification.md`, `output_match.md`, `error_signal.md`, `holistic.md`
- Modify: `pyproject.toml` (ship the `.md` fragments)
- Test: `tests/test_criteria.py`

**Interfaces:**
- Consumes: `ClaimType`, `Claim`, `Evidence`.
- Produces:
  - `CODE_CLAIM_TYPES: set[ClaimType]`.
  - `select_criteria(claim_type: ClaimType) -> list[str]` — `["specification","output_match","error_signal"]` for code claims, `["holistic"]` otherwise.
  - `load_criterion(name: str) -> str` — reads `criteria/{name}.md` via importlib.resources (cached).
  - `build_criterion_prompt(name, claim, evidence_text, transcript_context) -> str`.
  - `combine_criteria(scores: dict[str, float]) -> float` — weak-verifier combine in [0,1]; `error_signal` is min-biased (a strong contradiction sinks the claim).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_criteria.py
import pytest

from receipts.criteria import (
    CODE_CLAIM_TYPES, build_criterion_prompt, combine_criteria, load_criterion, select_criteria,
)
from receipts.models import Claim, ClaimType


def test_select_criteria_code_vs_generic():
    assert select_criteria(ClaimType.TEST_PASS) == ["specification", "output_match", "error_signal"]
    assert select_criteria(ClaimType.GENERIC) == ["holistic"]


def test_load_criterion_reads_shipped_markdown():
    text = load_criterion("error_signal")
    assert text.strip()  # non-empty fragment shipped in-package
    assert "error" in text.lower()


def test_build_criterion_prompt_includes_claim_and_evidence():
    claim = Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS)
    p = build_criterion_prompt("error_signal", claim, "exit code 1", "pytest ... FAILED")
    assert "all tests pass" in p
    assert "exit code 1" in p
    assert "pytest" in p


def test_combine_criteria_error_signal_is_min_biased():
    # A strong error signal (low) must sink an otherwise-optimistic claim.
    optimistic = {"specification": 0.9, "output_match": 0.9, "error_signal": 0.05}
    combined = combine_criteria(optimistic)
    assert combined < 0.3  # error signal dominates downward
    # With no contradicting error signal, score reflects the other criteria.
    clean = {"specification": 0.9, "output_match": 0.85, "error_signal": 0.9}
    assert combine_criteria(clean) > 0.8


def test_combine_criteria_single_holistic():
    assert combine_criteria({"holistic": 0.72}) == pytest.approx(0.72)


def test_combine_criteria_empty_returns_half():
    assert combine_criteria({}) == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_criteria.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'receipts.criteria'`.

- [ ] **Step 3: Write minimal implementation**

Create the four markdown fragments:

`src/receipts/criteria/specification.md`:
```markdown
Assess ONLY whether the change the agent made matches what the claim says it did.
Did the agent actually perform the action described (edit the right file, implement
the described behavior)? Ignore whether tests pass — judge specification match only.
```

`src/receipts/criteria/output_match.md`:
```markdown
Assess ONLY whether the observed output or behavior in the transcript matches what the
claim asserts. Compare the claim against actual command output, printed results, or
returned values. If the transcript shows no output that confirms the claimed behavior,
score low.
```

`src/receipts/criteria/error_signal.md`:
```markdown
Assess ONLY whether there are error signals in the transcript that CONTRADICT the claim:
failing tests, non-zero exit codes, tracebacks, exceptions, or compiler/linter errors
relevant to the claim. A strong contradicting error signal means the claim is refuted,
even if other things look fine. If there are no relevant error signals, score high.
```

`src/receipts/criteria/holistic.md`:
```markdown
Holistically assess whether the transcript evidence supports the agent's claim. Weigh
what the agent said it did against what the tool calls, command outputs, and file
changes actually show. Score low when the claim is unsupported or contradicted, high
when the evidence clearly confirms it.
```

`src/receipts/criteria.py`:
```python
"""Criteria decomposition — a weak-verifier ensemble over a claim (2506.18203).

Code claims are scored on three sub-criteria (the base paper's coding triad);
non-code/generic claims use a single holistic criterion. `error_signal` is
min-biased in combination so one strong contradiction sinks an optimistic claim.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from receipts.models import Claim, ClaimType

CODE_CLAIM_TYPES: set[ClaimType] = {
    ClaimType.TEST_PASS,
    ClaimType.BUG_FIXED,
    ClaimType.FEATURE_ADDED,
    ClaimType.REFACTORED,
    ClaimType.FILE_MODIFIED,
    ClaimType.COMMAND_RUN,
}

_CODE_CRITERIA = ["specification", "output_match", "error_signal"]
_GENERIC_CRITERIA = ["holistic"]


def select_criteria(claim_type: ClaimType) -> list[str]:
    """Pick the criterion set for a claim type."""
    return list(_CODE_CRITERIA) if claim_type in CODE_CLAIM_TYPES else list(_GENERIC_CRITERIA)


@lru_cache(maxsize=None)
def load_criterion(name: str) -> str:
    """Load a shipped criterion prompt fragment by name."""
    return (resources.files("receipts") / "criteria" / f"{name}.md").read_text(encoding="utf-8")


def build_criterion_prompt(
    name: str, claim: Claim, evidence_text: str, transcript_context: str
) -> str:
    """Assemble the scoring prompt body for one criterion."""
    return (
        f"{load_criterion(name)}\n\n"
        f"CLAIM: {claim.text}\n\n"
        f"DETERMINISTIC EVIDENCE:\n{evidence_text or 'None found.'}\n\n"
        f"TRANSCRIPT CONTEXT:\n{transcript_context[:3000]}"
    )


def combine_criteria(scores: dict[str, float]) -> float:
    """Combine per-criterion 0-1 scores into one 0-1 score.

    error_signal is min-biased: the combined score is pulled toward the minimum of the
    mean and the error_signal score, so a strong contradiction dominates. Empty → 0.5.
    """
    if not scores:
        return 0.5
    mean = sum(scores.values()) / len(scores)
    if "error_signal" in scores:
        # Weighted pull toward the error signal when it is low.
        return min(mean, 0.5 * mean + 0.5 * scores["error_signal"])
    return mean
```

Add package-data config to `pyproject.toml` (after the existing `[tool.hatch.build.targets.wheel]` block):

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/receipts/criteria" = "receipts/criteria"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_criteria.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/receipts/criteria.py src/receipts/criteria/ pyproject.toml tests/test_criteria.py
git commit -m "feat(criteria): weak-verifier ensemble with shipped prompt fragments

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `verifier.py` — fast-path, adaptive K, abstention, three-way labeling

**Files:**
- Create: `src/receipts/verifier.py`
- Test: `tests/test_verifier.py`

**Interfaces:**
- Consumes: `Claim`, `Evidence`, `ClaimType`, `FactLabel`, `ScoreMethod`; `Config`, `check_independence`; `scorer.ScorePass`, `scorer.score_pass`, `scorer.generate_critique`; `criteria.select_criteria`, `criteria.build_criterion_prompt`, `criteria.combine_criteria`; `evidence.match_evidence`; `ParsedTranscript`.
- Produces:
  - Bands/thresholds: `SUPPORTED_BAND = 85.0`, `REFUTED_BAND = 45.0`, `ESCALATE_PEAKINESS = 0.75`.
  - `@dataclass ClaimVerdict{ claim; label; score; confidence; per_criterion; critique; method; passes; evidence }`.
  - `@dataclass SessionReport{ verdicts: list[ClaimVerdict]; checker_model: str; checker_independent: bool; duration_ms: int }`.
  - `fast_path(claim, evidence) -> ClaimVerdict | None` — deterministic entailment.
  - `aggregate_confidence(score01s: list[float], peakinesses: list[float]) -> float`.
  - `label_for(score: float, confidence: float, min_confidence: float) -> FactLabel`.
  - `verify_claim(claim, evidence, transcript_context, config, *, agent="claude-code") -> ClaimVerdict`.
  - `verify_session(claims, transcript, transcript_context, config, *, agent="claude-code") -> SessionReport`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verifier.py
from unittest.mock import patch

from receipts.config import Config, JudgeProvider
from receipts.models import (
    Claim, ClaimType, Evidence, EvidenceSource, FactLabel, ScoreMethod,
)
from receipts.scorer import ScorePass
from receipts.verifier import (
    aggregate_confidence, fast_path, label_for, verify_claim,
)


def _cfg(**kw):
    return Config(provider=JudgeProvider.OPENAI, model="m", api_key="k", **kw)


def _sp(score01, peak=0.95):
    return ScorePass(score01=score01, peakiness=peak, method=ScoreMethod.LOGPROB, letter="T")


def test_fast_path_refutes_failing_tests_without_llm():
    claim = Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS)
    ev = [Evidence(source=EvidenceSource.EXIT_CODE, content="exit 1", supports_claim=False)]
    v = fast_path(claim, ev)
    assert v is not None
    assert v.label == FactLabel.REFUTED
    assert v.score == 0.0
    assert v.method == ScoreMethod.DETERMINISTIC


def test_fast_path_supports_created_file_without_llm():
    claim = Claim(text="created src/utils.py", claim_type=ClaimType.FILE_CREATED)
    ev = [Evidence(source=EvidenceSource.FILE_WRITE, content="wrote src/utils.py", supports_claim=True)]
    v = fast_path(claim, ev)
    assert v is not None and v.label == FactLabel.SUPPORTED and v.score == 100.0


def test_fast_path_returns_none_for_fuzzy_claim():
    claim = Claim(text="improved error handling", claim_type=ClaimType.GENERIC)
    assert fast_path(claim, []) is None


def test_label_for_bands_and_abstention():
    assert label_for(90.0, confidence=0.9, min_confidence=0.6) == FactLabel.SUPPORTED
    assert label_for(10.0, confidence=0.9, min_confidence=0.6) == FactLabel.REFUTED
    assert label_for(65.0, confidence=0.9, min_confidence=0.6) == FactLabel.NOT_ENOUGH_INFO
    # Low confidence forces abstention even on a decisive score.
    assert label_for(95.0, confidence=0.3, min_confidence=0.6) == FactLabel.NOT_ENOUGH_INFO


def test_aggregate_confidence_agreement_and_peakiness():
    # Perfect agreement + high peakiness → high confidence.
    assert aggregate_confidence([0.8, 0.8], [0.9, 0.9]) > 0.85
    # Disagreement collapses confidence.
    assert aggregate_confidence([0.1, 0.9], [0.9, 0.9]) < 0.5


def test_verify_claim_high_supported_single_pass():
    claim = Claim(text="improved logging", claim_type=ClaimType.GENERIC)
    with patch("receipts.verifier.score_pass", return_value=_sp(0.95, 0.95)) as m, \
         patch("receipts.verifier.generate_critique", return_value="log calls added"):
        v = verify_claim(claim, [], "ctx", _cfg(score_passes=3))
    assert v.label == FactLabel.SUPPORTED
    assert v.passes == 1  # clear + peaked → no escalation
    m.assert_called_once()


def test_verify_claim_escalates_on_low_peakiness():
    claim = Claim(text="refactored the parser", claim_type=ClaimType.REFACTORED)
    # First pass low peakiness → escalate; code claim has 3 criteria × passes calls.
    with patch("receipts.verifier.score_pass", return_value=_sp(0.6, 0.4)), \
         patch("receipts.verifier.generate_critique", return_value="unclear"):
        v = verify_claim(claim, [], "ctx", _cfg(score_passes=3))
    assert v.passes == 3  # escalated to K_max


def test_verify_claim_abstains_when_scorer_unavailable():
    claim = Claim(text="cleaned up things", claim_type=ClaimType.GENERIC)
    with patch("receipts.verifier.score_pass", return_value=None):
        v = verify_claim(claim, [], "ctx", _cfg())
    assert v.label == FactLabel.NOT_ENOUGH_INFO
    assert v.method == ScoreMethod.NONE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_verifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'receipts.verifier'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/receipts/verifier.py
"""The verifier core: fast-path → calibrated scorer ensemble → uncertainty control.

Produces a three-way calibrated verdict (SUPPORTED / REFUTED / NOT_ENOUGH_INFO) per
claim with a 0-100 score, confidence, per-criterion breakdown, and a specific critique.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean, pstdev

from receipts.config import Config, check_independence
from receipts.criteria import build_criterion_prompt, combine_criteria, select_criteria
from receipts.evidence import match_evidence
from receipts.models import (
    Claim, Evidence, FactLabel, ParsedTranscript, ScoreMethod,
)
from receipts.scorer import ScorePass, generate_critique, score_pass

SUPPORTED_BAND = 85.0       # score ≥ this → SUPPORTED (when not abstaining)
REFUTED_BAND = 45.0         # score < this → REFUTED (when not abstaining)
ESCALATE_PEAKINESS = 0.75   # first-pass peakiness below this triggers adaptive-K escalation


@dataclass
class ClaimVerdict:
    """The verifier's decision for one claim."""

    claim: Claim
    label: FactLabel
    score: float                       # 0..100
    confidence: float                  # 0..1
    per_criterion: dict[str, float]    # criterion name → 0..100
    critique: str
    method: ScoreMethod
    passes: int
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class SessionReport:
    """All claim verdicts for a session plus checker metadata."""

    verdicts: list[ClaimVerdict]
    checker_model: str
    checker_independent: bool
    duration_ms: int


def label_for(score: float, confidence: float, min_confidence: float) -> FactLabel:
    """Apply score→label bands, with confidence-gated abstention (meta-honesty)."""
    if confidence < min_confidence:
        return FactLabel.NOT_ENOUGH_INFO
    if score >= SUPPORTED_BAND:
        return FactLabel.SUPPORTED
    if score < REFUTED_BAND:
        return FactLabel.REFUTED
    return FactLabel.NOT_ENOUGH_INFO


def aggregate_confidence(score01s: list[float], peakinesses: list[float]) -> float:
    """Confidence from cross-pass agreement × mean distribution peakiness.

    Agreement = 1 - min(1, 2·pstdev(per-pass scores)); with <2 passes agreement is 1.0
    and confidence is carried entirely by peakiness.
    """
    if not score01s:
        return 0.0
    agreement = 1.0 - min(1.0, 2.0 * pstdev(score01s)) if len(score01s) > 1 else 1.0
    mean_peak = mean(peakinesses) if peakinesses else 0.0
    return agreement * mean_peak


def fast_path(claim: Claim, evidence: list[Evidence]) -> ClaimVerdict | None:
    """Deterministic entailment: decide with certainty when hard evidence is conclusive."""
    if not evidence:
        return None
    # Hard contradiction (e.g. failing test exit code) → REFUTED with certainty.
    if all(not e.supports_claim for e in evidence):
        return ClaimVerdict(
            claim=claim, label=FactLabel.REFUTED, score=0.0, confidence=1.0,
            per_criterion={}, critique=evidence[0].content, method=ScoreMethod.DETERMINISTIC,
            passes=0, evidence=evidence,
        )
    # Hard support (e.g. matching file write) → SUPPORTED with certainty.
    if all(e.supports_claim for e in evidence):
        return ClaimVerdict(
            claim=claim, label=FactLabel.SUPPORTED, score=100.0, confidence=1.0,
            per_criterion={}, critique=evidence[0].content, method=ScoreMethod.DETERMINISTIC,
            passes=0, evidence=evidence,
        )
    return None  # mixed evidence → send to the LLM scorer


def _abstain(claim: Claim, evidence: list[Evidence], reason: str) -> ClaimVerdict:
    """Evidence-only / no-checker abstention (deepest fallback rung)."""
    return ClaimVerdict(
        claim=claim, label=FactLabel.NOT_ENOUGH_INFO, score=50.0, confidence=0.0,
        per_criterion={}, critique=reason, method=ScoreMethod.NONE, passes=0, evidence=evidence,
    )


def verify_claim(
    claim: Claim,
    evidence: list[Evidence],
    transcript_context: str,
    config: Config,
    *,
    agent: str = "claude-code",
) -> ClaimVerdict:
    """Verify one claim: fast-path, else adaptive-K calibrated scoring + abstention."""
    fp = fast_path(claim, evidence)
    if fp is not None:
        return fp

    evidence_text = "\n".join(f"- {e.source.value}: {e.content}" for e in evidence)
    criteria = select_criteria(claim.claim_type)
    prompts = {c: build_criterion_prompt(c, claim, evidence_text, transcript_context)
               for c in criteria}

    # Pass 1.
    pass_criteria = _run_pass(prompts, config)
    if pass_criteria is None:
        return _abstain(claim, evidence, "No calibrated checker available for this claim.")

    per_pass_criteria: list[dict[str, ScorePass]] = [pass_criteria]
    method = next(iter(pass_criteria.values())).method

    # Adaptive-K escalation: low first-pass peakiness → run up to K_max passes.
    first_peak = mean(sp.peakiness for sp in pass_criteria.values())
    if first_peak < ESCALATE_PEAKINESS and config.score_passes > 1:
        for _ in range(config.score_passes - 1):
            extra = _run_pass(prompts, config)
            if extra is not None:
                per_pass_criteria.append(extra)

    # Combine: per criterion average across passes → weak-verifier combine → 0..100.
    per_criterion01: dict[str, float] = {}
    for name in criteria:
        vals = [p[name].score01 for p in per_pass_criteria if name in p]
        per_criterion01[name] = mean(vals) if vals else 0.5
    combined01 = combine_criteria(per_criterion01)
    score = combined01 * 100.0

    # Confidence: per-pass combined scores + mean peakiness.
    per_pass_combined = [
        combine_criteria({n: p[n].score01 for n in p}) for p in per_pass_criteria
    ]
    peaks = [sp.peakiness for p in per_pass_criteria for sp in p.values()]
    confidence = aggregate_confidence(per_pass_combined, peaks)

    label = label_for(score, confidence, config.min_confidence)
    critique = generate_critique(
        claim.text, evidence_text, transcript_context, label.value, config
    ) or "No specific critique available."

    return ClaimVerdict(
        claim=claim,
        label=label,
        score=score,
        confidence=confidence,
        per_criterion={n: v * 100.0 for n, v in per_criterion01.items()},
        critique=critique,
        method=method,
        passes=len(per_pass_criteria),
        evidence=evidence,
    )


def _run_pass(prompts: dict[str, str], config: Config) -> dict[str, ScorePass] | None:
    """Score every criterion once. Returns None if the scorer is entirely unavailable."""
    out: dict[str, ScorePass] = {}
    for name, prompt in prompts.items():
        sp = score_pass(prompt, config)
        if sp is not None:
            out[name] = sp
    return out or None


def verify_session(
    claims: list[Claim],
    transcript: ParsedTranscript,
    transcript_context: str,
    config: Config,
    *,
    agent: str = "claude-code",
) -> SessionReport:
    """Verify every claim in a session and assemble the report."""
    start = time.time()
    independent = check_independence(config, agent)
    verdicts = [
        verify_claim(claim, match_evidence(claim, transcript), transcript_context,
                     config, agent=agent)
        for claim in claims
    ]
    return SessionReport(
        verdicts=verdicts,
        checker_model=config.checker_model,
        checker_independent=independent,
        duration_ms=int((time.time() - start) * 1000),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_verifier.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/receipts/verifier.py tests/test_verifier.py
git commit -m "feat(verifier): fast-path, adaptive-K, abstention, three-way labeling

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Wire `verify_session` into the hook pipeline

**Files:**
- Modify: `src/receipts/hook.py`
- Test: `tests/test_hook_v2.py`

**Interfaces:**
- Consumes: `verify_session`, `SessionReport`, `ClaimVerdict`, `verdict_for_label`, existing `parse_transcript`, `extract_claims`, `_build_transcript_context`, `Receipt`, `save_receipt`, `render_receipt`.
- Produces: `verdict_to_verified_claim(cv: ClaimVerdict) -> VerifiedClaim` and a rewritten `process_transcript` that builds the `Receipt` from a `SessionReport`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hook_v2.py
from unittest.mock import patch

from receipts.config import Config, JudgeProvider
from receipts.hook import verdict_to_verified_claim
from receipts.models import (
    Claim, ClaimType, FactLabel, ScoreMethod, Verdict,
)
from receipts.verifier import ClaimVerdict


def test_verdict_to_verified_claim_maps_fields():
    cv = ClaimVerdict(
        claim=Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS),
        label=FactLabel.REFUTED, score=4.0, confidence=0.95,
        per_criterion={"error_signal": 2.0}, critique="pytest exit 1",
        method=ScoreMethod.LOGPROB, passes=1, evidence=[],
    )
    vc = verdict_to_verified_claim(cv)
    assert vc.verdict == Verdict.REFUTED          # back-compat mapping
    assert vc.label == FactLabel.REFUTED
    assert vc.score == 4.0
    assert vc.critique == "pytest exit 1"
    assert vc.reasoning == "pytest exit 1"        # critique mirrored for old renderers
    assert vc.method == ScoreMethod.LOGPROB
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hook_v2.py -v`
Expected: FAIL — `ImportError: cannot import name 'verdict_to_verified_claim'`.

- [ ] **Step 3: Write minimal implementation**

In `src/receipts/hook.py`, replace the `judge_claim` import and the claim-verification loop. New imports at top:

```python
from receipts.models import Receipt, VerifiedClaim, verdict_for_label
from receipts.verifier import ClaimVerdict, verify_session
```

(Remove the `from receipts.judge import judge_claim` and `Verdict` imports if now unused — keep `Verdict` only if still referenced.)

Add the mapping helper:

```python
def verdict_to_verified_claim(cv: ClaimVerdict) -> VerifiedClaim:
    """Convert a v2 ClaimVerdict into the persisted VerifiedClaim (with back-compat verdict)."""
    return VerifiedClaim(
        claim=cv.claim,
        verdict=verdict_for_label(cv.label),
        evidence=cv.evidence,
        reasoning=cv.critique,       # mirror critique into reasoning for v1 renderers/tools
        label=cv.label,
        score=cv.score,
        confidence=cv.confidence,
        per_criterion=cv.per_criterion,
        critique=cv.critique,
        method=cv.method,
        passes=cv.passes,
    )
```

Replace the body of `process_transcript` between "3. Verify claims" and "4. Generate receipt" with:

```python
        # 3. Verify claims (calibrated three-way verifier)
        transcript_context = _build_transcript_context(transcript)
        report = verify_session(claims, transcript, transcript_context, config)
        verified_claims = [verdict_to_verified_claim(cv) for cv in report.verdicts]

        # 4. Generate receipt
        receipt = Receipt(
            session_id=transcript.session_id,
            user_request=transcript.user_request[:500],
            claims=verified_claims,
            judge_model=report.checker_model,
            judge_duration_ms=report.duration_ms,
            checker_independent=report.checker_independent,
        )
```

Remove the now-dead `start_time`/`duration_ms` timing (the report carries duration) and the old per-claim `match_evidence`/`judge_claim` loop.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hook_v2.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Verify nothing else regressed, then commit**

Run: `python -m pytest -q`
Expected: PASS (all suites green).

```bash
git add src/receipts/hook.py tests/test_hook_v2.py
git commit -m "feat(hook): route claims through the calibrated verify_session pipeline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: `render.py` — three-way calibrated receipt card

**Files:**
- Modify: `src/receipts/render.py`
- Test: `tests/test_render_v2.py`

**Interfaces:**
- Consumes: `Receipt`, `VerifiedClaim`, `FactLabel`, `ScoreMethod`. Renders to any `TextIO`.
- Produces: upgraded `render_receipt` — header shows `Verified-Done Score: NN/100` and ✅/❌/❔ counts; per-claim glyph + score + critique + per-criterion mini-bars (code claims) + method tag; a maker≠checker warning line when `checker_independent is False`. `render_receipt_json` unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_render_v2.py
import io

from receipts.models import (
    Claim, ClaimType, FactLabel, Receipt, ScoreMethod, VerifiedClaim, Verdict,
)
from receipts.render import render_receipt


def _receipt(**kw):
    claims = [
        VerifiedClaim(
            claim=Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS),
            verdict=Verdict.REFUTED, label=FactLabel.REFUTED, score=4.0, confidence=0.95,
            per_criterion={"error_signal": 2.0}, critique="pytest exit 1, 2 failures",
            method=ScoreMethod.LOGPROB, passes=1,
        ),
        VerifiedClaim(
            claim=Claim(text="improved error handling", claim_type=ClaimType.GENERIC),
            verdict=Verdict.UNVERIFIED, label=FactLabel.NOT_ENOUGH_INFO, score=60.0,
            confidence=0.4, critique="no try/except changes found", method=ScoreMethod.SAMPLED,
            passes=3,
        ),
    ]
    return Receipt(session_id="s1", claims=claims, judge_model="gemini-3-flash", **kw)


def test_card_shows_verified_done_score_and_counts():
    buf = io.StringIO()
    render_receipt(_receipt(), file=buf)
    out = buf.getvalue()
    assert "Verified-Done Score" in out
    assert "/100" in out
    assert "pytest exit 1" in out                 # critique rendered
    assert "improved error handling" in out


def test_card_shows_method_tags():
    buf = io.StringIO()
    render_receipt(_receipt(), file=buf)
    out = buf.getvalue().lower()
    assert "calibrated" in out or "logprob" in out
    assert "sampled" in out


def test_card_warns_when_not_independent():
    buf = io.StringIO()
    render_receipt(_receipt(checker_independent=False), file=buf)
    assert "independ" in buf.getvalue().lower()  # maker≠checker warning surfaced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_render_v2.py -v`
Expected: FAIL — assertion error (`Verified-Done Score` not present; card still v1).

- [ ] **Step 3: Write minimal implementation**

Rewrite `render_receipt` in `src/receipts/render.py`:

```python
from receipts.models import FactLabel, Receipt, ScoreMethod

_GLYPH = {
    FactLabel.SUPPORTED: ("✅", "green"),
    FactLabel.REFUTED: ("❌", "red"),
    FactLabel.NOT_ENOUGH_INFO: ("❔", "yellow"),
}
_METHOD_TAG = {
    ScoreMethod.LOGPROB: "calibrated",
    ScoreMethod.SAMPLED: "sampled",
    ScoreMethod.DETERMINISTIC: "deterministic",
    ScoreMethod.NONE: "evidence-only",
}


def _bar(value01: float, width: int = 10) -> str:
    filled = max(0, min(width, round(value01 * width)))
    return "█" * filled + "░" * (width - filled)


def render_receipt(receipt: Receipt, file: TextIO = sys.stderr) -> None:
    """Render a calibrated three-way receipt card."""
    console = Console(file=file)
    vds = receipt.verified_done_score
    score_color = "green" if vds >= 85 else "yellow" if vds >= 45 else "red"

    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        Text("🧾 RECEIPT", style="bold white"),
        Text(f"Verified-Done Score: {vds:.0f}/100", style=f"bold {score_color}"),
    )
    counts = Text(
        f"✅ {receipt.supported_count} supported   "
        f"❌ {receipt.refuted_count} refuted   "
        f"❔ {receipt.nei_count} not-enough-info",
        style="dim",
    )

    body = Text()
    if not receipt.claims:
        body.append("\nNo completion claims found.", style="dim")
    else:
        body.append("\n")
        for vc in receipt.claims:
            label = vc.label or FactLabel.NOT_ENOUGH_INFO
            glyph, color = _GLYPH[label]
            score_str = f"{vc.score:.0f}" if vc.score is not None else "--"
            tag = _METHOD_TAG.get(vc.method, "") if vc.method else ""
            body.append(f"  {glyph} {vc.claim.text}  ", style=color)
            body.append(f"[{score_str}/100 · {tag}]\n", style="dim")
            if vc.critique:
                body.append(f"     └─ {vc.critique}\n", style="dim")
            for name, cscore in (vc.per_criterion or {}).items():
                body.append(f"        {name:<14} {_bar(cscore / 100.0)} {cscore:.0f}\n", style="dim")

    if not receipt.checker_independent:
        body.append(
            "\n  ⚠️  checker shares the audited agent's provider — independence not guaranteed\n",
            style="yellow",
        )

    footer = Table.grid(expand=True)
    footer.add_column()
    footer.add_column()
    footer.add_row(
        Text(f"Session: {receipt.session_id} │ Agent: {receipt.agent}", style="dim"),
        Text(
            f"Checker: {receipt.judge_model} │ {receipt.judge_duration_ms / 1000:.1f}s",
            style="dim", justify="right",
        ),
    )

    panel_group = Table.grid(expand=True)
    panel_group.add_column()
    panel_group.add_row(header)
    panel_group.add_row(counts)
    panel_group.add_row(body)
    panel_group.add_row(Text("─" * 50, style="dim"))
    panel_group.add_row(footer)

    console.print(Panel(panel_group, expand=False, border_style=score_color, padding=(1, 2)))
```

Remove the now-unused `from receipts.models import Receipt, Verdict` line (replaced above) — keep only what the new code imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_render_v2.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/receipts/render.py tests/test_render_v2.py
git commit -m "feat(render): three-way calibrated card with score, critique, criterion bars

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: `stats.py` numeric aggregates + README stat fix

**Files:**
- Modify: `src/receipts/stats.py`
- Modify: `README.md`
- Test: `tests/test_stats_v2.py`

**Interfaces:**
- Consumes: `Receipt`, `FactLabel`, existing `load_receipts`, `save_receipt`.
- Produces: `render_stats` reporting a mean Verified-Done Score, supported/refuted/NEI totals, and abstention rate; backward-compatible with v1 records (missing `score`/`label` treated as verdict-only).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stats_v2.py
import io

from receipts.models import (
    Claim, ClaimType, FactLabel, Receipt, ScoreMethod, VerifiedClaim, Verdict,
)
from receipts.stats import render_stats


def _receipt():
    return Receipt(
        session_id="s1",
        claims=[
            VerifiedClaim(claim=Claim(text="a"), verdict=Verdict.VERIFIED,
                          label=FactLabel.SUPPORTED, score=95.0),
            VerifiedClaim(claim=Claim(text="b"), verdict=Verdict.REFUTED,
                          label=FactLabel.REFUTED, score=5.0),
            VerifiedClaim(claim=Claim(text="c"), verdict=Verdict.UNVERIFIED,
                          label=FactLabel.NOT_ENOUGH_INFO, score=60.0),
        ],
    )


def test_stats_reports_calibrated_aggregates():
    buf = io.StringIO()
    render_stats([_receipt()], days=7, file=buf)
    out = buf.getvalue()
    assert "Verified-Done Score" in out
    assert "refuted" in out.lower()
    assert "not-enough-info" in out.lower() or "abstain" in out.lower()


def test_stats_handles_v1_records_without_scores():
    v1 = Receipt(
        session_id="old",
        claims=[VerifiedClaim(claim=Claim(text="x"), verdict=Verdict.VERIFIED)],
    )
    buf = io.StringIO()
    render_stats([v1], days=7, file=buf)  # must not raise
    assert "session" in buf.getvalue().lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stats_v2.py -v`
Expected: FAIL — `Verified-Done Score` not in stats output.

- [ ] **Step 3: Write minimal implementation**

In `src/receipts/stats.py`, extend `render_stats`. Add after the existing `verification_rate` computation:

```python
    # v2 calibrated aggregates (fall back gracefully for v1 records).
    all_scores = [c.score for r in receipts for c in r.claims if c.score is not None]
    mean_vds = sum(all_scores) / len(all_scores) if all_scores else None
    supported = sum(r.supported_count for r in receipts)
    refuted = sum(r.refuted_count for r in receipts)
    nei = sum(r.nei_count for r in receipts)
    labeled = supported + refuted + nei
    abstain_rate = (nei / labeled * 100) if labeled else 0.0
```

And add these nodes to the tree (after the existing `verification_rate` node):

```python
    if mean_vds is not None:
        vds_color = "green" if mean_vds >= 85 else "yellow" if mean_vds >= 45 else "red"
        tree.add(Text(f"Verified-Done Score: {mean_vds:.0f}/100", style=f"bold {vds_color}"))
        tree.add(
            f"{supported} supported · {refuted} refuted · {nei} not-enough-info "
            f"({abstain_rate:.0f}% abstained)"
        )
```

Add `from receipts.models import FactLabel` to the imports (alongside `Receipt, Verdict`).

Fix `README.md` line 22 — replace the unsourced figure:

```markdown
> **The problem:** AI coding agents confidently claim "Done! All tests pass!" — but a 20,574-session study found **22.58% of agent misalignment episodes are inaccurate self-reporting** (arXiv 2605.29442): the agent claims success without having verified it. You don't find out until three tasks are stacked on top and everything breaks.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stats_v2.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify the README no longer contains the unsourced figure, then commit**

Run: `grep -n "47%" README.md || echo "no unsourced 47% remaining"`
Expected: `no unsourced 47% remaining`.

```bash
git add src/receipts/stats.py README.md tests/test_stats_v2.py
git commit -m "feat(stats): calibrated aggregates; fix README stat to cited 22.58%

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: `pivot_tournament.py` — Probabilistic Pivot Tournament (Phase 2)

**Files:**
- Create: `src/receipts/pivot_tournament.py`
- Test: `tests/test_pivot_tournament.py`

**Interfaces:**
- Produces:
  - `DEFAULT_PIVOTS = 2`.
  - `bradley_terry_prob(ra: float, rb: float) -> float` — `1/(1+exp(-(ra-rb)))`.
  - `@dataclass Candidate{ id: str; rating: float }` (rating = calibrated 0..1 score of the candidate).
  - `pivot_tournament(candidates: list[Candidate], pivots: int = DEFAULT_PIVOTS) -> Candidate` — best-of-N by pairwise Bradley-Terry comparisons against `pivots` pivots; O(N·k).
  - `comparison_count(n: int, pivots: int) -> int` — number of pairwise comparisons performed (for the O(Nk) test).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pivot_tournament.py
import math

import pytest

from receipts.pivot_tournament import (
    Candidate, DEFAULT_PIVOTS, bradley_terry_prob, comparison_count, pivot_tournament,
)


def test_bradley_terry_symmetry_and_midpoint():
    assert bradley_terry_prob(1.0, 1.0) == pytest.approx(0.5)
    assert bradley_terry_prob(2.0, 0.0) > 0.8
    assert bradley_terry_prob(0.0, 2.0) < 0.2


def test_default_pivots_is_two():
    assert DEFAULT_PIVOTS == 2


def test_pivot_tournament_selects_highest_rated():
    cands = [Candidate(id="a", rating=0.2), Candidate(id="b", rating=0.9),
             Candidate(id="c", rating=0.5)]
    assert pivot_tournament(cands).id == "b"


def test_pivot_tournament_single_candidate():
    only = Candidate(id="solo", rating=0.3)
    assert pivot_tournament([only]).id == "solo"


def test_comparison_count_is_linear_Nk():
    # O(N·k): each of N candidates compared against `pivots` pivots.
    assert comparison_count(10, 2) == 20
    assert comparison_count(5, 2) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pivot_tournament.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'receipts.pivot_tournament'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/receipts/pivot_tournament.py
"""Probabilistic Pivot Tournament — best-of-N selection in O(N·k).

Ported from the base paper's reference repo. Pairwise Bradley-Terry comparisons against
a small set of pivots avoid the instability of pointwise scores while staying linear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_PIVOTS = 2


@dataclass
class Candidate:
    """A ranking candidate with a calibrated 0..1 rating."""

    id: str
    rating: float


def bradley_terry_prob(ra: float, rb: float) -> float:
    """P(a beats b) = 1 / (1 + exp(-(ra - rb)))."""
    return 1.0 / (1.0 + math.exp(-(ra - rb)))


def comparison_count(n: int, pivots: int) -> int:
    """Number of pairwise comparisons performed for n candidates and `pivots` pivots."""
    return n * pivots


def pivot_tournament(
    candidates: list[Candidate], pivots: int = DEFAULT_PIVOTS
) -> Candidate:
    """Select the best candidate via pairwise Bradley-Terry wins against `pivots` pivots."""
    if not candidates:
        raise ValueError("no candidates to rank")
    if len(candidates) == 1:
        return candidates[0]

    ordered = sorted(candidates, key=lambda c: c.rating, reverse=True)
    pivot_set = ordered[: max(1, min(pivots, len(ordered)))]

    def expected_wins(c: Candidate) -> float:
        return sum(bradley_terry_prob(c.rating, p.rating) for p in pivot_set)

    return max(candidates, key=expected_wins)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pivot_tournament.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/receipts/pivot_tournament.py tests/test_pivot_tournament.py
git commit -m "feat(ppt): Probabilistic Pivot Tournament best-of-N selection

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: `retry.py` — self-healing (generate → score → PPT → present → opt-in apply)

**Files:**
- Create: `src/receipts/retry.py`
- Test: `tests/test_retry.py`

**Interfaces:**
- Consumes: `ClaimVerdict`, `FactLabel`, `Config`; `pivot_tournament.Candidate`, `pivot_tournament`; `scorer.score_pass`; `criteria.build_criterion_prompt`.
- Produces:
  - `DEFAULT_CANDIDATES = 3`.
  - `@dataclass FixProposal{ diff: str; rating: float; explanation: str }`.
  - `generate_fixes(verdict, transcript_context, config, n=DEFAULT_CANDIDATES) -> list[FixProposal]` — asks the checker for N candidate unified diffs.
  - `score_fix(fix, verdict, transcript_context, config) -> float` — calibrated 0..1 rating of a candidate.
  - `select_fix(fixes, config) -> FixProposal | None` — score each, PPT-select; `None` if empty.
  - `propose_fix(verdict, transcript_context, config) -> FixProposal | None` — end-to-end for one REFUTED verdict (no writes).
  - `apply_fix(fix, cwd) -> bool` — writes the diff via `git apply` **only when called** (caller gates on opt-in).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retry.py
from unittest.mock import patch

from receipts.config import Config, JudgeProvider
from receipts.models import Claim, ClaimType, FactLabel, ScoreMethod
from receipts.retry import (
    DEFAULT_CANDIDATES, FixProposal, propose_fix, select_fix,
)
from receipts.verifier import ClaimVerdict


def _cfg(**kw):
    return Config(provider=JudgeProvider.OPENAI, model="m", api_key="k", **kw)


def _refuted():
    return ClaimVerdict(
        claim=Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS),
        label=FactLabel.REFUTED, score=5.0, confidence=0.9, per_criterion={},
        critique="pytest exit 1", method=ScoreMethod.LOGPROB, passes=1, evidence=[],
    )


def test_select_fix_picks_highest_rated():
    fixes = [
        FixProposal(diff="diff-a", rating=0.2, explanation="a"),
        FixProposal(diff="diff-b", rating=0.9, explanation="b"),
    ]
    with patch("receipts.retry.score_fix", side_effect=lambda f, *a, **k: f.rating):
        best = select_fix(fixes, _cfg())
    assert best.diff == "diff-b"


def test_select_fix_empty_returns_none():
    assert select_fix([], _cfg()) is None


def test_propose_fix_returns_winning_proposal():
    fixes = [FixProposal(diff="d1", rating=0.0, explanation="x"),
             FixProposal(diff="d2", rating=0.0, explanation="y")]
    with patch("receipts.retry.generate_fixes", return_value=fixes), \
         patch("receipts.retry.score_fix", side_effect=[0.3, 0.8]):
        out = propose_fix(_refuted(), "ctx", _cfg())
    assert out.diff == "d2"


def test_default_candidates_is_three():
    assert DEFAULT_CANDIDATES == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'receipts.retry'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/receipts/retry.py
"""Phase 2 self-healing: generate candidate fixes, rank with PPT, present, opt-in apply.

External ranking is what makes this work — self-correction alone collapses (2310.01798);
the generation-verification gap is closed by an independent verifier (2506.18203).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from receipts.config import Config, JudgeProvider
from receipts.pivot_tournament import Candidate, pivot_tournament
from receipts.scorer import score_pass
from receipts.verifier import ClaimVerdict

DEFAULT_CANDIDATES = 3


@dataclass
class FixProposal:
    """A candidate fix as a unified diff with a calibrated rating."""

    diff: str
    rating: float
    explanation: str


def generate_fixes(
    verdict: ClaimVerdict,
    transcript_context: str,
    config: Config,
    n: int = DEFAULT_CANDIDATES,
) -> list[FixProposal]:
    """Ask the checker for N candidate unified diffs that would make the claim true."""
    prompt = (
        "An AI coding agent claimed the following, but it was REFUTED by the evidence. "
        "Propose a single unified-diff patch that would make the claim genuinely true. "
        "Respond as JSON: {\"diff\": \"<unified diff>\", \"explanation\": \"<one line>\"}.\n\n"
        f"CLAIM: {verdict.claim.text}\n"
        f"WHY REFUTED: {verdict.critique}\n\n"
        f"TRANSCRIPT:\n{transcript_context[:3000]}"
    )
    fixes: list[FixProposal] = []
    for _ in range(n):
        raw = _completion(prompt, config, max_tokens=800, temperature=0.8)
        parsed = _parse_fix_json(raw)
        if parsed is not None:
            fixes.append(parsed)
    return fixes


def _parse_fix_json(raw: str) -> FixProposal | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    diff = obj.get("diff", "").strip()
    if not diff:
        return None
    return FixProposal(diff=diff, rating=0.0, explanation=obj.get("explanation", ""))


def score_fix(
    fix: FixProposal, verdict: ClaimVerdict, transcript_context: str, config: Config
) -> float:
    """Calibrated 0..1 rating of how well a candidate diff resolves the refuted claim."""
    prompt = (
        f"CLAIM: {verdict.claim.text}\n"
        f"PROPOSED PATCH:\n{fix.diff}\n\n"
        f"CONTEXT:\n{transcript_context[:2000]}\n\n"
        "Score whether this patch would make the claim genuinely SUPPORTED."
    )
    sp = score_pass(prompt, config)
    return sp.score01 if sp is not None else 0.0


def select_fix(fixes: list[FixProposal], config: Config,
               verdict: ClaimVerdict | None = None,
               transcript_context: str = "") -> FixProposal | None:
    """Score each candidate and PPT-select the best. None if there are no candidates."""
    if not fixes:
        return None
    for fix in fixes:
        fix.rating = score_fix(fix, verdict, transcript_context, config) if verdict else fix.rating
    winner = pivot_tournament([Candidate(id=str(i), rating=f.rating)
                               for i, f in enumerate(fixes)])
    return fixes[int(winner.id)]


def propose_fix(
    verdict: ClaimVerdict, transcript_context: str, config: Config
) -> FixProposal | None:
    """End-to-end for one REFUTED claim: generate → score → PPT-select. Never writes."""
    fixes = generate_fixes(verdict, transcript_context, config)
    return select_fix(fixes, config, verdict=verdict, transcript_context=transcript_context)


def apply_fix(fix: FixProposal, cwd: Path | None = None) -> bool:
    """Apply the winning diff with `git apply`. ONLY called under explicit opt-in."""
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(fix.diff if fix.diff.endswith("\n") else fix.diff + "\n")
        patch_path = f.name
    try:
        result = subprocess.run(
            ["git", "apply", patch_path],
            cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        Path(patch_path).unlink(missing_ok=True)


def _completion(prompt: str, config: Config, *, max_tokens: int, temperature: float) -> str:
    """Provider-agnostic text completion (reuses the checker credentials)."""
    if config.provider == JudgeProvider.OPENAI:
        url = f"{config.api_url}/chat/completions"
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": config.checker_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens,
        }
        with httpx.Client(timeout=config.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    if config.provider == JudgeProvider.GEMINI:
        url = f"{config.api_url}/models/{config.checker_model}:generateContent?key={config.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        with httpx.Client(timeout=config.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    url = f"{config.api_url}/messages"
    headers = {
        "x-api-key": config.api_key, "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.checker_model, "max_tokens": max_tokens,
        "temperature": temperature, "messages": [{"role": "user", "content": prompt}],
    }
    with httpx.Client(timeout=config.timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
```

Note: `select_fix` in the test is called as `select_fix(fixes, cfg)` with pre-set ratings (verdict omitted), and `score_fix` is patched — so the signature keeps `verdict`/`transcript_context` optional and only re-scores when a verdict is supplied. The `propose_fix` test patches `score_fix` directly.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retry.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/receipts/retry.py tests/test_retry.py
git commit -m "feat(retry): self-healing generate→score→PPT→present, opt-in apply

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 12: Wire Phase 2 into the hook (present winning diff; `RECEIPTS_AUTOFIX`)

**Files:**
- Modify: `src/receipts/hook.py`
- Modify: `src/receipts/models.py` (add optional `proposed_fix: str` + `fix_applied: bool` to `VerifiedClaim`)
- Modify: `src/receipts/render.py` (show the proposed diff on refuted rows)
- Test: `tests/test_hook_phase2.py`

**Interfaces:**
- Consumes: `propose_fix`, `apply_fix`, `FixProposal`; `ClaimVerdict`, `FactLabel`, `Config.autofix`.
- Produces: after building verdicts, for each REFUTED verdict call `propose_fix`; store `proposed_fix` (diff) on the `VerifiedClaim`; when `config.autofix` is True call `apply_fix` and set `fix_applied`. Render shows the diff snippet.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hook_phase2.py
from unittest.mock import patch

from receipts.config import Config, JudgeProvider
from receipts.hook import maybe_propose_fixes
from receipts.models import Claim, ClaimType, FactLabel, ScoreMethod, VerifiedClaim, Verdict
from receipts.retry import FixProposal


def _cfg(**kw):
    return Config(provider=JudgeProvider.OPENAI, model="m", api_key="k", **kw)


def _refuted_vc():
    return VerifiedClaim(
        claim=Claim(text="all tests pass", claim_type=ClaimType.TEST_PASS),
        verdict=Verdict.REFUTED, label=FactLabel.REFUTED, score=5.0,
        critique="pytest exit 1", method=ScoreMethod.LOGPROB,
    )


def test_maybe_propose_fixes_attaches_diff_but_does_not_apply_by_default():
    vcs = [_refuted_vc()]
    fix = FixProposal(diff="--- a/x\n+++ b/x\n", rating=0.9, explanation="fix it")
    with patch("receipts.hook.propose_fix", return_value=fix), \
         patch("receipts.hook.apply_fix") as apply_mock:
        maybe_propose_fixes(vcs, "ctx", _cfg(autofix=False))
    assert vcs[0].proposed_fix.startswith("--- a/x")
    assert vcs[0].fix_applied is False
    apply_mock.assert_not_called()


def test_maybe_propose_fixes_applies_when_autofix_enabled():
    vcs = [_refuted_vc()]
    fix = FixProposal(diff="--- a/x\n+++ b/x\n", rating=0.9, explanation="fix it")
    with patch("receipts.hook.propose_fix", return_value=fix), \
         patch("receipts.hook.apply_fix", return_value=True) as apply_mock:
        maybe_propose_fixes(vcs, "ctx", _cfg(autofix=True))
    assert vcs[0].fix_applied is True
    apply_mock.assert_called_once()


def test_maybe_propose_fixes_skips_non_refuted():
    vc = VerifiedClaim(claim=Claim(text="a"), verdict=Verdict.VERIFIED,
                       label=FactLabel.SUPPORTED, score=95.0)
    with patch("receipts.hook.propose_fix") as pf:
        maybe_propose_fixes([vc], "ctx", _cfg(autofix=True))
    pf.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hook_phase2.py -v`
Expected: FAIL — `ImportError: cannot import name 'maybe_propose_fixes'`.

- [ ] **Step 3: Write minimal implementation**

Add optional fields to `VerifiedClaim` in `models.py`:

```python
    proposed_fix: str = Field(default="", description="Phase 2: winning candidate diff (if any)")
    fix_applied: bool = Field(default=False, description="Phase 2: whether the fix was auto-applied")
```

In `hook.py`, add imports and the helper, and call it in `process_transcript` after `verified_claims` is built:

```python
from receipts.retry import FixProposal, apply_fix, propose_fix
from receipts.models import FactLabel
```

```python
def maybe_propose_fixes(
    verified_claims: list[VerifiedClaim], transcript_context: str, config: Config
) -> None:
    """Phase 2: for each REFUTED claim, propose a fix; apply only under RECEIPTS_AUTOFIX."""
    for vc in verified_claims:
        if vc.label != FactLabel.REFUTED:
            continue
        from receipts.verifier import ClaimVerdict  # local import to avoid cycle
        cv = ClaimVerdict(
            claim=vc.claim, label=vc.label, score=vc.score or 0.0,
            confidence=vc.confidence or 0.0, per_criterion=vc.per_criterion,
            critique=vc.critique, method=vc.method, passes=vc.passes, evidence=vc.evidence,
        )
        fix = propose_fix(cv, transcript_context, config)
        if fix is None:
            continue
        vc.proposed_fix = fix.diff
        if config.autofix:
            vc.fix_applied = apply_fix(fix)
```

Call it in `process_transcript` right after `verified_claims = [...]`:

```python
        maybe_propose_fixes(verified_claims, transcript_context, config)
```

In `render.py`, in the per-claim loop, after rendering the critique, show a proposed-fix hint:

```python
            if getattr(vc, "proposed_fix", ""):
                status = "applied" if vc.fix_applied else "proposed (run with RECEIPTS_AUTOFIX=1 to apply)"
                body.append(f"     🔧 fix {status}\n", style="cyan")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hook_phase2.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/receipts/hook.py src/receipts/models.py src/receipts/render.py tests/test_hook_phase2.py
git commit -m "feat(phase2): propose fixes on refuted claims; opt-in RECEIPTS_AUTOFIX apply

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 13: End-to-end integration test

**Files:**
- Create: `tests/fixtures/session_refuted_and_ambiguous.jsonl`
- Create: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `parse_transcript`, `process_transcript`, `Config`, `verify_session`. Mocks only the network boundary (`score_pass`, `generate_critique`, `propose_fix`).

This is the "reproduce it as a real user would" gate from global guidance: a genuine transcript with (a) a refuted "all tests pass" claim contradicted by a `pytest` exit-1, and (b) an ambiguous "improved error handling" claim with no supporting diff.

- [ ] **Step 1: Create the fixture transcript**

Create `tests/fixtures/session_refuted_and_ambiguous.jsonl` — a minimal Claude-Code-style transcript. Match the shape `parse_transcript` expects (inspect `src/receipts/transcript.py` first and mirror its field names). Include:
- a user event with the request text,
- a Bash tool_use running `pytest` and a tool_result with `exit_code: 1` and failure output,
- a final assistant message: `"All tests pass and I improved error handling."`

- [ ] **Step 2: Write the failing E2E test**

```python
# tests/test_end_to_end.py
from pathlib import Path
from unittest.mock import patch

from receipts.config import Config, JudgeProvider
from receipts.models import FactLabel, ScoreMethod
from receipts.scorer import ScorePass
from receipts.transcript import parse_transcript
from receipts.verifier import verify_session

FIXTURE = Path(__file__).parent / "fixtures" / "session_refuted_and_ambiguous.jsonl"


def _cfg():
    return Config(provider=JudgeProvider.OPENAI, model="gpt-x", api_key="k", min_confidence=0.6)


def test_end_to_end_refuted_and_abstained():
    transcript = parse_transcript(FIXTURE)
    from receipts.claims import extract_claims
    claims = extract_claims(transcript.final_message)
    # The ambiguous "improved error handling" claim scores mid with low peakiness → NEI.
    low = ScorePass(score01=0.6, peakiness=0.3, method=ScoreMethod.LOGPROB, letter="M")
    with patch("receipts.verifier.score_pass", return_value=low), \
         patch("receipts.verifier.generate_critique", return_value="no try/except changes found"):
        report = verify_session(claims, transcript,
                                "pytest ... exit 1 ... 2 failed", _cfg())
    labels = {v.claim.claim_type.value: v.label for v in report.verdicts}
    # "all tests pass" is refuted deterministically by the pytest exit-1 fast-path.
    from receipts.models import ClaimType
    test_claim = next(v for v in report.verdicts if v.claim.claim_type == ClaimType.TEST_PASS)
    assert test_claim.label == FactLabel.REFUTED
    assert test_claim.method == ScoreMethod.DETERMINISTIC
    # The fuzzy claim abstains rather than guessing.
    fuzzy = [v for v in report.verdicts if v.claim.claim_type != ClaimType.TEST_PASS]
    assert fuzzy and all(v.label == FactLabel.NOT_ENOUGH_INFO for v in fuzzy)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_end_to_end.py -v`
Expected: FAIL initially (fixture shape or extraction mismatch). Adjust the fixture until `extract_claims` yields both a `TEST_PASS` claim and a fuzzy claim, and `match_evidence` finds the pytest exit-1.

- [ ] **Step 4: Iterate to green**

Run: `python -m pytest tests/test_end_to_end.py -v`
Expected: PASS. If the pytest fast-path doesn't fire, verify the fixture's Bash tool_result carries `exit_code: 1` and `tool_input.command` contains `pytest` (per `evidence.match_evidence`).

- [ ] **Step 5: Full suite + lint, then commit**

Run: `python -m pytest -q && ruff check src/receipts`
Expected: all tests pass; ruff clean (fix any lint before committing).

```bash
git add tests/test_end_to_end.py tests/fixtures/session_refuted_and_ambiguous.jsonl
git commit -m "test(e2e): real-session refuted + abstained verification gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-implementation: Fable 5 calibration checkpoint (human/high-stakes gate)

Not a code task — a review gate before opening the PR. Reserve **Fable 5** to review:
- calibration correctness of the score bands (`SUPPORTED_BAND`/`REFUTED_BAND`) and the abstention threshold on real sessions,
- that `check_independence` correctly refuses to self-audit,
- the meta-honesty guarantee (no forced accusations) holds on ambiguous claims.

Then open the PR against `harish-nair-ai/agentic-receipts` (**PR, not `main`; no PyPI publish**).

---

## Self-Review (against the spec)

**Spec coverage:**
- §8.1 fast-path → Task 6 (`fast_path`). ✅
- §8.2 calibrated scorer / Eq 3.1 / critique → Tasks 3–4. ✅
- §8.3 criteria weak-verifier ensemble → Task 5. ✅
- §8.4 adaptive K + abstention + bands → Task 6 (`verify_claim`, `label_for`, `aggregate_confidence`). ✅
- §8.5 fallback ladder (logprob → sampled → evidence-only) → Task 4 (`score_pass` dispatch) + Task 6 (`_abstain`). ✅
- §8.6 upgraded card → Task 8. ✅
- §8.7 stats → Task 9. ✅
- §9 Phase 2 PPT + self-healing → Tasks 10–12. ✅
- §10 data model → Task 1 (+ Task 12 fix fields). ✅
- §12 config vars → Task 2. ✅
- §13 maker≠checker + local-first → Task 2 (`check_independence`), surfaced in Tasks 6/8; no network except checker API. ✅
- §14 testing (scorer math, labeling/abstention, adaptive K, fast-path, criteria, fallback, PPT, README fix, E2E) → Tasks 3,6,5,10,9,13. ✅
- §11 Honesty Index → intentionally NOT built (data model kept forward-compatible). ✅

**Placeholder scan:** No TBD/TODO; every code step carries concrete code; every test carries real assertions. ✅

**Type consistency:** `ClaimVerdict` fields identical across Tasks 6/7/11/12; `ScorePass` identical across Tasks 3/4/6; `FactLabel`/`ScoreMethod` from Task 1 used consistently; `Candidate` from Task 10 used in Task 11; `Config.checker_model`/`score_passes`/`min_confidence`/`autofix` from Task 2 used in Tasks 4/6/11/12. ✅

**Known integration risk to watch:** Task 13's fixture must match `transcript.py`'s parser field names — inspect that file before writing the fixture (called out in Task 13, Step 1).
