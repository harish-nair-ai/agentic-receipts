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
        'Respond as JSON: {"diff": "<unified diff>", "explanation": "<one line>"}.\n\n'
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


def select_fix(
    fixes: list[FixProposal],
    config: Config,
    verdict: ClaimVerdict | None = None,
    transcript_context: str = "",
) -> FixProposal | None:
    """Score each candidate and PPT-select the best. None if there are no candidates."""
    if not fixes:
        return None
    for fix in fixes:
        if verdict is not None:
            fix.rating = score_fix(fix, verdict, transcript_context, config)
    winner = pivot_tournament(
        [Candidate(id=str(i), rating=f.rating) for i, f in enumerate(fixes)]
    )
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
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
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
        url = (
            f"{config.api_url}/models/{config.checker_model}"
            f":generateContent?key={config.api_key}"
        )
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
