"""Claude Code hook handler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from receipts.claims import extract_claims
from receipts.config import Config
from receipts.models import FactLabel, Receipt, VerifiedClaim, verdict_for_label
from receipts.render import render_receipt
from receipts.retry import apply_fix, propose_fix
from receipts.stats import save_receipt
from receipts.transcript import parse_transcript
from receipts.verifier import ClaimVerdict, verify_session


def handle_hook(config: Config) -> int:
    """Handle Claude Code Stop hook via stdin.

    Returns:
        Exit code (0 = success, 2 = block).
    """
    # Read event from stdin
    try:
        input_data = sys.stdin.read()
        if not input_data:
            # Not running in a hook context, or empty input
            return 0
        event = json.loads(input_data)
    except Exception:
        # Fail silently if not valid JSON (e.g. manual CLI run without args)
        return 0

    transcript_path_str = event.get("transcript_path")
    if not transcript_path_str:
        return 0

    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        return 0

    return process_transcript(transcript_path, config)


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


def maybe_propose_fixes(
    verified_claims: list[VerifiedClaim], transcript_context: str, config: Config
) -> None:
    """Phase 2: for each REFUTED claim, propose a fix; apply only under RECEIPTS_AUTOFIX."""
    for vc in verified_claims:
        if vc.label != FactLabel.REFUTED:
            continue
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


def process_transcript(transcript_path: Path, config: Config) -> int:
    """Process a transcript and return an exit code."""
    try:
        # 1. Parse
        transcript = parse_transcript(transcript_path)
        if not transcript.final_message:
            return 0  # Nothing to verify

        # 2. Extract claims
        claims = extract_claims(transcript.final_message)
        if not claims:
            return 0  # No claims made

        # 3. Verify claims (calibrated three-way verifier)
        transcript_context = _build_transcript_context(transcript)
        report = verify_session(claims, transcript, transcript_context, config)
        verified_claims = [verdict_to_verified_claim(cv) for cv in report.verdicts]
        maybe_propose_fixes(verified_claims, transcript_context, config)

        # 4. Generate receipt
        receipt = Receipt(
            session_id=transcript.session_id,
            user_request=transcript.user_request[:500],  # truncate
            claims=verified_claims,
            judge_model=report.checker_model,
            judge_duration_ms=report.duration_ms,
            checker_independent=report.checker_independent,
        )

        # 5. Save & Render
        save_receipt(receipt, config)
        render_receipt(receipt)

        # 6. Decide exit code
        if config.block_on_unverified and receipt.has_unverified:
            return 2  # Block session end

        return 0

    except Exception as e:
        # Don't break the agent if verification fails
        print(f"Receipts error: {e}", file=sys.stderr)
        return 0


def _build_transcript_context(transcript) -> str:
    """Build a concise text representation of tool events for the judge."""
    lines = []
    for event in transcript.tool_events:
        if event.event_type == "tool_use":
            lines.append(f"Tool Run: {event.tool_name}")
            if event.tool_input:
                # Truncate very long inputs
                input_str = json.dumps(event.tool_input)
                if len(input_str) > 500:
                    input_str = input_str[:500] + "... [truncated]"
                lines.append(f"Input: {input_str}")
        else:
            exit_code = f" (Exit {event.exit_code})" if event.exit_code is not None else ""
            lines.append(f"Result{exit_code}:")
            # Truncate very long outputs
            content = event.content
            if len(content) > 1000:
                content = content[:1000] + "... [truncated]"
            lines.append(content)
            lines.append("---")

    return "\n".join(lines)
