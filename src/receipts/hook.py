"""Claude Code hook handler."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from receipts.claims import extract_claims
from receipts.config import Config
from receipts.evidence import match_evidence
from receipts.judge import judge_claim
from receipts.models import Receipt, VerifiedClaim, Verdict
from receipts.render import render_receipt
from receipts.stats import save_receipt
from receipts.transcript import parse_transcript


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


def process_transcript(transcript_path: Path, config: Config) -> int:
    """Process a transcript and return an exit code."""
    try:
        start_time = time.time()
        
        # 1. Parse
        transcript = parse_transcript(transcript_path)
        if not transcript.final_message:
            return 0  # Nothing to verify
            
        # 2. Extract claims
        claims = extract_claims(transcript.final_message)
        if not claims:
            return 0  # No claims made
            
        # 3. Verify claims
        verified_claims: list[VerifiedClaim] = []
        transcript_context = _build_transcript_context(transcript)
        
        for claim in claims:
            # Deterministic pass
            evidence = match_evidence(claim, transcript)
            
            # If we found strong evidence, skip LLM judge
            if any(e.supports_claim for e in evidence):
                verified_claims.append(
                    VerifiedClaim(
                        claim=claim,
                        verdict=Verdict.VERIFIED,
                        evidence=evidence,
                        reasoning="Deterministically verified by transcript events.",
                    )
                )
                continue
                
            # If we found strong refuting evidence, skip LLM judge
            if evidence and all(not e.supports_claim for e in evidence):
                verified_claims.append(
                    VerifiedClaim(
                        claim=claim,
                        verdict=Verdict.REFUTED,
                        evidence=evidence,
                        reasoning="Deterministically refuted by transcript events.",
                    )
                )
                continue
                
            # LLM Judge pass
            judge_res = judge_claim(claim, evidence, transcript_context, config)
            verified_claims.append(
                VerifiedClaim(
                    claim=claim,
                    verdict=judge_res.verdict,
                    evidence=evidence,
                    reasoning=judge_res.reasoning,
                )
            )
            
        # 4. Generate receipt
        duration_ms = int((time.time() - start_time) * 1000)
        receipt = Receipt(
            session_id=transcript.session_id,
            user_request=transcript.user_request[:500], # truncate
            claims=verified_claims,
            judge_model=config.model,
            judge_duration_ms=duration_ms,
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
