"""Deterministic evidence matching."""

from __future__ import annotations

import re

from receipts.models import Claim, ClaimType, Evidence, EvidenceSource, ParsedTranscript


def match_evidence(claim: Claim, transcript: ParsedTranscript) -> list[Evidence]:
    """Gather deterministic evidence from the transcript for a given claim.

    Args:
        claim: The extracted claim.
        transcript: The parsed session transcript.

    Returns:
        List of evidence objects. Empty list means no deterministic evidence found
        (must fallback to LLM judge).
    """
    evidence: list[Evidence] = []
    
    if claim.claim_type == ClaimType.TEST_PASS:
        # Find the last test command
        for event in reversed(transcript.bash_results):
            if event.tool_input and isinstance(event.tool_input, dict):
                cmd = event.tool_input.get("command", "")
                if any(x in cmd for x in ["pytest", "npm test", "cargo test", "go test", "jest"]):
                    # We found a test run
                    if event.exit_code == 0:
                        evidence.append(
                            Evidence(
                                source=EvidenceSource.EXIT_CODE,
                                content=f"Command '{cmd}' exited with code 0",
                                supports_claim=True,
                            )
                        )
                    elif event.exit_code is not None:
                        evidence.append(
                            Evidence(
                                source=EvidenceSource.EXIT_CODE,
                                content=f"Command '{cmd}' exited with code {event.exit_code}",
                                supports_claim=False,
                            )
                        )
                    break

    elif claim.claim_type == ClaimType.FILE_CREATED:
        # Extract filename from claim
        match = re.search(r"(created|added) (file )?([a-zA-Z0-9_./-]+)", claim.text, re.IGNORECASE)
        if match:
            filename = match.group(3)
            for event in transcript.file_writes:
                if event.file_path and filename in event.file_path:
                    # Found the file creation
                    evidence.append(
                        Evidence(
                            source=EvidenceSource.FILE_WRITE,
                            content=f"Found tool call writing to {event.file_path}",
                            supports_claim=True,
                        )
                    )
                    break
            if not evidence:
                 evidence.append(
                    Evidence(
                        source=EvidenceSource.FILE_WRITE,
                        content=f"No file writes found matching '{filename}'",
                        supports_claim=False,
                    )
                 )

    elif claim.claim_type == ClaimType.FILE_MODIFIED:
        match = re.search(r"(modified|updated|changed) (file )?([a-zA-Z0-9_./-]+)", claim.text, re.IGNORECASE)
        if match:
            filename = match.group(3)
            for event in transcript.file_writes:
                if event.file_path and filename in event.file_path:
                    evidence.append(
                        Evidence(
                            source=EvidenceSource.FILE_WRITE,
                            content=f"Found tool call editing {event.file_path}",
                            supports_claim=True,
                        )
                    )
                    break
                    
    elif claim.claim_type == ClaimType.COMMAND_RUN:
        for event in transcript.bash_results:
             if event.tool_input and isinstance(event.tool_input, dict):
                 cmd = event.tool_input.get("command", "")
                 if cmd:
                     evidence.append(
                         Evidence(
                             source=EvidenceSource.COMMAND_LOG,
                             content=f"Ran command: {cmd}",
                             supports_claim=True,
                         )
                     )
                     break
                     
    # For GENERIC, BUG_FIXED, FEATURE_ADDED, REFACTORED we often need the LLM judge
    
    return evidence
