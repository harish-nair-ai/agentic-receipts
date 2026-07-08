"""Claim extraction logic."""

from __future__ import annotations

import re

from receipts.models import Claim, ClaimType


def extract_claims(final_message: str) -> list[Claim]:
    """Extract completion claims from the agent's final message.

    Uses a two-phase extraction:
    1. Regex/heuristic pass for common patterns.
    2. Fallback to sentence splitting if heuristics find nothing.

    Args:
        final_message: The agent's final message.

    Returns:
        List of extracted claims.
    """
    claims: list[Claim] = []
    
    # Clean markdown
    clean_msg = _strip_markdown(final_message)
    
    # Phase 1: Regex heuristics
    test_match = re.search(r"(tests? pass|test suite passes|all \d+ tests? pass)", clean_msg, re.IGNORECASE)
    if test_match:
        claims.append(Claim(text=test_match.group(0), claim_type=ClaimType.TEST_PASS))
        
    file_create_match = re.search(r"(created|added) (file )?([a-zA-Z0-9_./-]+)", clean_msg, re.IGNORECASE)
    if file_create_match:
        claims.append(Claim(text=file_create_match.group(0), claim_type=ClaimType.FILE_CREATED))
        
    file_mod_match = re.search(r"(modified|updated|changed) (file )?([a-zA-Z0-9_./-]+)", clean_msg, re.IGNORECASE)
    if file_mod_match:
        claims.append(Claim(text=file_mod_match.group(0), claim_type=ClaimType.FILE_MODIFIED))
        
    bug_match = re.search(r"(fixed|resolved) (the |a )?(bug|issue|error|TypeError|ValueError|Exception)", clean_msg, re.IGNORECASE)
    if bug_match:
        claims.append(Claim(text=bug_match.group(0), claim_type=ClaimType.BUG_FIXED))
        
    feature_match = re.search(r"(implemented|added support for|added feature) ([a-zA-Z0-9_ -]+)", clean_msg, re.IGNORECASE)
    if feature_match:
        claims.append(Claim(text=feature_match.group(0), claim_type=ClaimType.FEATURE_ADDED))
        
    cmd_match = re.search(r"(ran|executed) (the )?(command|migration|build)", clean_msg, re.IGNORECASE)
    if cmd_match:
        claims.append(Claim(text=cmd_match.group(0), claim_type=ClaimType.COMMAND_RUN))
        
    refactor_match = re.search(r"(refactored|cleaned up) ([a-zA-Z0-9_ -]+)", clean_msg, re.IGNORECASE)
    if refactor_match:
        claims.append(Claim(text=refactor_match.group(0), claim_type=ClaimType.REFACTORED))
        
    # Phase 2: Fallback (if no explicit claims found, look for "done" indicators)
    if not claims:
        done_match = re.search(r"(I have finished|I'm done|task is complete|all done)", clean_msg, re.IGNORECASE)
        if done_match:
            claims.append(Claim(text="Task completed", claim_type=ClaimType.GENERIC))

    return claims


def _strip_markdown(text: str) -> str:
    """Basic markdown stripping for easier regex matching."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # Remove code blocks
    text = re.sub(r"`(.*?)`", r"\1", text)                   # Remove inline code
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)             # Remove bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)                 # Remove italic
    return text
