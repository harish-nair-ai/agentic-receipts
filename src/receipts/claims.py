"""Claim extraction logic."""

from __future__ import annotations

import re

from receipts.models import Claim, ClaimType


# Patterns mapped to claim types. Each pattern should have meaningful capture groups.
_CLAIM_PATTERNS: list[tuple[str, ClaimType]] = [
    (r"((?:all\s+\d+\s+)?tests?\s+pass(?:es|ed)?|test\s+suite\s+pass(?:es|ed)?)", ClaimType.TEST_PASS),
    (r"(created|added)\s+(?:file\s+)?([a-zA-Z0-9_./-]+)", ClaimType.FILE_CREATED),
    (r"(modified|updated|changed)\s+(?:file\s+)?([a-zA-Z0-9_./-]+)", ClaimType.FILE_MODIFIED),
    (r"(fixed|resolved)\s+(?:the\s+|a\s+)?(bug|issue|error|TypeError|ValueError|KeyError|Exception|crash)", ClaimType.BUG_FIXED),
    (r"(implemented|added\s+support\s+for|added\s+feature)\s+([a-zA-Z0-9_ -]+)", ClaimType.FEATURE_ADDED),
    (r"(ran|executed)\s+(?:the\s+)?(command|migration|build|script)", ClaimType.COMMAND_RUN),
    (r"(refactored|cleaned\s+up)\s+([a-zA-Z0-9_ -]+)", ClaimType.REFACTORED),
]


def extract_claims(final_message: str) -> list[Claim]:
    """Extract completion claims from the agent's final message.

    Uses a two-phase extraction:
    1. Regex/heuristic pass for common patterns (finditer for multiple matches).
    2. Fallback to "done" indicators if heuristics find nothing.

    Args:
        final_message: The agent's final message.

    Returns:
        List of extracted claims.
    """
    claims: list[Claim] = []
    seen_texts: set[str] = set()

    # Clean markdown
    clean_msg = _strip_markdown(final_message)

    # Phase 1: Regex heuristics (finditer catches ALL occurrences)
    for pattern, claim_type in _CLAIM_PATTERNS:
        for match in re.finditer(pattern, clean_msg, re.IGNORECASE):
            text = match.group(0).strip()
            if text not in seen_texts:
                seen_texts.add(text)
                claims.append(Claim(text=text, claim_type=claim_type))

    # Phase 2: Fallback (if no explicit claims found, look for "done" indicators)
    if not claims:
        done_match = re.search(
            r"(I have finished|I'm done|task is complete|all done|completed successfully)",
            clean_msg,
            re.IGNORECASE,
        )
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
