"""Main transcript parsing entrypoint."""

from __future__ import annotations

from pathlib import Path

from receipts.adapters.claude_code import ClaudeCodeAdapter
from receipts.models import ParsedTranscript


def parse_transcript(path: Path, agent: str = "claude-code") -> ParsedTranscript:
    """Parse an agent transcript file.

    Args:
        path: Path to the transcript file.
        agent: Agent type (currently supports 'claude-code').

    Returns:
        ParsedTranscript containing extracted events.

    Raises:
        ValueError: If agent is not supported.
    """
    if agent == "claude-code":
        adapter = ClaudeCodeAdapter()
        return adapter.parse(path)
    
    raise ValueError(f"Unsupported agent: {agent}")


def find_latest_transcript(agent: str = "claude-code") -> Path | None:
    """Find the most recent session transcript for an agent.

    Args:
        agent: Agent type.

    Returns:
        Path to the latest transcript, or None if none found.
    """
    if agent == "claude-code":
        return ClaudeCodeAdapter().find_latest_session()
        
    return None
