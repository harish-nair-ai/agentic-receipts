"""Adapter for Claude Code JSONL transcripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from receipts.adapters.base import TranscriptAdapter
from receipts.models import ParsedTranscript, TranscriptEvent


class ClaudeCodeAdapter(TranscriptAdapter):
    """Parses Claude Code JSONL transcripts."""

    def __init__(self, projects_dir: Path | None = None) -> None:
        """Initialize the adapter.

        Args:
            projects_dir: Override default ~/.claude/projects/ directory.
        """
        self.projects_dir = projects_dir or Path.home() / ".claude" / "projects"

    def parse(self, transcript_path: Path) -> ParsedTranscript:
        """Parse a Claude Code JSONL transcript."""
        events: list[TranscriptEvent] = []
        user_request = ""
        final_message = ""
        session_id = transcript_path.stem

        if not transcript_path.exists():
            return ParsedTranscript(session_id=session_id)

        # Stream line by line to handle large transcripts
        with transcript_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = data.get("type")
                if not event_type:
                    continue

                content = self._extract_content(data.get("content"))
                timestamp = data.get("timestamp", "")
                
                # Extract user request (first user message)
                if event_type == "user" and not user_request:
                    user_request = content

                # Track final assistant message
                if event_type == "assistant" and content:
                    final_message = content

                tool_name = None
                tool_input = None
                exit_code = None
                file_path = None

                if event_type == "tool_use":
                    tool_name = data.get("name")
                    tool_input = data.get("input")
                    if tool_input and isinstance(tool_input, dict):
                        file_path = tool_input.get("file_path") or tool_input.get("path")
                
                elif event_type == "tool_result":
                    # Sometimes Claude Code nests these
                    if isinstance(content, list) and len(content) > 0:
                        first_content = content[0]
                        if isinstance(first_content, dict):
                             # Look for exit codes in output if available or infer from is_error
                             if data.get("is_error"):
                                 exit_code = 1
                             else:
                                 exit_code = 0

                event = TranscriptEvent(
                    event_type=event_type,
                    timestamp=timestamp,
                    content=content if isinstance(content, str) else json.dumps(content),
                    tool_name=tool_name,
                    tool_input=tool_input,
                    exit_code=exit_code,
                    file_path=file_path,
                )
                events.append(event)

        return ParsedTranscript(
            session_id=session_id,
            user_request=user_request,
            final_message=final_message,
            events=events,
        )

    def find_latest_session(self) -> Path | None:
        """Find the most recently modified .jsonl file in ~/.claude/projects/."""
        if not self.projects_dir.exists():
            return None

        jsonl_files: list[Path] = []
        for project_dir in self.projects_dir.iterdir():
            if project_dir.is_dir():
                jsonl_files.extend(project_dir.glob("*.jsonl"))

        if not jsonl_files:
            return None

        # Sort by modification time, descending
        latest_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
        return latest_file
        
    def _extract_content(self, content: Any) -> str:
        """Extract text content from Claude Code's block format."""
        if isinstance(content, str):
            return content
            
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and "text" in block:
                        texts.append(block["text"])
            return "\n".join(texts)
            
        return ""
