"""Abstract base class for transcript adapters."""

from __future__ import annotations

import abc
from pathlib import Path

from receipts.models import ParsedTranscript


class TranscriptAdapter(abc.ABC):
    """Base class for parsing agent session transcripts."""

    @abc.abstractmethod
    def parse(self, transcript_path: Path) -> ParsedTranscript:
        """Parse a transcript file into a ParsedTranscript model.

        Args:
            transcript_path: Path to the transcript file.

        Returns:
            A ParsedTranscript containing extracted events and context.
        """
        pass

    @abc.abstractmethod
    def find_latest_session(self) -> Path | None:
        """Find the path to the most recent session transcript.

        Returns:
            Path to the transcript file, or None if no sessions found.
        """
        pass
