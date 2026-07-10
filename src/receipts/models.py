"""Data models for Receipts.

These Pydantic models are the backbone of the verification pipeline:
Claim → Evidence → VerifiedClaim → Receipt
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python 3.10."""
        pass

from pydantic import BaseModel, Field


class ClaimType(StrEnum):
    """Categories of completion claims an agent can make."""

    TEST_PASS = "test_pass"          # "tests pass", "all 8 tests pass"
    FILE_CREATED = "file_created"    # "created src/utils.py"
    FILE_MODIFIED = "file_modified"  # "updated the config", "modified X"
    BUG_FIXED = "bug_fixed"          # "fixed the TypeError", "resolved the issue"
    FEATURE_ADDED = "feature_added"  # "implemented feature X", "added support for Y"
    COMMAND_RUN = "command_run"      # "ran the migration", "executed the build"
    REFACTORED = "refactored"        # "refactored for clarity", "cleaned up"
    GENERIC = "generic"              # anything else


class Verdict(StrEnum):
    """Outcome of verifying a single claim."""

    VERIFIED = "verified"        # Evidence confirms the claim
    UNVERIFIED = "unverified"    # No evidence found to confirm or deny
    REFUTED = "refuted"          # Evidence contradicts the claim
    SKIPPED = "skipped"          # Claim too vague to verify


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


class EvidenceSource(StrEnum):
    """Where the evidence came from."""

    EXIT_CODE = "exit_code"        # Command exit code (0 = success)
    TOOL_OUTPUT = "tool_output"    # Output from a tool call
    FILE_WRITE = "file_write"      # A file was written/edited
    DIFF = "diff"                  # Content of a diff/edit
    COMMAND_LOG = "command_log"    # Command was found in transcript
    JUDGE = "judge"                # LLM judge verdict


class Claim(BaseModel):
    """A single completion claim extracted from an agent's output.

    Example: "All 8 tests pass" → Claim(text="All 8 tests pass", claim_type=ClaimType.TEST_PASS)
    """

    text: str = Field(description="The raw claim text from the agent's message")
    claim_type: ClaimType = Field(
        default=ClaimType.GENERIC,
        description="Categorized type of the claim",
    )


class Evidence(BaseModel):
    """A piece of evidence supporting or refuting a claim.

    Evidence is gathered from the transcript — tool calls, exit codes, file edits, etc.
    """

    source: EvidenceSource = Field(description="Where this evidence came from")
    content: str = Field(description="The evidence text (exit code, output snippet, etc.)")
    supports_claim: bool = Field(description="Whether this evidence supports the claim")


class VerifiedClaim(BaseModel):
    """A claim with its evidence and final verdict."""

    claim: Claim = Field(description="The original claim")
    verdict: Verdict = Field(description="Verification outcome")
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Evidence gathered for this claim",
    )
    reasoning: str = Field(
        default="",
        description="Human-readable explanation of the verdict",
    )

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
    proposed_fix: str = Field(default="", description="Phase 2: winning candidate diff (if any)")
    fix_applied: bool = Field(
        default=False, description="Phase 2: whether the fix was auto-applied"
    )


class Receipt(BaseModel):
    """The full receipt for an agent session — the final output of the pipeline.

    A Receipt is the proof-of-work for a coding agent session. It lists every
    claim the agent made and whether that claim was independently verified.
    """

    receipt_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Unique receipt identifier",
    )
    session_id: str = Field(description="Agent session identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the receipt was generated",
    )
    agent: str = Field(default="claude-code", description="Which agent produced the session")
    user_request: str = Field(default="", description="The original user request/task")
    claims: list[VerifiedClaim] = Field(
        default_factory=list,
        description="All verified claims in the session",
    )
    judge_model: str = Field(default="", description="Model used for the independent judge")
    judge_duration_ms: int = Field(
        default=0, description="Time spent on judge calls in milliseconds"
    )
    checker_independent: bool = Field(
        default=True,
        description="Whether checker model differed from the audited agent (maker≠checker)",
    )

    @property
    def verified_count(self) -> int:
        """Number of claims that were verified."""
        return sum(1 for c in self.claims if c.verdict == Verdict.VERIFIED)

    @property
    def unverified_count(self) -> int:
        """Number of claims that were unverified."""
        return sum(1 for c in self.claims if c.verdict == Verdict.UNVERIFIED)

    @property
    def refuted_count(self) -> int:
        """Number of claims that were refuted."""
        return sum(1 for c in self.claims if c.verdict == Verdict.REFUTED)

    @property
    def total_claims(self) -> int:
        """Total number of claims (excluding skipped)."""
        return sum(1 for c in self.claims if c.verdict != Verdict.SKIPPED)

    @property
    def score(self) -> float:
        """Verification score: fraction of claims verified (0.0 to 1.0)."""
        total = self.total_claims
        if total == 0:
            return 1.0
        return self.verified_count / total

    @property
    def has_unverified(self) -> bool:
        """Whether any claims are unverified or refuted."""
        return self.unverified_count > 0 or self.refuted_count > 0

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


class TranscriptEvent(BaseModel):
    """A single event from an agent's session transcript."""

    event_type: str = Field(description="Event type (user, assistant, tool_use, tool_result)")
    timestamp: str = Field(default="", description="ISO timestamp")
    content: str = Field(default="", description="Text content of the event")

    # Tool-specific fields
    tool_name: str | None = Field(default=None, description="Tool name (for tool events)")
    tool_input: dict | None = Field(default=None, description="Tool input arguments")
    exit_code: int | None = Field(default=None, description="Command exit code")
    file_path: str | None = Field(default=None, description="File path (for file operations)")


class ParsedTranscript(BaseModel):
    """A parsed agent session transcript — the input to the verification pipeline."""

    session_id: str = Field(description="Session identifier")
    user_request: str = Field(default="", description="The original user request")
    final_message: str = Field(default="", description="The agent's final assistant message")
    events: list[TranscriptEvent] = Field(
        default_factory=list, description="All events in chronological order"
    )

    @property
    def tool_events(self) -> list[TranscriptEvent]:
        """All tool_use and tool_result events."""
        return [e for e in self.events if e.event_type in ("tool_use", "tool_result")]

    @property
    def bash_results(self) -> list[TranscriptEvent]:
        """All Bash tool results (command outputs)."""
        return [
            e
            for e in self.events
            if e.event_type == "tool_result" and e.tool_name in ("Bash", "bash", "terminal")
        ]

    @property
    def file_writes(self) -> list[TranscriptEvent]:
        """All file write/edit events."""
        return [
            e
            for e in self.events
            if e.event_type == "tool_use"
            and e.tool_name in ("Write", "write_to_file", "Edit", "Replace", "MultiEdit",
                                "replace_file_content", "multi_replace_file_content",
                                "write", "edit", "create")
        ]
