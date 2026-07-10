"""Terminal renderer using Rich."""

from __future__ import annotations

import sys
from typing import TextIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from receipts.models import FactLabel, Receipt, ScoreMethod

_GLYPH = {
    FactLabel.SUPPORTED: ("✅", "green"),
    FactLabel.REFUTED: ("❌", "red"),
    FactLabel.NOT_ENOUGH_INFO: ("❔", "yellow"),
}
_METHOD_TAG = {
    ScoreMethod.LOGPROB: "calibrated",
    ScoreMethod.SAMPLED: "sampled",
    ScoreMethod.DETERMINISTIC: "deterministic",
    ScoreMethod.NONE: "evidence-only",
}


def _bar(value01: float, width: int = 10) -> str:
    filled = max(0, min(width, round(value01 * width)))
    return "█" * filled + "░" * (width - filled)


def render_receipt(receipt: Receipt, file: TextIO = sys.stderr) -> None:
    """Render a calibrated three-way receipt card.

    Args:
        receipt: The generated Receipt.
        file: Output stream (default stderr so it doesn't break hook stdout).
    """
    console = Console(file=file)
    vds = receipt.verified_done_score
    score_color = "green" if vds >= 85 else "yellow" if vds >= 45 else "red"

    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        Text("🧾 RECEIPT", style="bold white"),
        Text(f"Verified-Done Score: {vds:.0f}/100", style=f"bold {score_color}"),
    )
    counts = Text(
        f"✅ {receipt.supported_count} supported   "
        f"❌ {receipt.refuted_count} refuted   "
        f"❔ {receipt.nei_count} not-enough-info",
        style="dim",
    )

    body = Text()
    if not receipt.claims:
        body.append("\nNo completion claims found.", style="dim")
    else:
        body.append("\n")
        for vc in receipt.claims:
            label = vc.label or FactLabel.NOT_ENOUGH_INFO
            glyph, color = _GLYPH[label]
            score_str = f"{vc.score:.0f}" if vc.score is not None else "--"
            tag = _METHOD_TAG.get(vc.method, "") if vc.method else ""
            body.append(f"  {glyph} {vc.claim.text}  ", style=color)
            body.append(f"[{score_str}/100 · {tag}]\n", style="dim")
            if vc.critique:
                body.append(f"     └─ {vc.critique}\n", style="dim")
            if getattr(vc, "proposed_fix", ""):
                status = (
                    "applied" if vc.fix_applied
                    else "proposed (run with RECEIPTS_AUTOFIX=1 to apply)"
                )
                body.append(f"     🔧 fix {status}\n", style="cyan")
            for name, cscore in (vc.per_criterion or {}).items():
                body.append(
                    f"        {name:<14} {_bar(cscore / 100.0)} {cscore:.0f}\n", style="dim"
                )

    if not receipt.checker_independent:
        body.append(
            "\n  ⚠️  checker shares the audited agent's provider — independence not guaranteed\n",
            style="yellow",
        )

    footer = Table.grid(expand=True)
    footer.add_column()
    footer.add_column()
    footer.add_row(
        Text(f"Session: {receipt.session_id} │ Agent: {receipt.agent}", style="dim"),
        Text(
            f"Checker: {receipt.judge_model} │ {receipt.judge_duration_ms / 1000:.1f}s",
            style="dim", justify="right",
        ),
    )

    panel_group = Table.grid(expand=True)
    panel_group.add_column()
    panel_group.add_row(header)
    panel_group.add_row(counts)
    panel_group.add_row(body)
    panel_group.add_row(Text("─" * 50, style="dim"))
    panel_group.add_row(footer)

    console.print(Panel(panel_group, expand=False, border_style=score_color, padding=(1, 2)))


def render_receipt_json(receipt: Receipt) -> str:
    """Return receipt as a JSON string."""
    return receipt.model_dump_json(indent=2)
