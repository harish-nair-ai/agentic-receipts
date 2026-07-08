"""Terminal renderer using Rich."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from receipts.models import Receipt, Verdict


def render_receipt(receipt: Receipt, file: TextIO = sys.stderr) -> None:
    """Render a receipt as a beautiful terminal card.

    Args:
        receipt: The generated Receipt.
        file: Output stream (default stderr so it doesn't break hook stdout).
    """
    console = Console(file=file)
    
    score_color = "green" if receipt.score == 1.0 else "yellow" if receipt.score > 0.5 else "red"
    score_text = Text(f"Score: {receipt.verified_count}/{receipt.total_claims}", style=f"bold {score_color}")
    
    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(Text("🧾 RECEIPT", style="bold white"), score_text)
    
    body = Text()
    
    if not receipt.claims:
        body.append("\nNo completion claims found.", style="dim")
    else:
        body.append("\n")
        for vc in receipt.claims:
            if vc.verdict == Verdict.VERIFIED:
                body.append(f"  ✅ {vc.claim.text}\n", style="green")
            elif vc.verdict == Verdict.UNVERIFIED:
                body.append(f"  ⚠️  {vc.claim.text} — unverified\n", style="yellow")
                body.append(f"     └─ {vc.reasoning}\n", style="dim")
            elif vc.verdict == Verdict.REFUTED:
                body.append(f"  ❌ {vc.claim.text} — refuted\n", style="red")
                body.append(f"     └─ {vc.reasoning}\n", style="dim")
            elif vc.verdict == Verdict.SKIPPED:
                body.append(f"  ➖ {vc.claim.text} — skipped\n", style="dim")
                body.append(f"     └─ {vc.reasoning}\n", style="dim")
                
    footer = Table.grid(expand=True)
    footer.add_column()
    footer.add_column()
    footer.add_row(
        Text(f"Session: {receipt.session_id} │ Agent: {receipt.agent}", style="dim"),
        Text(f"Judge: {receipt.judge_model} │ {receipt.judge_duration_ms/1000:.1f}s", style="dim", justify="right")
    )
    
    panel_group = Table.grid(expand=True)
    panel_group.add_column()
    panel_group.add_row(header)
    panel_group.add_row(body)
    panel_group.add_row(Text("─" * 50, style="dim"))
    panel_group.add_row(footer)
    
    panel = Panel(
        panel_group,
        expand=False,
        border_style=score_color,
        padding=(1, 2)
    )
    
    console.print(panel)


def render_receipt_json(receipt: Receipt) -> str:
    """Return receipt as a JSON string."""
    return receipt.model_dump_json(indent=2)
