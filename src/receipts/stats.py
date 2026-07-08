"""Stats tracking and aggregation."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from receipts.config import Config
from receipts.models import Receipt, Verdict


def save_receipt(receipt: Receipt, config: Config) -> Path:
    """Save a receipt to the user's history."""
    sessions_dir = config.receipts_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = sessions_dir / f"{receipt.timestamp.strftime('%Y%m%d_%H%M%S')}_{receipt.receipt_id}.json"
    with file_path.open("w", encoding="utf-8") as f:
        f.write(receipt.model_dump_json(indent=2))
        
    return file_path


def load_receipts(config: Config, days: int = 7) -> list[Receipt]:
    """Load receipts from the last N days."""
    sessions_dir = config.receipts_dir / "sessions"
    if not sessions_dir.exists():
        return []
        
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    receipts = []
    
    for path in sessions_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                receipt = Receipt.model_validate(data)
                if receipt.timestamp >= cutoff:
                    receipts.append(receipt)
        except Exception:
            continue
            
    # Sort newest first
    receipts.sort(key=lambda r: r.timestamp, reverse=True)
    return receipts


def render_stats(receipts: list[Receipt], days: int = 7, file: TextIO = sys.stdout) -> None:
    """Render aggregate stats as a Rich panel."""
    console = Console(file=file)
    
    if not receipts:
        console.print(Panel(f"No receipts found in the last {days} days.", style="dim"))
        return
        
    total_sessions = len(receipts)
    perfect_sessions = sum(1 for r in receipts if not r.has_unverified and r.total_claims > 0)
    
    total_claims = sum(r.total_claims for r in receipts)
    verified_claims = sum(r.verified_count for r in receipts)
    unverified_claims = sum(r.unverified_count for r in receipts)
    refuted_claims = sum(r.refuted_count for r in receipts)
    
    verification_rate = (verified_claims / total_claims * 100) if total_claims > 0 else 0
    
    # Analyze unverified claims
    unverified_types = Counter()
    for r in receipts:
        for vc in r.claims:
            if vc.verdict in (Verdict.UNVERIFIED, Verdict.REFUTED):
                unverified_types[vc.claim.claim_type.value] += 1
                
    # Build the tree
    tree = Tree(f"🧾 Your agent (last {days} days):")
    tree.add(f"{total_sessions} sessions audited ({perfect_sessions} verified perfectly)")
    tree.add(f"{total_claims} total completion claims")
    
    bad_claims = unverified_claims + refuted_claims
    bad_node = tree.add(
        Text(f"{bad_claims} unverified/refuted claims caught", style="yellow" if bad_claims > 0 else "green")
    )
    
    if unverified_types:
        top_types = unverified_types.most_common(3)
        types_node = bad_node.add("Top unverified claim types:")
        for claim_type, count in top_types:
            types_node.add(f"'{claim_type}' ({count}×)")
            
    rate_color = "green" if verification_rate > 90 else "yellow" if verification_rate > 70 else "red"
    tree.add(Text(f"{verification_rate:.1f}% verification rate", style=rate_color))
    
    panel = Panel(tree, expand=False, padding=(1, 2))
    console.print(panel)
