"""CLI entrypoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from receipts.config import Config, ConfigError
from receipts.hook import handle_hook, process_transcript
from receipts.stats import load_receipts, render_stats
from receipts.transcript import find_latest_transcript


@click.group()
def main() -> None:
    """Receipts — The verified-done layer for AI coding agents.

    Your agent says 'Done!' — Receipts checks the work.
    """
    pass


@main.command()
@click.option("--hook", is_flag=True, help="Run in hook mode (reads event JSON from stdin)")
@click.argument("transcript_path", type=click.Path(exists=True, path_type=Path), required=False)
def verify(hook: bool, transcript_path: Path | None) -> None:
    """Verify an agent session.
    
    If TRANSCRIPT_PATH is omitted, verifies the most recent session.
    """
    try:
        config = Config.from_env()
    except ConfigError as e:
        click.secho(f"Configuration Error: {e}", fg="red", err=True)
        sys.exit(1)
        
    if hook:
        sys.exit(handle_hook(config))
        
    if not transcript_path:
        transcript_path = find_latest_transcript()
        if not transcript_path:
            click.secho("No recent transcript found.", fg="red", err=True)
            sys.exit(1)
            
    click.secho(f"Verifying session: {transcript_path.stem}...", fg="dim")
    sys.exit(process_transcript(transcript_path, config))


@main.command()
@click.option("--days", type=int, default=7, help="Number of days to analyze")
def stats(days: int) -> None:
    """Show verification stats for the last N days."""
    try:
        config = Config.from_env()
    except ConfigError:
        # Stats doesn't technically need an API key to read local files,
        # so we'll build a dummy config if env is missing
        config = Config.model_construct(
            provider="dummy", model="dummy", api_key="", receipts_dir=Path.home() / ".receipts"
        )
        
    receipts = load_receipts(config, days)
    render_stats(receipts, days)


@main.command()
def install() -> None:
    """Install the Claude Code Stop hook."""
    settings_dir = Path.home() / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = settings_dir / "settings.json"
    
    config_data = {}
    if settings_path.exists():
        try:
            with settings_path.open("r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            click.secho(f"Error reading existing settings: {e}", fg="red", err=True)
            sys.exit(1)
            
    hooks = config_data.setdefault("hooks", {})
    stop_hooks = hooks.setdefault("Stop", [])
    
    # Check if already installed
    for hook_def in stop_hooks:
        for inner_hook in hook_def.get("hooks", []):
            if inner_hook.get("command") == "receipts verify --hook":
                click.secho("Receipts hook is already installed in ~/.claude/settings.json", fg="green")
                return
                
    # Install
    stop_hooks.append({
        "matcher": "",
        "hooks": [
            {
                "type": "command",
                "command": "receipts verify --hook",
                "timeout": 30
            }
        ]
    })
    
    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
        
    click.secho("✅ Successfully installed Receipts Stop hook for Claude Code!", fg="green")
    click.secho("The hook will run automatically when Claude finishes a task.", fg="dim")


if __name__ == "__main__":
    main()
