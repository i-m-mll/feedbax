"""Shared Markdown rendering helpers for analysis materializers and reports."""

from __future__ import annotations

from collections.abc import Sequence


def render_markdown_note(
    *,
    title: str,
    rows: Sequence[tuple[str, object]],
    narrative: str | None = None,
) -> str:
    """Render the standard human-readable note used by analysis outputs."""
    lines = [f"# {title}", ""]
    if narrative:
        lines.extend([narrative, ""])
    lines.extend(["| Field | Value |", "| --- | --- |"])
    for label, value in rows:
        escaped = str(value).replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {label} | {escaped} |")
    lines.append("")
    return "\n".join(lines)
