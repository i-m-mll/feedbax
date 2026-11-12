"""Render ordered reports using references to already saved Plotly objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
import json
import re

from markdown_it import MarkdownIt

from feedbax.contracts.figure_document import PlotlyFigureProduct
from feedbax.contracts.report_document import ReportDocument, ReportText
from feedbax.contracts.request_refs import ContentRef


_HTML_NAME = "report.html"
_PLOTLY_SCRIPT = "plotly.min.js"
_DRAW_FIGURES = """
<script>
  const sources = JSON.parse(document.getElementById("report-figure-sources").textContent);
  const loaded = new Map();
  for (const target of document.querySelectorAll("[data-report-source]")) {
    const source = sources[Number(target.dataset.reportSource)];
    if (!loaded.has(source)) {
      loaded.set(source, fetch(source).then(response => {
        if (!response.ok) throw new Error("Figure returned HTTP " + response.status);
        return response.json();
      }));
    }
    loaded.get(source).then(spec => {
      const figure = structuredClone(spec);
      if (!Array.isArray(figure.data)) throw new Error("Saved figure has no Plotly data.");
      return Plotly.newPlot(target, figure);
    }).catch(error => {
      target.setAttribute("role", "alert");
      target.textContent = "Could not load the saved figure.";
      console.error(error);
    });
  }
</script>
""".strip()


@dataclass(frozen=True)
class ReportRender:
    """Companion report.html and Markdown, with relative paths for saved figure objects.

    Serve the HTML with the referenced assets and one sibling plotly.min.js.
    """

    html: str
    markdown: str
    figures: dict[str, ContentRef]


def _literal_markdown(value: str) -> str:
    return re.sub(r"([\\\x60*_~\[\]()#<>&])", r"\\\1", " ".join(value.splitlines()))


def render(document: ReportDocument, figures: Mapping[str, PlotlyFigureProduct]) -> ReportRender:
    """Render a document from bound products without opening figures or datasets."""
    roles = document.figure_roles
    missing = [role for role in roles if role not in figures]
    if missing:
        raise ValueError(f"Missing saved report figures: {', '.join(missing)}")
    if document.completeness == "complete":
        omissions = [
            f"{role}/{item.layer}: {item.reason}"
            for role in roles
            for item in figures[role].omitted_layers
        ]
        if omissions:
            raise ValueError("Complete report has omitted figure layers: " + "; ".join(omissions))

    assets: dict[str, ContentRef] = {}
    indexes: dict[str, int] = {}
    source_by_role: dict[str, int] = {}
    for role in roles:
        content = figures[role].content
        path = f"figures/{content.sha256}.json"
        if path in assets and assets[path] != content:
            raise ValueError(f"Conflicting saved figure identities for {path}.")
        indexes.setdefault(path, len(indexes))
        assets[path] = content
        source_by_role[role] = indexes[path]

    markdown = [f"# {_literal_markdown(document.title)}"]
    title = escape(document.title)
    html = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{title}</title></head><body>",
        f"<main><h1>{title}</h1>",
    ]
    parser = MarkdownIt("js-default")
    for index, block in enumerate(document.blocks, start=1):
        if isinstance(block, ReportText):
            markdown.append(block.markdown)
            html.append(parser.render(block.markdown).rstrip())
            continue

        caption = block.caption or block.figure
        anchor = f"report-figure-{index}"
        source = source_by_role[block.figure]
        markdown.append(f"[{_literal_markdown(caption)}]({_HTML_NAME}#{anchor})")
        html.extend(
            [
                f'<figure id="{anchor}">',
                f'<div data-report-source="{source}"></div>',
                f"<figcaption>{escape(caption)}</figcaption>",
            ]
        )
        omitted = figures[block.figure].omitted_layers
        if omitted:
            reasons = [f"Omitted layer {item.layer}: {item.reason}" for item in omitted]
            markdown.append("\n".join(f"- {_literal_markdown(reason)}" for reason in reasons))
            html.extend(["<ul>", *(f"<li>{escape(reason)}</li>" for reason in reasons), "</ul>"])
        html.append("</figure>")

    html.append("</main>")
    if assets:
        html.extend(
            [
                '<script type="application/json" id="report-figure-sources">'
                f"{json.dumps(list(assets), separators=(',', ':'))}</script>",
                f'<script src="{_PLOTLY_SCRIPT}"></script>',
                _DRAW_FIGURES,
            ]
        )
    html.append("</body></html>")
    return ReportRender(
        html="\n".join(html) + "\n",
        markdown="\n\n".join(markdown) + "\n",
        figures=assets,
    )
