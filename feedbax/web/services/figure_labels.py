"""Project current Task signal labels onto a saved figure's inferred axes."""

from collections.abc import Mapping, Sequence
from functools import cache

from pydantic import JsonValue

from feedbax.contracts.dataset import DatasetDescriptor
from feedbax.contracts.effective_success import Bindings
from feedbax.contracts.figure_document import FigureDocument, VariableBinding, position_encodings
from feedbax.contracts.operation import DatasetPort
from feedbax.contracts.studio_figure import FigureProduct
from feedbax.host.light_jobs import dataset_member
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset


def _axis_title(layout: dict[str, JsonValue], reference: str) -> dict[str, JsonValue] | None:
    if "." in reference:
        scene, name = reference.split(".", 1)
        axes = layout.get(scene)
        axis = axes.get(name) if isinstance(axes, dict) else None
    else:
        axis = layout.get(f"{reference[0]}axis{reference[1:]}")
    title = axis.get("title") if isinstance(axis, dict) else None
    return title if isinstance(title, dict) else None


def project_figure_task_labels(
    content: bytes, document: FigureDocument, bindings: Bindings, store: ImmutableArtifactBlobProvider,
    evaluations: Mapping[tuple[str, str], Sequence[DatasetPort]],
) -> bytes:
    """Return a view of the figure, keeping the stored product in immutable custody."""
    if not evaluations:
        return content
    layers = {layer.name: layer for layer in document.layers}
    task_axes: dict[str, dict[str, str]] = {}
    for layer in layers.values():
        for dimension, encoding in position_encodings(layer):
            name = encoding.variable if isinstance(encoding, VariableBinding) else encoding
            if dimension in ("x", "y", "z") and isinstance(name, str) and name.startswith("/v1/task/output/"):
                task_axes.setdefault(layer.name, {})[dimension] = name
    if not task_axes:
        return content

    figure = FigureProduct.model_validate_json(content)
    first_layers = {}
    for trace in figure.data:
        layer = layers[trace.meta.layer]
        first_layers.setdefault(layer.panel, layer)

    @cache
    def descriptors(role: str) -> tuple[DatasetDescriptor, ...]:
        retained = []
        for member in bindings[role]:
            with open_stored_dataset(store, dataset_member(member)) as dataset:
                retained.append(dataset.descriptor)
        return tuple(retained)

    changed = False
    for panel in document.layout:
        layer = first_layers.get(panel.name)
        if layer is None or (axes := task_axes.get(layer.name)) is None:
            continue
        runs = figure.layout.meta.runs.get(layer.role, ())
        sources = [evaluations.get((source.job_id, source.request_id), ()) for source in runs]
        if not sources or not all(sources):
            continue
        view = figure.layout.meta.panels[panel.name]
        for dimension, name in axes.items():
            if getattr(panel.axes_labels, dimension, None) is not None:
                continue
            planned = [port.variables.get(name) for group in sources for port in group]
            labels = {field.label if field is not None else None for field in planned}
            if len(labels) != 1 or (label := labels.pop()) is None:
                continue
            reference = getattr(view, f"{dimension}_axis")
            title = _axis_title(figure.layout.__pydantic_extra__, reference) if reference is not None else None
            if title is None:
                continue
            units = {descriptor.field(name).units for descriptor in descriptors(layer.role)}
            if len(units) != 1:
                continue
            unit = units.pop()
            text = f"{label} ({unit})" if unit else label
            if title.get("text") != text:
                title["text"] = text
                changed = True
    return figure.model_dump_json(exclude_unset=True).encode() if changed else content
