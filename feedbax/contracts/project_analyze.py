"""Lower a resolved Analyze row through declared dataset and request contracts."""

from __future__ import annotations

from collections.abc import Mapping
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import TYPE_CHECKING

from pydantic import JsonValue, ValidationError

from feedbax.contracts.analysis_document import (
    AnalysisDocument,
    AnalyzeInput,
    DatasetSource,
    FigureElement,
    OperationDataset,
    OperationElement,
    StepInput,
)
from feedbax.contracts.figure_document import (
    FIGURE_DOCUMENT_SCHEMA,
    PLOTLY_FIGURE_ROLE,
    FigureBindingError,
    FigureDocument,
    FigureTemplateRef,
    PlotlyFigurePort,
    is_figure_document_pin,
    layer_encodings,
    validate_figure_bindings,
)
from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.member_stack import labelled_members, stack_layout
from feedbax.contracts.operation import DatasetPort, require_dataset_fields, require_dataset_port
from feedbax.contracts.project import Document, Payload, json_pointer_parts
from feedbax.contracts.project_plan import (
    ElementKey,
    LabelName,
    LabelNames,
    PlannedElement,
    PlannedInput,
    RunKey,
)
from feedbax.contracts.project_resolution import ProjectDiagnostic, ResolutionCode, compose_document
from feedbax.contracts.request_identity import EncodedRequest, encode_request
from feedbax.contracts.request_refs import InputMember, ProductRef
from feedbax.contracts.request_specs import (
    AnalysisRequestSpec,
    EvaluationRequestSpec,
    RenderRequestSpec,
    RequestSpec,
)
from feedbax.contracts.rollout import ROLLOUT_TYPE, RolloutParameters
from feedbax.plugins.declarations import missing_declaration

if TYPE_CHECKING:
    from feedbax.analysis.operations import OperationRegistry
    from feedbax.plot.templates import FigureTemplateRegistry


def _names(inputs: Mapping[str, tuple[PlannedInput, ...]]) -> dict[str, dict[str, LabelName]]:
    """What each step-valued label value ID of ``inputs`` names."""
    names: dict[str, dict[str, LabelName]] = {}
    for bindings in inputs.values():
        for binding in bindings:
            for label, name in binding.named.items():
                if (value := binding.member.labels.get(label)) is not None:
                    names.setdefault(label, {})[value.value_id] = name
    return names


def _merged(*tables: LabelNames) -> dict[str, dict[str, LabelName]]:
    merged: dict[str, dict[str, LabelName]] = {}
    for table in tables:
        for label, names in table.items():
            merged.setdefault(label, {}).update(names)
    return merged


def _printed_labels(figure: FigureDocument) -> set[str]:
    """The coordinates a figure splits or selects by, whose names it therefore prints or matches."""
    used = set()
    for layer in figure.layers:
        used.update(layer.group, layer.select)
        for _, encoding in layer_encodings(layer):
            used.update(getattr(encoding, "select", {}))
    return used


class _ElementIssue(ValueError):
    def __init__(self, path: str, detail: str):
        super().__init__(detail)
        self.path = path


class _Lowerer:
    def __init__(
        self,
        key: RunKey,
        document_ids: Mapping[str, str],
        document_pointers: Mapping[str, str],
        documents: Mapping[str, Document],
        step_inputs: Mapping[str, tuple[PlannedInput, ...]],
        evaluations: Mapping[str, RequestSpec],
        operations: OperationRegistry,
        figure_templates: FigureTemplateRegistry | None,
    ):
        self.key = key
        self.document_ids = document_ids
        self.document_pointers = document_pointers
        self.documents = documents
        self.step_inputs = step_inputs
        self.evaluations = evaluations
        self.operations = operations
        self.declarations: Mapping[str, AnalyzeInput] = {}
        self.figure_templates = figure_templates
        self.elements: dict[str, PlannedElement] = {}
        self.labels: dict[str, dict[str, FieldDescriptor]] = {}
        """The label coordinates each operation's dataset outputs carry from its stacked inputs."""
        self.figures: dict[str, FigureDocument] = {}
        self.encoded: list[EncodedRequest] = []
        self.problems: list[ProjectDiagnostic] = []

    def problem(
        self,
        path: str,
        detail: str,
        element: str | None = None,
        *,
        code: ResolutionCode = ResolutionCode.INVALID_PAYLOAD,
        document_path: str | None = None,
    ) -> None:
        doc_path = max(
            (base for base in self.document_ids if path == base or path.startswith(base + ".")),
            key=len,
            default=None,
        )
        name = self.document_ids[doc_path] if doc_path is not None else None
        if name is not None and doc_path is not None and document_path is None and path.split(".", 4)[3:4] == ["figure"]:
            authored = compose_document(self.documents.get, name)
            document_path = (
                "payload.path" if not isinstance(authored, ProjectDiagnostic)
                and authored.schema_id == FIGURE_DOCUMENT_SCHEMA and is_figure_document_pin(authored.payload)
                else ".".join(("payload", *json_pointer_parts(self.document_pointers.get(doc_path)))) + path[len(doc_path):]
            )
        self.problems.append(
            ProjectDiagnostic(
                code,
                path,
                detail,
                self.key.step,
                document=name,
                row=self.key.row,
                variant=self.key.variant,
                element=element,
                document_path=document_path,
            )
        )

    def _input(
        self,
        source: DatasetSource,
        path: str,
    ) -> tuple[tuple[PlannedInput, ...], DatasetPort]:
        """The bindings of a dataset input and its port, with the coordinates its stacked labels add."""
        if isinstance(source, StepInput):
            inputs = self._step_input(source, "dataset", path)
            schemas = set()
            for binding in inputs:
                reference = binding.member.reference
                assert isinstance(reference, ProductRef)
                evaluation = self.evaluations[reference.request_id]
                if (
                    not isinstance(evaluation, EvaluationRequestSpec)
                    or evaluation.evaluation_type != ROLLOUT_TYPE
                ):
                    raise _ElementIssue(
                        path, f"States member {binding.member.key!r} requires a saved rollout Evaluate request."
                    )
                try:
                    schemas.add(RolloutParameters.model_validate(evaluation.params).states_schema)
                except ValidationError as error:
                    raise _ElementIssue(path, f"Invalid saved Evaluate parameters: {error}") from error
            # The schema is fixed by retention. Concrete model-owned variables are
            # authenticated when the actual saved dataset is read.
            if len(schemas) != 1:
                raise _ElementIssue(path, "Selected Evaluate states must use the same dataset schema.")
            members = tuple(binding.member for binding in inputs)
            try:
                labels = stack_layout(labelled_members(members)).fields() if any(
                    member.labels for member in members
                ) else {}
            except ValueError as error:
                raise _ElementIssue(path, str(error)) from error
            return inputs, DatasetPort(schema_ids=tuple(schemas), coordinates=labels)
        producer = self.elements.get(source.element)
        if producer is None:
            raise _ElementIssue(path, f"Unknown or unplannable operation {source.element!r}.")
        output = producer.datasets.get(source.product)
        if output is None:
            raise _ElementIssue(
                path, f"Operation {source.element!r} has no dataset product {source.product!r}."
            )
        binding = PlannedInput(
            self.key,
            InputMember(
                key="default",
                reference=ProductRef(request_id=producer.request_id, role=source.product),
            ),
            producer_element=source.element,
        )
        return (binding,), output.model_copy(update={"coordinates": {
            **self.labels.get(source.element, {}), **output.coordinates,
        }})

    def _step_input(
        self, source: DatasetSource, kind: str, path: str,
    ) -> tuple[PlannedInput, ...]:
        if not isinstance(source, StepInput):
            raise _ElementIssue(path, f"Connect an Analyze {kind} input role.")
        declaration = self.declarations.get(source.input)
        if declaration is None or declaration.kind != kind:
            raise _ElementIssue(path, f"Analyze input {source.input!r} does not declare {kind}.")
        if not (members := self.step_inputs.get(source.input)):
            raise _ElementIssue(path, f"Analyze input {source.input!r} has no binding.")
        return members

    def _inputs(
        self,
        sources: Mapping[str, DatasetSource],
        roles: set[str],
    ) -> tuple[dict[str, tuple[PlannedInput, ...]], dict[str, DatasetPort]]:
        self._check_roles(sources, roles)
        inputs, schemas = {}, {}
        for role in sorted(roles):
            inputs[role], schemas[role] = self._input(sources[role], f"inputs.{role}")
        return inputs, schemas

    @staticmethod
    def _check_roles(sources: Mapping[str, DatasetSource], roles: set[str]) -> None:
        if unknown := sources.keys() - roles:
            role = min(unknown)
            raise _ElementIssue(f"inputs.{role}", f"Unknown input {role!r}.")
        if missing := roles - sources.keys():
            role = min(missing)
            raise _ElementIssue(f"inputs.{role}", f"Required input {role!r} is missing.")

    def _record(
        self,
        name: str,
        spec: AnalysisRequestSpec | RenderRequestSpec,
        inputs: Mapping[str, tuple[PlannedInput, ...]],
        outputs: Mapping[str, DatasetPort | PlotlyFigurePort],
    ) -> None:
        names = _merged(*(
            self.elements[binding.producer_element].names
            for bindings in inputs.values() for binding in bindings
            if binding.producer == self.key and binding.producer_element in self.elements
        ), _names(inputs))
        if isinstance(spec, RenderRequestSpec):
            printed = _printed_labels(spec.document)
            spec = spec.model_copy(update={"names": {
                label: {value_id: name.printed for value_id, name in values.items()}
                for label, values in sorted(names.items()) if label in printed
            }})
        encoded = encode_request(
            spec, {role: tuple(binding.member for binding in bindings) for role, bindings in inputs.items()}
        )
        kind = "analyze" if isinstance(spec, AnalysisRequestSpec) else "render"
        self.elements[name] = PlannedElement(
            ElementKey(self.key, name),
            kind,
            encoded.id,
            MappingProxyType(dict(inputs)),
            MappingProxyType(dict(outputs)),
            MappingProxyType({label: MappingProxyType(values) for label, values in names.items()}),
        )
        self.encoded.append(encoded)

    def operation(self, name: str, element: OperationElement) -> None:
        path = f"payload.operations.{name}"
        try:
            registered = self.operations.resolve(element.operation)
        except KeyError:
            self.problem(
                path + ".operation", missing_declaration("operation", element.operation), name
            )
            return
        try:
            params = registered.value.parameter_model.model_validate(element.params)
        except ValidationError as error:
            for item in error.errors(include_url=False, include_input=False):
                location = ".".join(map(str, ("params", *item["loc"])))
                self.problem(f"{path}.{location}", item["msg"], name)
            return
        try:
            definition = registered.value.definition
            self._check_roles(element.inputs, set(definition.inputs))
            inputs = {}
            labels: dict[str, FieldDescriptor] = {}
            for role, expected in definition.inputs.items():
                source = element.inputs[role]
                wire_path = f"inputs.{role}"
                if not isinstance(expected, DatasetPort):
                    inputs[role] = self._step_input(source, expected.kind, wire_path)
                    continue
                inputs[role], actual = self._input(source, wire_path)
                labels |= self.labels.get(source.element, {}) if isinstance(source, OperationDataset) else actual.coordinates
                try:
                    require_dataset_port(
                        actual, expected, fields=isinstance(source, OperationDataset)
                    )
                    if isinstance(source, OperationDataset):
                        for parameter, requirement in expected.selected_variables.items():
                            require_dataset_fields(
                                actual.variables, {getattr(params, parameter): requirement}
                            )
                except ValueError as error:
                    raise _ElementIssue(wire_path, str(error)) from error
            self.labels[name] = labels
            self._record(
                name,
                AnalysisRequestSpec(
                    analysis_type=registered.key,
                    params=params.model_dump(mode="json"),
                ),
                inputs,
                definition.outputs,
            )
        except _ElementIssue as error:
            self.problem(
                f"{path}.{error.path}", str(error), name, code=ResolutionCode.BAD_REFERENCE
            )
        except ValueError as error:
            self.problem(path, str(error), name)

    def view(self, name: str, element: FigureElement[JsonValue]) -> None:
        path = f"payload.views.{name}"
        figure_path = path + ".figure"
        document = self.document_ids.get(figure_path)
        if document is not None:
            schema = self.documents[document].schema_id
            if schema != FIGURE_DOCUMENT_SCHEMA:
                self.problem(
                    figure_path, f"Expected {FIGURE_DOCUMENT_SCHEMA!r}, found {schema!r}.", name,
                    code=ResolutionCode.INVALID_DOCUMENT, document_path="schema",
                )
                return
        validation_path = figure_path
        try:
            if isinstance(element.figure, dict) and "template" in element.figure:
                reference = FigureTemplateRef.model_validate(element.figure)
                registry = self.figure_templates
                try:
                    registration = registry.resolve(reference.template) if registry is not None else None
                except KeyError:
                    registration = None
                if registration is None:
                    package = reference.template.partition(".")[0]
                    self.problem(
                        figure_path + ".template",
                        f"No installed figure template provides {reference.template!r} "
                        f"from package {package!r}.",
                        name,
                    )
                    return
                validation_path = figure_path + ".settings"
                settings = registration.settings_model.model_validate(reference.settings)
                validation_path = figure_path
                figure = registration.function(settings)
            else:
                figure = FigureDocument.model_validate(element.figure)
        except ValidationError as error:
            for item in error.errors(include_url=False, include_input=False):
                location = item["loc"]
                if location[:1] == ("layers",) and location[2:3] in {("line",), ("band",)}:
                    location = (*location[:2], *location[3:])
                self.problem(
                    ".".join(map(str, (validation_path, *location))), item["msg"], name,
                    code=ResolutionCode.INVALID_DOCUMENT if document else ResolutionCode.INVALID_PAYLOAD,
                )
            return
        self.figures[name] = figure
        try:
            inputs, schemas = self._inputs(element.inputs, set(figure.data))
        except _ElementIssue as error:
            self.problem(
                f"{path}.{error.path}", str(error), name, code=ResolutionCode.BAD_REFERENCE
            )
            return
        try:
            validate_figure_bindings(figure, schemas)
            self._record(
                name,
                RenderRequestSpec(document=figure),
                inputs,
                {PLOTLY_FIGURE_ROLE: PlotlyFigurePort()},
            )
        except FigureBindingError as error:
            self.problem(f"{figure_path}.{error.path}", str(error), name)
        except ValueError as error:
            self.problem(figure_path, str(error), name)


def lower_analyze(
    key: RunKey,
    payload: Payload,
    document_ids: Mapping[str, str],
    document_pointers: Mapping[str, str],
    documents: Mapping[str, Document],
    document_problems: tuple[ProjectDiagnostic, ...],
    step_inputs: Mapping[str, tuple[PlannedInput, ...]],
    evaluations: Mapping[str, RequestSpec],
    operations: OperationRegistry,
    *,
    figure_templates: FigureTemplateRegistry | None = None,
) -> tuple[
    dict[str, PlannedElement], list[EncodedRequest], list[ProjectDiagnostic], dict[str, FigureDocument]
]:
    lower = _Lowerer(
        key, document_ids, document_pointers, documents, step_inputs, evaluations, operations, figure_templates,
    )
    lower.problems.extend(document_problems)
    try:
        # Each figure validates below so a bad figure leaves its siblings plannable.
        document = AnalysisDocument[JsonValue].model_validate(payload)
    except ValidationError as error:
        for item in error.errors(include_url=False, include_input=False):
            location = item["loc"]
            element = (
                str(location[1])
                if len(location) >= 2 and location[0] in ("operations", "views")
                else None
            )
            lower.problem(".".join(map(str, ("payload", *location))), item["msg"], element)
        return lower.elements, lower.encoded, lower.problems, lower.figures

    lower.declarations = document.inputs
    graph = TopologicalSorter()
    for name, element in sorted(document.operations.items()):
        graph.add(
            name,
            *sorted(
                {
                    source.element
                    for source in element.inputs.values()
                    if isinstance(source, OperationDataset)
                    and source.element in document.operations
                }
            ),
        )
    cycle = set()
    try:
        graph.prepare()
    except CycleError as error:
        members = error.args[1]
        cycle = set(members)
        detail = "Operation dependency cycle: " + " -> ".join(members)
        for name in sorted(cycle):
            lower.problem(
                f"payload.operations.{name}.inputs", detail, name, code=ResolutionCode.BAD_REFERENCE
            )
    visited = set(cycle)
    while graph.is_active():
        for name in graph.get_ready():
            lower.operation(name, document.operations[name])
            graph.done(name)
            visited.add(name)
    for name in sorted(document.operations.keys() - visited):
        lower.problem(
            f"payload.operations.{name}.inputs",
            "An input depends on an operation dependency cycle.",
            name,
            code=ResolutionCode.BAD_REFERENCE,
        )
    refused = {problem.element for problem in document_problems}
    for name, element in sorted(document.views.items()):
        if name not in refused:
            lower.view(name, element)
    return lower.elements, lower.encoded, lower.problems, lower.figures
