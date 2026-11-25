"""Lower a resolved Analyze row through declared dataset and request contracts."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
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
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
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
from feedbax.contracts.operation import (
    DatasetInputPort, DatasetOutputPort, DatasetPort, WholeDimensions, require_dataset_fields, require_dataset_port,
)
from feedbax.contracts.project import Document, Payload, json_pointer_parts
from feedbax.planner.project_plan import (
    ElementKey,
    LabelName,
    LabelNames,
    label_value_names,
    PlannedElement,
    PlannedInput,
    RunKey,
)
from feedbax.contracts.project_resolution import ProjectDiagnostic, ResolutionCode, compose_document
from feedbax.contracts.request_identity import EncodedRequest, encode_request
from feedbax.contracts.project_sweep import SweepInputMember
from feedbax.contracts.request_refs import InputMember, MemberLabel, ProductRef
from feedbax.contracts.request_specs import (
    AnalysisRequestSpec,
    EvaluationRequestSpec,
    RenderRequestSpec,
    RequestSpec,
)
from feedbax.contracts.rollout import ROLLOUT_TYPE, RolloutParameters, authored_variation
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


def _condition_label(params: RolloutParameters) -> str | None:
    """Display authored variation values for a rollout whose row has no label."""
    values = {}
    for _, edit in authored_variation(params.variation):
        value = edit.value
        if edit.path == "task" and isinstance(value, dict):
            for key in tuple(values):
                if key.startswith("task."):
                    del values[key]
            values["task"] = value.get("id") or value.get("type")
            values.update((f"task.{key}", item) for key, item in value.get("params", {}).items())
            for variable in value.get("variables", ()):
                if isinstance(expression := variable["value"], dict) and expression.get("kind") == "literal":
                    values[f"task.{variable['name']}"] = expression["value"]
        else:
            name = edit.path.removeprefix("graph.nodes.").replace(".params.", ".")
            if isinstance(value, dict) and value.get("kind") == "literal":
                value = value["value"]
            values[name] = "removed" if edit.op == "remove" else value
    return ", ".join(
        f"{name}={value if isinstance(value, str) else canonical_json_v2_bytes(value).decode()}"
        for name, value in values.items()
    ) or None


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


type _Entries = tuple[SweepInputMember, ...]


def _fields(entries: _Entries, path: str) -> dict[str, FieldDescriptor]:
    try:
        return stack_layout(entries).fields() if entries else {}
    except ValueError as error:
        raise _ElementIssue(path, str(error)) from error


def _function_of(label: str, whole: set[str], bindings: tuple[PlannedInput, ...]) -> bool:
    """Whether a label's value is fixed by the values of ``whole``, so that it is a coordinate along them."""
    seen: dict[tuple[str | None, ...], str | None] = {}
    for binding in bindings:
        labels = binding.member.labels
        key = tuple(labels[name].value_id if name in labels else None for name in sorted(whole))
        value = labels[label].value_id if label in labels else None
        if seen.setdefault(key, value) != value:
            return False
    return True


@dataclass(frozen=True)
class _Group:
    """The members of a role one mapped request reads, and the labels it is one combination of."""

    labels: Mapping[str, MemberLabel | None]
    bindings: tuple[PlannedInput, ...]


def _value_id(label: MemberLabel | None) -> str | None:
    return label.value_id if label is not None else None


def _stripped(binding: PlannedInput, mapped: Collection[str]) -> PlannedInput:
    labels = {name: label for name, label in binding.member.labels.items() if name not in mapped}
    return replace(binding, member=binding.member.model_copy(update={"labels": labels}))



def _carried(entries: Mapping[str, _Entries], outputs: Mapping[str, DatasetPort | PlotlyFigurePort]) -> _Entries:
    """The labelled entries an operation's output carries from the roles it takes whole: theirs when they are equal,
    otherwise those of the role its output port names."""
    distinct: list[_Entries] = []
    for found in entries.values():
        if found and found not in distinct:
            distinct.append(found)
    if len(distinct) <= 1:
        return distinct[0] if distinct else ()
    named = {port.carries for port in outputs.values() if isinstance(port, DatasetOutputPort) and port.carries}
    if len(named) != 1 or (role := named.pop()) not in entries:
        roles = sorted(role for role, found in entries.items() if found)
        raise _ElementIssue("", f"Inputs {roles!r} carry different labels; its output names none of them to carry.")
    return entries[role]


def _member_ids(requests: list[Mapping[str, MemberLabel]], names: LabelNames) -> list[str]:
    """Each mapped request's member id: its values, by name, of the labels that vary across the element's requests.

    A label varies when two requests hold different values of it; one that only some requests hold, with one value,
    tells apart nothing another label does not."""
    named = [label_value_names(mapped, names) for mapped in requests]
    order = tuple(dict.fromkeys(label for values in named for label in values))
    varying = [label for label in order if len({values[label] for values in named if label in values}) > 1]
    return [",".join(f"{label}={values[label]}" for label in varying if label in values) for values in named]


class _Lowerer:
    def __init__(
        self,
        key: RunKey,
        document_ids: Mapping[str, str],
        document_pointers: Mapping[str, str],
        documents: Mapping[str, Document],
        step_inputs: Mapping[str, tuple[PlannedInput, ...]],
        evaluations: Mapping[RunKey, tuple[RequestSpec, LabelName]],
        upstream: Mapping[ElementKey, PlannedElement],
        operations: OperationRegistry,
        figure_templates: FigureTemplateRegistry | None,
    ):
        self.key = key
        self.document_ids = document_ids
        self.document_pointers = document_pointers
        self.documents = documents
        self.step_inputs = step_inputs
        self.evaluations = evaluations
        self.upstream = upstream
        self.operations = operations
        self.declarations: Mapping[str, AnalyzeInput] = {}
        self.element_names: dict[str, str] = {}
        self.figure_templates = figure_templates
        self.elements: dict[str, list[PlannedElement]] = {}
        self.figures: dict[str, FigureDocument] = {}
        self.encoded: list[EncodedRequest] = []
        self.problems: list[ProjectDiagnostic] = []

    @property
    def planned(self) -> dict[ElementKey, PlannedElement]:
        return {element.key: element for requests in self.elements.values() for element in requests}

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
                element_name=self.element_names.get(element) if element is not None else None,
                document_path=document_path,
            )
        )

    def _producer(self, binding: PlannedInput) -> PlannedElement | None:
        """The operation request a binding reads, in this card or in the card across an Analyze-to-Analyze edge."""
        key = binding.element
        if key is None:
            return None
        if key.run != self.key:
            return self.upstream.get(key)
        return next((element for element in self.elements.get(key.name, ()) if element.key == key), None)

    def _input(
        self,
        source: DatasetSource,
        path: str,
    ) -> tuple[tuple[PlannedInput, ...], DatasetPort, _Entries]:
        """The bindings of a dataset input, its port, and the labelled entries the stacked dataset holds."""
        if isinstance(source, OperationDataset):
            requests = None if source.unresolved else self.elements.get(source.element)
            if not requests:
                raise _ElementIssue(path, f"Unknown or unplannable operation {source.element!r}.")
            bindings = tuple(
                PlannedInput(
                    self.key,
                    InputMember(
                        key=producer.stack_key,
                        reference=ProductRef(request_id=producer.request_id, role=source.product),
                        labels=dict(producer.mapped),
                    ),
                    producer_element=source.element,
                    producer_member=producer.key.member,
                )
                for producer in requests
            )
            return self._operation_outputs(bindings, path)
        inputs = self._step_input(source, "dataset", path)
        if any(binding.producer_element is not None for binding in inputs):
            return self._operation_outputs(inputs, path)
        schemas = set()
        for binding in inputs:
            reference = binding.member.reference
            assert isinstance(reference, ProductRef) and binding.producer is not None
            evaluation, _ = self.evaluations[binding.producer]
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
        entries = self._entries(inputs, path)
        return inputs, DatasetPort(schema_ids=tuple(schemas), coordinates=_fields(entries, path)), entries

    def _entries(self, inputs: tuple[PlannedInput, ...], path: str) -> _Entries:
        """The labelled entries of a stack of bindings, as ``stack_dataset_members`` lays them out at run time.

        A binding that reads an operation output holding labelled entries of its own contributes each of them, with
        its inner labels and its own, keyed by the inner key qualified by the binding's key. One output read alone
        with no labels is the dataset the operation made.
        """
        if len(inputs) == 1 and not inputs[0].member.labels:
            producer = self._producer(inputs[0])
            return producer.entries if producer is not None else ()
        entries = []
        for binding, outer in zip(inputs, labelled_members(tuple(binding.request_member for binding in inputs)), strict=True):
            producer = self._producer(binding)
            if producer is None or not producer.entries:
                entries.append(outer)
                continue
            for inner in producer.entries:
                if "element" in inner.coordinates:
                    raise _ElementIssue(
                        path, f"Member {outer.key!r} already holds the outputs of several operations; "
                        "a list stacks one level of operation outputs.",
                    )
                if shared := sorted(outer.coordinates.keys() & inner.coordinates.keys()):
                    raise _ElementIssue(path, f"Member {outer.key!r} already carries the labels {shared!r}.")
                entries.append(SweepInputMember(
                    f"{outer.key}/{inner.key}", {**outer.coordinates, **inner.coordinates},
                ))
        return tuple(entries) if any(entry.coordinates for entry in entries) else ()

    def _operation_outputs(
        self, inputs: tuple[PlannedInput, ...], path: str,
    ) -> tuple[tuple[PlannedInput, ...], DatasetPort, _Entries]:
        """Operation outputs read by a wire or by a role: one output is the dataset the operation made, and several
        are stacked flat by their labels, which requires them to have one port."""
        ports = []
        for binding in inputs:
            producer = self._producer(binding)
            reference = binding.member.reference
            if producer is None or not isinstance(reference, ProductRef):
                raise _ElementIssue(path, f"Unknown or unplannable operation {binding.producer_element!r}.")
            output = producer.datasets.get(reference.role)
            if output is None:
                raise _ElementIssue(
                    path, f"Operation {binding.producer_element!r} has no dataset product {reference.role!r}."
                )
            ports.append(output)
        if any(port != ports[0] for port in ports[1:]):
            raise _ElementIssue(path, "Stacked operation outputs must have one dataset port.")
        entries = self._entries(inputs, path)
        labels = _fields(entries, path)
        return inputs, ports[0].model_copy(update={"coordinates": {**labels, **ports[0].coordinates}}), entries

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
            inputs[role], schemas[role], _ = self._input(sources[role], f"inputs.{role}")
        return inputs, schemas

    @staticmethod
    def _check_roles(sources: Mapping[str, DatasetSource], roles: set[str]) -> None:
        if unknown := sources.keys() - roles:
            role = min(unknown)
            raise _ElementIssue(f"inputs.{role}", f"Unknown input {role!r}.")
        if missing := roles - sources.keys():
            role = min(missing)
            raise _ElementIssue(f"inputs.{role}", f"Required input {role!r} is missing.")

    def _names(self, inputs: Mapping[str, tuple[PlannedInput, ...]]) -> dict[str, dict[str, LabelName]]:
        names = _merged(*(
            producer.names
            for bindings in inputs.values() for binding in bindings
            if (producer := self._producer(binding)) is not None
        ), _names(inputs))
        for bindings in inputs.values():
            for binding in bindings:
                if binding.producer is None or (evaluation := self.evaluations.get(binding.producer)) is None:
                    continue
                spec, name = evaluation
                if isinstance(spec, EvaluationRequestSpec) and spec.evaluation_type == ROLLOUT_TYPE:
                    params = RolloutParameters.model_validate(spec.params)
                    if name.label is None:
                        name = replace(name, label=_condition_label(params))
                    names.setdefault("condition", {}).setdefault(params.condition_id(), name)
        return names

    def _record(
        self,
        name: str,
        spec: AnalysisRequestSpec | RenderRequestSpec,
        inputs: Mapping[str, tuple[PlannedInput, ...]],
        outputs: Mapping[str, DatasetPort | PlotlyFigurePort],
        label: str | None,
        names: Mapping[str, Mapping[str, LabelName]],
        *,
        member: str | None = None,
        mapped: Mapping[str, MemberLabel] = MappingProxyType({}),
        entries: _Entries = (),
    ) -> None:
        if isinstance(spec, RenderRequestSpec):
            printed = _printed_labels(spec.document)
            spec = spec.model_copy(update={"names": {
                label: {value_id: name.printed for value_id, name in values.items()}
                for label, values in sorted(names.items()) if label in printed
            }})
        encoded = encode_request(
            spec, {role: tuple(binding.request_member for binding in bindings) for role, bindings in inputs.items()}
        )
        kind = "analyze" if isinstance(spec, AnalysisRequestSpec) else "render"
        self.elements.setdefault(name, []).append(PlannedElement(
            ElementKey(self.key, name, member),
            kind,
            encoded.id,
            MappingProxyType(dict(inputs)),
            MappingProxyType(dict(outputs)),
            MappingProxyType({name: MappingProxyType(dict(values)) for name, values in names.items()}),
            MappingProxyType(dict(mapped)),
            entries,
            label,
        ))
        self.encoded.append(encoded)

    def _groups(
        self, role: str, takes: str | WholeDimensions, bindings: tuple[PlannedInput, ...], params, path: str,
    ) -> tuple[_Group, ...] | None:
        """The members of a role each mapped request reads, or None when the port takes the whole stack."""
        if takes == "all":
            return None
        expands = [
            binding.member.key for binding in bindings
            if (producer := self._producer(binding)) is not None and len(producer.entries) > 1
        ]
        names = tuple(dict.fromkeys(label for binding in bindings for label in binding.member.labels))
        if not names:
            return None
        if takes == "one":
            if expands:
                raise _ElementIssue(
                    path, f"Input {role!r} takes one member, but {expands[0]!r} holds several; "
                    "map the operation that made it.",
                )
            return tuple(_Group({label: binding.member.labels.get(label) for label in names}, (binding,))
                         for binding in bindings)
        assert isinstance(takes, WholeDimensions)
        listed = getattr(params, takes.parameter)
        if not isinstance(listed, (list, tuple)) or not all(isinstance(item, str) for item in listed):
            raise _ElementIssue(f"params.{takes.parameter}", "Dimensions are a list of label names.")
        inner = {label for binding in bindings if (producer := self._producer(binding)) is not None
                 for entry in producer.entries for label in entry.coordinates}
        if unknown := sorted(set(listed) - set(names) - inner):
            raise _ElementIssue(
                f"params.{takes.parameter}", f"Input {role!r} has no label dimensions {unknown!r}.",
            )
        whole = set(listed)
        whole |= {label for label in names if label not in whole and _function_of(label, whole, bindings)} if whole else set()
        if mapped_inner := sorted(inner - whole):
            raise _ElementIssue(
                path, f"Input {role!r} holds {mapped_inner!r} inside one operation output; "
                "map the operation that made it.",
            )
        mapped = tuple(label for label in names if label not in whole)
        if not mapped:
            return None
        groups: dict[tuple[str | None, ...], list[PlannedInput]] = {}
        for binding in bindings:
            labels = binding.member.labels
            groups.setdefault(tuple(labels[label].value_id if label in labels else None for label in mapped), []).append(binding)
        return tuple(
            _Group({label: grouped[0].member.labels.get(label) for label in mapped}, tuple(grouped))
            for grouped in groups.values()
        )

    @staticmethod
    def _paired(groups: Mapping[str, tuple[_Group, ...]], path: str) -> list[dict[str, _Group]]:
        """Combinations of one group per mapped role that agree on the labels they share."""
        combinations: list[dict[str, _Group]] = [{}]
        for role, role_groups in groups.items():
            paired = []
            for combination in combinations:
                for group in role_groups:
                    if all(
                        _value_id(other.labels[label]) == _value_id(group.labels[label])
                        for other in combination.values() for label in other.labels.keys() & group.labels.keys()
                    ):
                        paired.append(combination | {role: group})
            combinations = paired
        used = {(role, id(group)) for combination in combinations for role, group in combination.items()}
        if any((role, id(group)) not in used for role, role_groups in groups.items() for group in role_groups):
            raise _ElementIssue(path, "Mapped inputs do not pair one to one by their shared labels.")
        return combinations

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
            entries: dict[str, _Entries] = {}
            for role, expected in definition.inputs.items():
                source = element.inputs[role]
                wire_path = f"inputs.{role}"
                if not isinstance(expected, DatasetPort):
                    inputs[role] = self._step_input(source, expected.kind, wire_path)
                    continue
                inputs[role], actual, entries[role] = self._input(source, wire_path)
                outputs = any(binding.producer_element is not None for binding in inputs[role])
                try:
                    require_dataset_port(actual, expected, fields=outputs)
                    if outputs:
                        for parameter, requirement in expected.selected_variables.items():
                            require_dataset_fields(
                                actual.variables, {getattr(params, parameter): requirement}
                            )
                except ValueError as error:
                    raise _ElementIssue(wire_path, str(error)) from error
            ports = {role: port for role, port in definition.inputs.items() if isinstance(port, DatasetInputPort)}
            groups = {
                role: grouped for role, port in ports.items()
                if (grouped := self._groups(role, port.takes, inputs[role], params, f"inputs.{role}")) is not None
            }
            names = self._names(inputs)
            spec = AnalysisRequestSpec(analysis_type=registered.key, params=params.model_dump(mode="json"))
            if not groups:
                self._record(
                    name, spec, inputs, definition.outputs, element.label, names,
                    entries=_carried(entries, definition.outputs),
                )
                return
            requests = []
            for combination in self._paired(groups, "inputs"):
                mapped = {
                    label: value for group in combination.values() for label, value in group.labels.items()
                    if value is not None
                }
                labels = {label for group in combination.values() for label in group.labels}
                request = {
                    role: tuple(_stripped(binding, labels) for binding in combination[role].bindings)
                    if role in combination else bindings
                    for role, bindings in inputs.items()
                }
                inner = _carried(
                    {role: self._entries(request[role], f"inputs.{role}") for role in entries}, definition.outputs,
                )
                requests.append((mapped, request, inner))
            members = _member_ids([mapped for mapped, *_ in requests], names)
            if len(set(members)) != len(requests):
                raise _ElementIssue("inputs", "Mapped requests repeat one combination of labels.")
            for member, (mapped, request, inner) in zip(members, requests, strict=True):
                self._record(
                    name, spec, request, definition.outputs, element.label, names,
                    member=member, mapped=mapped, entries=inner,
                )
        except _ElementIssue as error:
            self.problem(
                f"{path}.{error.path}" if error.path else path, str(error), name, code=ResolutionCode.BAD_REFERENCE
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
                element.label,
                self._names(inputs),
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
    evaluations: Mapping[RunKey, tuple[RequestSpec, LabelName]],
    upstream: Mapping[ElementKey, PlannedElement],
    operations: OperationRegistry,
    *,
    figure_templates: FigureTemplateRegistry | None = None,
) -> tuple[
    dict[ElementKey, PlannedElement], list[EncodedRequest], list[ProjectDiagnostic], dict[str, FigureDocument]
]:
    lower = _Lowerer(
        key, document_ids, document_pointers, documents, step_inputs, evaluations, upstream, operations,
        figure_templates,
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
        return lower.planned, lower.encoded, lower.problems, lower.figures

    lower.declarations = document.inputs
    lower.element_names = {
        name: item.label or name
        for items in (document.operations, document.views) for name, item in items.items()
    }
    graph = TopologicalSorter()
    for name, element in sorted(document.operations.items()):
        graph.add(
            name,
            *sorted(
                {
                    source.element
                    for source in element.inputs.values()
                    if isinstance(source, OperationDataset) and not source.unresolved
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
    return lower.planned, lower.encoded, lower.problems, lower.figures
