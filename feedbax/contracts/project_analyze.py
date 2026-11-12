"""Lower a resolved Analyze row through declared dataset and request contracts."""

from __future__ import annotations

from collections.abc import Mapping
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import TYPE_CHECKING

from pydantic import ValidationError

from feedbax.contracts.analysis_document import (
    AnalysisDocument,
    DatasetSource,
    FigureElement,
    OperationDataset,
    OperationElement,
    StepDataset,
)
from feedbax.contracts.figure_document import FigureBindingError, validate_figure_bindings
from feedbax.contracts.operation import DatasetPort, require_dataset_fields, require_dataset_port
from feedbax.contracts.project import Payload
from feedbax.contracts.project_plan import ElementKey, PlannedElement, PlannedInput, RunKey
from feedbax.contracts.project_resolution import ProjectDiagnostic, ResolutionCode
from feedbax.contracts.request_identity import EncodedRequest, encode_request
from feedbax.contracts.request_refs import InputMember, ProductRef
from feedbax.contracts.request_specs import (
    AnalysisRequestSpec,
    EvaluationRequestSpec,
    RenderRequestSpec,
    RequestSpec,
)
from feedbax.contracts.rollout import ROLLOUT_TYPE, RolloutParameters

if TYPE_CHECKING:
    from feedbax.analysis.operations import OperationRegistry


class _ElementIssue(ValueError):
    def __init__(self, path: str, detail: str):
        super().__init__(detail)
        self.path = path


class _Lowerer:
    def __init__(
        self,
        key: RunKey,
        document_ids: Mapping[str, str],
        source: PlannedInput,
        evaluation: RequestSpec,
        operations: OperationRegistry,
    ):
        self.key = key
        self.document_ids = document_ids
        self.source = source
        self.evaluation = evaluation
        self.operations = operations
        self.elements: dict[str, PlannedElement] = {}
        self.encoded: list[EncodedRequest] = []
        self.problems: list[ProjectDiagnostic] = []

    def problem(
        self,
        path: str,
        detail: str,
        element: str | None = None,
        *,
        code: ResolutionCode = ResolutionCode.INVALID_PAYLOAD,
    ) -> None:
        doc_path = max(
            (base for base in self.document_ids if path == base or path.startswith(base + ".")),
            key=len,
            default=None,
        )
        self.problems.append(
            ProjectDiagnostic(
                code,
                path,
                detail,
                self.key.step,
                document=self.document_ids[doc_path] if doc_path is not None else None,
                row=self.key.row,
                variant=self.key.variant,
                element=element,
            )
        )

    def _input(
        self,
        source: DatasetSource,
        path: str,
    ) -> tuple[PlannedInput, DatasetPort]:
        if isinstance(source, StepDataset):
            if source.input != "states":
                raise _ElementIssue(path, f"Unknown Analyze step input {source.input!r}.")
            evaluation = self.evaluation
            if (
                not isinstance(evaluation, EvaluationRequestSpec)
                or evaluation.evaluation_type != ROLLOUT_TYPE
            ):
                raise _ElementIssue(
                    path, "The states input requires a saved rollout Evaluate request."
                )
            try:
                params = RolloutParameters.model_validate(evaluation.params)
            except ValidationError as error:
                raise _ElementIssue(path, f"Invalid saved Evaluate parameters: {error}") from error
            # The schema is fixed by retention. Concrete model-owned variables are
            # authenticated when the actual saved dataset is read.
            return self.source, DatasetPort(schema_ids=(params.states_schema,))
        producer = self.elements.get(source.element)
        if producer is None:
            raise _ElementIssue(path, f"Unknown or unplannable operation {source.element!r}.")
        output = producer.datasets.get(source.product)
        if output is None:
            raise _ElementIssue(
                path, f"Operation {source.element!r} has no dataset product {source.product!r}."
            )
        return PlannedInput(
            self.key,
            InputMember(
                key="default",
                reference=ProductRef(request_id=producer.request_id, role=source.product),
            ),
            producer_element=source.element,
        ), output

    def _inputs(
        self,
        sources: Mapping[str, DatasetSource],
        roles: set[str],
    ) -> tuple[dict[str, PlannedInput], dict[str, DatasetPort]]:
        if unknown := sources.keys() - roles:
            role = min(unknown)
            raise _ElementIssue(f"inputs.{role}", f"Unknown dataset input {role!r}.")
        if missing := roles - sources.keys():
            role = min(missing)
            raise _ElementIssue(f"inputs.{role}", f"Required dataset input {role!r} is missing.")
        inputs, schemas = {}, {}
        for role in sorted(roles):
            inputs[role], schemas[role] = self._input(sources[role], f"inputs.{role}")
        return inputs, schemas

    def _record(
        self,
        name: str,
        spec: AnalysisRequestSpec | RenderRequestSpec,
        inputs: Mapping[str, PlannedInput],
        outputs: Mapping[str, DatasetPort],
    ) -> None:
        encoded = encode_request(
            spec, {role: (binding.member,) for role, binding in inputs.items()}
        )
        kind = "analyze" if isinstance(spec, AnalysisRequestSpec) else "render"
        self.elements[name] = PlannedElement(
            ElementKey(self.key, name),
            kind,
            encoded.id,
            MappingProxyType(dict(inputs)),
            MappingProxyType(dict(outputs)),
        )
        self.encoded.append(encoded)

    def operation(self, name: str, element: OperationElement) -> None:
        path = f"payload.operations.{name}"
        try:
            operation = self.operations.resolve(element.operation)
        except KeyError:
            self.problem(
                path + ".operation", f"Unknown registered operation {element.operation!r}.", name
            )
            return
        try:
            params = operation.parameter_model.model_validate(element.params)
        except ValidationError as error:
            for item in error.errors(include_url=False, include_input=False):
                location = ".".join(map(str, ("params", *item["loc"])))
                self.problem(f"{path}.{location}", item["msg"], name)
            return
        try:
            definition = operation.definition
            inputs, schemas = self._inputs(element.inputs, set(definition.inputs))
            for role, expected in definition.inputs.items():
                actual = schemas[role]
                try:
                    require_dataset_port(
                        actual, expected, fields=isinstance(element.inputs[role], OperationDataset)
                    )
                    if isinstance(element.inputs[role], OperationDataset):
                        for parameter, requirement in expected.selected_variables.items():
                            require_dataset_fields(
                                actual.variables, {getattr(params, parameter): requirement}
                            )
                except ValueError as error:
                    raise _ElementIssue(f"inputs.{role}", str(error)) from error
            self._record(
                name,
                AnalysisRequestSpec(
                    analysis_type=element.operation,
                    params=params.model_dump(mode="json"),
                    evaluation_states_policy="require_durable",
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

    def view(self, name: str, element: FigureElement) -> None:
        path = f"payload.views.{name}"
        try:
            inputs, schemas = self._inputs(element.inputs, set(element.figure.data))
        except _ElementIssue as error:
            self.problem(
                f"{path}.{error.path}", str(error), name, code=ResolutionCode.BAD_REFERENCE
            )
            return
        try:
            validate_figure_bindings(element.figure, schemas)
            self._record(name, RenderRequestSpec(figure=None, document=element.figure), inputs, {})
        except FigureBindingError as error:
            self.problem(f"{path}.figure.{error.path}", str(error), name)
        except ValueError as error:
            self.problem(path + ".figure", str(error), name)


def lower_analyze(
    key: RunKey,
    payload: Payload,
    document_ids: Mapping[str, str],
    source: PlannedInput,
    evaluation: RequestSpec,
    operations: OperationRegistry,
) -> tuple[dict[str, PlannedElement], list[EncodedRequest], list[ProjectDiagnostic]]:
    lower = _Lowerer(key, document_ids, source, evaluation, operations)
    try:
        document = AnalysisDocument.model_validate(payload)
    except ValidationError as error:
        for item in error.errors(include_url=False, include_input=False):
            location = item["loc"]
            element = (
                str(location[1])
                if len(location) >= 2 and location[0] in ("operations", "views")
                else None
            )
            lower.problem(".".join(map(str, ("payload", *location))), item["msg"], element)
        return lower.elements, lower.encoded, lower.problems

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
    for name, element in sorted(document.views.items()):
        lower.view(name, element)
    return lower.elements, lower.encoded, lower.problems
