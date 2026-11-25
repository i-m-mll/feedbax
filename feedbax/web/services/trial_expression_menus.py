"""Keep the expression scopes of recently prepared trial-authoring views."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.contracts.expression_site import ExpressionPartSite, project_expression_part
from feedbax.contracts.expression_values import Expression, Site
from feedbax.contracts.project_resolution import RunKey
from feedbax.contracts.studio_trials import StudioTrialExpressionMenuRequest
from feedbax.project_folder import ProjectFolderInvalid


@dataclass(frozen=True)
class TrialExpressionField:
    site: Site
    expression: Expression | None


@dataclass(frozen=True)
class _PreparedTrialExpressionView:
    project: str
    revision: str
    run: RunKey
    fields: Mapping[str, TrialExpressionField]


class TrialExpressionMenus:
    """A process-local cache of scoped expressions, keyed by the authoring view's token."""

    def __init__(self) -> None:
        self._views: OrderedDict[str, _PreparedTrialExpressionView] = OrderedDict()
        self._lock = Lock()

    def remember(
        self, project: str, revision: str, run: RunKey,
        fields: Mapping[str, TrialExpressionField],
    ) -> str:
        token = uuid4().hex
        view = _PreparedTrialExpressionView(project, revision, run, dict(fields))
        with self._lock:
            self._views[token] = view
            if len(self._views) > 32:
                self._views.popitem(last=False)
        return token

    def menu(
        self, project: str, body: StudioTrialExpressionMenuRequest, functions: FunctionRegistry,
    ) -> ExpressionPartSite | None:
        run = RunKey(body.run.variant, body.run.step, body.run.row)
        with self._lock:
            view = self._views.get(body.expression_context)
            if view is None or (view.project, view.revision, view.run) != (project, body.revision, run):
                return None
            self._views.move_to_end(body.expression_context)
        field = view.fields.get(body.field)
        if field is None:
            raise ProjectFolderInvalid("There is no expression menu at this trial field.")
        part = project_expression_part(field.site, functions, field.expression, body.part)
        if part is None:
            raise ProjectFolderInvalid("There is no part at this path in the trial field.")
        return part
