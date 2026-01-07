"""Main Dash application for the figure review dashboard."""

import json
import logging
import uuid
from typing import Any, Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash.dependencies import ALL, MATCH
from dash.exceptions import PreventUpdate
from sqlalchemy.orm import Session

from feedbax.config import configure_globals_for_package
from feedbax.database import FigureRecord, get_db_session
from feedbax.dashboard.backend.loader import FigureLoader
from feedbax.dashboard.backend.presets import PresetManager
from feedbax.dashboard.backend.query import FilterSpec, FigureQueryEngine
from feedbax.plugins import EXPERIMENT_REGISTRY

logger = logging.getLogger(__name__)


# Constants used to keep dropdown + validation logic centralized.
_VALID_TABLES = {"figures", "evaluations"}
_FILTER_OPERATORS = ["=", "!=", "<", ">", "<=", ">=", "IN", "LIKE", "IS NULL", "IS NOT NULL"]
_NULL_OPERATORS = {"IS NULL", "IS NOT NULL"}
_TABLE_OPTIONS = [
    {"label": "Figure Metadata", "value": "figures"},
    {"label": "Evaluation Metadata", "value": "evaluations"},
]
_OPERATOR_OPTIONS = [{"label": op, "value": op} for op in _FILTER_OPERATORS]


# Initialize database session globally (will be set in create_app)
_db_session: Optional[Session] = None
_query_engine: Optional[FigureQueryEngine] = None
_figure_loader: Optional[FigureLoader] = None
_preset_manager: Optional[PresetManager] = None


def _normalize_table_name(table: Optional[str]) -> str:
    """Restrict table names to known options."""
    return table if table in _VALID_TABLES else "evaluations"


def _normalize_operator(operator: Optional[str]) -> str:
    """Restrict operators to known options."""
    return operator if operator in _FILTER_OPERATORS else "="


def _format_value_for_display(value: Any, operator: str) -> str:
    """Format stored values as strings for form controls."""
    if operator in _NULL_OPERATORS:
        return ""
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value)
        except TypeError:
            return str(value)
    return str(value)


def _parse_value_from_display(value: Any, operator: str) -> Any:
    """Convert a text entry back into a typed value."""
    if operator in _NULL_OPERATORS:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return ""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return value
    return value


def _make_filter_dict(
    *,
    table: str = "evaluations",
    column: str = "",
    operator: str = "=",
    value: Any = None,
    uid: Optional[str] = None,
    locked: bool = False,
    is_identifier: bool = False,
    active: bool = True,
) -> dict[str, Any]:
    """Create a filter row representation for the UI."""
    normalized_op = _normalize_operator(operator)
    if value is None:
        value_display = None
        value_ready = False
    else:
        value_display = _format_value_for_display(value, normalized_op)
        value_ready = True
    return {
        "uid": uid or uuid.uuid4().hex,
        "table": _normalize_table_name(table),
        "column": column or "",
        "operator": normalized_op,
        "value": value_display,
        "value_ready": value_ready,
        "locked": locked,
        "is_identifier": is_identifier,
        "active": True if locked else active,
    }


def _update_filter_entry(filters: list[dict[str, Any]], uid: str, **updates) -> list[dict[str, Any]]:
    """Update a specific filter entry identified by uid."""
    updated_filters: list[dict[str, Any]] = []
    for entry in filters:
        if entry["uid"] != uid:
            updated_filters.append(entry)
            continue
        new_entry = {**entry}
        for key, value in updates.items():
            if entry.get("locked") and key not in ("value", "active"):
                # Preserve locked rows except for toggling active or selecting value
                continue
            if key == "operator":
                value = _normalize_operator(value)
                if value in _NULL_OPERATORS:
                    new_entry["value"] = None
                    new_entry["value_ready"] = True
            if key == "table":
                value = _normalize_table_name(value)
                new_entry["column"] = ""
                new_entry["value"] = None
                new_entry["value_ready"] = False
            if key == "column":
                value = value or ""
                if value == "":
                    new_entry["value"] = None
                    new_entry["value_ready"] = False
            if key == "value":
                if value is None:
                    new_entry["value"] = None
                    new_entry["value_ready"] = False
                    continue
                new_entry["value_ready"] = True
            new_entry[key] = value
        updated_filters.append(new_entry)
    return updated_filters


def _ensure_identifier_filter(filters: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Ensure the identifier filter row exists and is first."""
    identifier_entry: Optional[dict[str, Any]] = None
    remaining: list[dict[str, Any]] = []

    for entry in filters or []:
        if entry.get("is_identifier") and identifier_entry is None:
            identifier_entry = {
                **entry,
                "table": "figures",
                "column": "identifier",
                "operator": "=",
                "locked": True,
                "active": True,
            }
        elif entry.get("is_identifier"):
            continue
        else:
            remaining.append(entry)

    if identifier_entry is None:
        identifier_entry = _make_filter_dict(
            table="figures",
            column="identifier",
            operator="=",
            locked=True,
            is_identifier=True,
        )

    return [identifier_entry] + remaining


def _hydrate_filters(filter_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert persisted filter specs into UI rows."""
    return [
        _make_filter_dict(
            table=spec.get("table", "evaluations"),
            column=spec.get("column", ""),
            operator=spec.get("operator", "="),
            value=spec.get("value"),
        )
        for spec in filter_specs
    ]


def _convert_store_filters_to_specs(
    filters: list[dict[str, Any]], *, include_identifier: bool = True
) -> list[dict[str, Any]]:
    """Convert UI filter rows into backend filter specifications."""
    specs: list[dict[str, Any]] = []
    for entry in filters:
        if not include_identifier and entry.get("is_identifier"):
            continue
        if not entry.get("active", True):
            continue
        table = entry.get("table")
        column = entry.get("column")
        if not table or not column:
            continue
        operator = _normalize_operator(entry.get("operator"))
        value_field = entry.get("value")
        value_ready = entry.get("value_ready")
        if value_ready is None:
            value_ready = value_field not in (None, "")
        if operator not in _NULL_OPERATORS and not value_ready:
            continue
        value_text = "" if value_field is None else str(value_field)
        specs.append(
            {
                "table": _normalize_table_name(table),
                "column": column,
                "operator": operator,
                "value": _parse_value_from_display(value_text, operator),
            }
        )
    return specs


def _render_filter_row(filter_data: dict[str, Any]) -> html.Div:
    """Render a single filter row."""
    uid = filter_data["uid"]
    locked = filter_data.get("locked", False)
    is_identifier = filter_data.get("is_identifier", False)
    operator = filter_data.get("operator", "=")
    value_ready = filter_data.get("value_ready", False)

    row_classes = ["filter-row"]
    if locked:
        row_classes.append("filter-row--locked")
    if not filter_data.get("column"):
        row_classes.append("filter-row--incomplete")
    if is_identifier:
        row_classes.append("filter-row--identifier")
        if not value_ready:
            row_classes.append("filter-row--identifier-unset")
    if not filter_data.get("active", True):
        row_classes.append("filter-row--inactive")

    badge_element = None
    if is_identifier:
        badge_element = html.Div(
            dbc.Badge(
                html.Span(["FIGURE", html.Br(), "TYPE"]),
                color="primary",
                className="filter-badge-pill",
                pill=True,
            ),
            className="filter-row-badge",
        )

    table_dropdown = dcc.Dropdown(
        options=_TABLE_OPTIONS,
        value=filter_data.get("table"),
        clearable=False,
        disabled=locked,
        id={"type": "filter-table", "uid": uid},
    )

    value_options: list[dict[str, str]] = []
    table_name = filter_data.get("table")
    column_name = filter_data.get("column")
    if table_name and column_name and _query_engine is not None:
        try:
            raw_values = _query_engine.get_column_values(table_name, column_name)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to load value options for %s.%s: %s", table_name, column_name, exc
            )
        else:
            seen_values: set[str] = set()
            for val in raw_values:
                formatted = _format_value_for_display(val, "=")
                label = formatted if formatted != "" else "(empty)"
                if formatted in seen_values:
                    continue
                seen_values.add(formatted)
                value_options.append({"label": label, "value": formatted})

            stored_value = filter_data.get("value")
            if stored_value not in (None, "") and stored_value not in seen_values:
                value_options.append({"label": stored_value, "value": stored_value})

    value_placeholder = "Select value..."
    if is_identifier:
        value_placeholder = "Select figure type..."

    value_dropdown = dcc.Dropdown(
        id={"type": "filter-value", "uid": uid},
        placeholder=value_placeholder,
        value=filter_data.get("value"),
        clearable=True,
        disabled=(
            operator in _NULL_OPERATORS
            or not filter_data.get("column")
            or (locked and not is_identifier)
        ),
        options=value_options,
        className="filter-value-dropdown",
        searchable=True,
    )

    toggle_active = html.Button(
        "",
        id={"type": "filter-toggle", "uid": uid},
        title="Toggle filter",
        disabled=locked,
        type="button",
        className="filter-toggle-switch" + (" is-active" if filter_data.get("active", True) else ""),
    )

    remove_btn = html.Button(
        "",
        id={"type": "filter-remove", "uid": uid},
        title="Remove filter",
        disabled=locked,
        type="button",
        className="filter-trash-btn",
    )

    row_grid = html.Div(
        [
            html.Div(table_dropdown, className="filter-cell filter-cell--table"),
            html.Div(
                dcc.Dropdown(
                    id={"type": "filter-column", "uid": uid},
                    placeholder="Select column...",
                    value=filter_data.get("column") or None,
                    disabled=locked,
                    clearable=False,
                ),
                className="filter-cell filter-cell--column",
            ),
            html.Div(
                dcc.Dropdown(
                    options=_OPERATOR_OPTIONS,
                    value=operator,
                    clearable=False,
                    disabled=locked,
                    id={"type": "filter-operator", "uid": uid},
                ),
                className="filter-cell filter-cell--operator",
            ),
            html.Div(value_dropdown, className="filter-cell filter-cell--value"),
            html.Div(
                html.Div([toggle_active, remove_btn], className="filter-row-actions"),
                className="filter-cell filter-cell--actions",
            ),
        ],
        className="filter-row-grid",
    )

    children = [row_grid]
    if badge_element:
        children.insert(0, badge_element)

    return html.Div(
        children,
        className="filter-row-container " + " ".join(row_classes),
        key=uid,
    )


def create_app(db_name: str = "main") -> dash.Dash:
    """Create and configure the Dash application.

    Args:
        db_name: Name of the database to use

    Returns:
        Configured Dash app
    """
    global _db_session, _query_engine, _figure_loader, _preset_manager

    # Configure globals for the package
    single_package = EXPERIMENT_REGISTRY.single_package_name()
    if single_package:
        configure_globals_for_package(single_package, EXPERIMENT_REGISTRY)

    # Initialize database session
    _db_session = get_db_session(name=db_name)
    _query_engine = FigureQueryEngine(_db_session)
    _figure_loader = FigureLoader()
    _preset_manager = PresetManager()

    # Create Dash app with custom CSS
    # Note: Custom CSS in assets/custom.css will be automatically loaded
    import os
    assets_folder = os.path.join(os.path.dirname(__file__), "assets")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=assets_folder,
    )

    app.layout = create_layout()

    # Register callbacks
    register_callbacks(app)

    return app


def create_layout() -> html.Div:
    """Create the main dashboard layout.

    Returns:
        Dash HTML layout
    """
    return html.Div(
        [
            dbc.Container(
                [
                    html.H1("Figure Review Dashboard", className="mt-4 mb-2"),
                    html.P(
                        "Filter, organize, and review stored figures across evaluation runs.",
                        className="text-muted mb-4",
                    ),
                ],
                fluid=True,
            ),
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                html.H5("Presets", className="dashboard-card-title mb-0")
                                            ),
                                            dbc.CardBody(
                                                [
                                                    dcc.Dropdown(
                                                        id="preset-dropdown",
                                                        placeholder="Select preset...",
                                                        className="mb-2",
                                                    ),
                                                    dbc.Stack(
                                                        [
                                                            dbc.Button(
                                                                "Load",
                                                                id="load-preset-btn",
                                                                color="secondary",
                                                                outline=True,
                                                                size="sm",
                                                            ),
                                                            dbc.Button(
                                                                "Save As",
                                                                id="save-preset-btn",
                                                                color="primary",
                                                                size="sm",
                                                            ),
                                                            dbc.Button(
                                                                "Delete",
                                                                id="delete-preset-btn",
                                                                color="danger",
                                                                outline=True,
                                                                size="sm",
                                                            ),
                                                        ],
                                                        gap=2,
                                                        direction="horizontal",
                                                    ),
                                                    html.Div(id="preset-status-msg", className="mt-2"),
                                                ]
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                html.H5("Active Filters", className="dashboard-card-title mb-0")
                                            ),
                                            dbc.CardBody(
                                                [
                                                    html.Div(
                                                        html.Div(
                                                            [
                                                                html.Div("Table"),
                                                                html.Div("Column"),
                                                                html.Div("Operator"),
                                                                html.Div("Value"),
                                                                html.Div(""),
                                                            ],
                                                            className="filter-builder-grid",
                                                        ),
                                                        className="filter-builder-header-wrapper",
                                                    ),
                                                    html.Div(
                                                        id="filter-rows-container",
                                                        className="filter-builder",
                                                    ),
                                                    dbc.Button(
                                                        "Add Filter",
                                                        id="add-filter-row-btn",
                                                        color="primary",
                                                        className="w-100 mt-3",
                                                    ),
                                                    html.Small(
                                                        "Filters apply SQL-style operators. Separate list values with JSON syntax (e.g., [1, 2]).",
                                                        className="text-muted d-block mt-2",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                html.H5("Columns That Vary", className="dashboard-card-title mb-0")
                                            ),
                                            dbc.CardBody(
                                                html.Div(
                                                    id="distinguishing-columns-display",
                                                    className="distinguishing-columns",
                                                )
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                ],
                                lg=5,
                                className="mb-4",
                            ),
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                html.H5("Grid Configuration", className="dashboard-card-title mb-0")
                                            ),
                                            dbc.CardBody(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label("Row dimension", className="form-label"),
                                                                    dcc.Dropdown(
                                                                        id="row-dimension-dropdown",
                                                                        placeholder="Optional",
                                                                    ),
                                                                ],
                                                                lg=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label("Column dimension", className="form-label"),
                                                                    dcc.Dropdown(
                                                                        id="col-dimension-dropdown",
                                                                        placeholder="Select column dimension...",
                                                                    ),
                                                                ],
                                                                lg=6,
                                                            ),
                                                        ],
                                                        className="g-2",
                                                    ),
                                                    dbc.Button(
                                                        "Generate Grid",
                                                        id="generate-grid-btn",
                                                        color="success",
                                                        className="mt-3 w-100",
                                                    ),
                                                    html.Div(id="grid-validation-msg", className="mt-2"),
                                                ]
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                html.H5("Figure Grid", className="dashboard-card-title mb-0")
                                            ),
                                            dbc.CardBody(
                                                dcc.Loading(
                                                    id="loading-grid",
                                                    type="default",
                                                    children=html.Div(id="figure-grid-display"),
                                                )
                                            ),
                                        ]
                                    ),
                                ],
                                lg=7,
                                className="mb-4",
                            ),
                        ],
                        className="gx-3 gy-2",
                    ),
                ],
                fluid=True,
            ),
            # Modal for saving presets
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Save Preset")),
                    dbc.ModalBody(
                        [
                            html.Label("Preset Name:"),
                            dcc.Input(
                                id="preset-name-input",
                                type="text",
                                className="form-control",
                                placeholder="Enter preset name...",
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Cancel", id="cancel-save-preset-btn", className="me-2"),
                            dbc.Button("Save", id="confirm-save-preset-btn", color="primary"),
                        ]
                    ),
                ],
                id="save-preset-modal",
                is_open=False,
            ),
            # State stores
            dcc.Store(
                id="filters-store",
                data=_ensure_identifier_filter([_make_filter_dict(), _make_filter_dict()]),
            ),
            dcc.Store(id="filtered-df-store", data=None),
            dcc.Store(id="grid-config-store", data=None),
        ]
    )


def register_callbacks(app: dash.Dash):
    """Register all Dash callbacks.

    Args:
        app: Dash application instance
    """

    @app.callback(
        Output("preset-dropdown", "options"),
        Input("preset-dropdown", "id"),
        Input("confirm-save-preset-btn", "n_clicks"),
        Input("delete-preset-btn", "n_clicks"),
    )
    def update_preset_options(*_):
        """Populate preset dropdown with available presets."""
        if _preset_manager is None:
            return []
        presets = _preset_manager.list_presets()
        return [{"label": name, "value": name} for name in presets]

    @app.callback(
        Output("filter-rows-container", "children"),
        Input("filters-store", "data"),
    )
    def render_filter_rows(filters):
        """Render filter form rows from the store."""
        filters = filters or []
        if not filters:
            return dbc.Alert("No filters configured. Add one to get started.", color="light")
        return [_render_filter_row(row) for row in filters]

    @app.callback(
        Output("filters-store", "data"),
        Input("add-filter-row-btn", "n_clicks"),
        Input({"type": "filter-remove", "uid": ALL}, "n_clicks"),
        Input({"type": "filter-toggle", "uid": ALL}, "n_clicks"),
        Input({"type": "filter-table", "uid": ALL}, "value"),
        Input({"type": "filter-column", "uid": ALL}, "value"),
        Input({"type": "filter-operator", "uid": ALL}, "value"),
        Input({"type": "filter-value", "uid": ALL}, "value"),
        Input("load-preset-btn", "n_clicks"),
        State("preset-dropdown", "value"),
        State("filters-store", "data"),
        prevent_initial_call=True,
    )
    def manage_filters(
        add_clicks,
        _remove_clicks,
        _toggle_clicks,
        _table_values,
        _column_values,
        _operator_values,
        _value_values,
        _load_clicks,
        preset_name,
        stored_filters,
    ):
        """Synchronize the filter store with UI interactions."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        triggered_value = ctx.triggered[0].get("value")
        trigger = getattr(ctx, "triggered_id", None)
        if trigger is None:
            raw_id = ctx.triggered[0]["prop_id"].split(".")[0]
            try:
                trigger = json.loads(raw_id)
            except json.JSONDecodeError:
                trigger = raw_id
        filters = stored_filters or []

        logger.info("manage_filters trigger=%s value=%s", trigger, triggered_value)

        if trigger == "add-filter-row-btn":
            filters = filters + [_make_filter_dict()]
        elif trigger == "load-preset-btn":
            if preset_name and _preset_manager:
                loaded = _preset_manager.load_preset(preset_name) or []
                filters = _hydrate_filters(loaded)
        elif isinstance(trigger, dict):
            uid = trigger.get("uid")
            trigger_type = trigger.get("type")
            if trigger_type == "filter-remove" and uid:
                filters = [f for f in filters if f["uid"] != uid or f.get("locked")]
            elif trigger_type == "filter-toggle" and uid:
                target = next((f for f in filters if f["uid"] == uid), None)
                if target and not target.get("locked"):
                    filters = _update_filter_entry(filters, uid, active=not target.get("active", True))
            elif trigger_type == "filter-table" and uid:
                filters = _update_filter_entry(filters, uid, table=triggered_value or "evaluations")
            elif trigger_type == "filter-column" and uid:
                filters = _update_filter_entry(filters, uid, column=triggered_value or "")
            elif trigger_type == "filter-operator" and uid:
                filters = _update_filter_entry(filters, uid, operator=triggered_value or "=")
            elif trigger_type == "filter-value" and uid:
                filters = _update_filter_entry(filters, uid, value=triggered_value or "")

        filters = _ensure_identifier_filter(filters)

        return filters

    @app.callback(
        Output({"type": "filter-column", "uid": MATCH}, "options"),
        Input({"type": "filter-table", "uid": MATCH}, "value"),
    )
    def set_column_options(table_name):
        """Populate column dropdown options based on table selection."""
        if _query_engine is None or not table_name:
            return []

        columns = _query_engine.get_table_columns(table_name)
        return [{"label": col, "value": col} for col in columns]

    @app.callback(
        Output("save-preset-modal", "is_open"),
        Output("preset-status-msg", "children"),
        Input("save-preset-btn", "n_clicks"),
        Input("cancel-save-preset-btn", "n_clicks"),
        Input("confirm-save-preset-btn", "n_clicks"),
        Input("delete-preset-btn", "n_clicks"),
        State("preset-name-input", "value"),
        State("preset-dropdown", "value"),
        State("filters-store", "data"),
        State("save-preset-modal", "is_open"),
        prevent_initial_call=True,
    )
    def manage_presets(save_clicks, cancel_clicks, confirm_clicks, delete_clicks,
                      preset_name, selected_preset, filters, modal_open):
        """Handle preset save/load/delete operations."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return False, ""

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Open save modal
        if trigger_id == "save-preset-btn":
            return True, ""

        # Cancel save modal
        if trigger_id == "cancel-save-preset-btn":
            return False, ""

        # Confirm save
        if trigger_id == "confirm-save-preset-btn":
            if preset_name and _preset_manager:
                filter_specs = _convert_store_filters_to_specs(filters or [], include_identifier=False)
                success = _preset_manager.save_preset(preset_name, filter_specs)
                if success:
                    return False, dbc.Alert(f"Preset '{preset_name}' saved", color="success", dismissable=True)
                else:
                    return False, dbc.Alert("Failed to save preset", color="danger", dismissable=True)
            return False, dbc.Alert("Please enter a preset name", color="warning", dismissable=True)

        # Delete preset
        if trigger_id == "delete-preset-btn":
            if selected_preset and _preset_manager:
                success = _preset_manager.delete_preset(selected_preset)
                if success:
                    return False, dbc.Alert(f"Preset '{selected_preset}' deleted", color="success", dismissable=True)
                else:
                    return False, dbc.Alert("Failed to delete preset", color="danger", dismissable=True)
            return False, dbc.Alert("Please select a preset to delete", color="warning", dismissable=True)

        return modal_open, ""

    @app.callback(
        Output("filtered-df-store", "data"),
        Input("filters-store", "data"),
    )
    def update_filtered_data(filters):
        """Update filtered DataFrame based on current filters."""
        if _query_engine is None or not filters:
            return None

        valid_filters = _convert_store_filters_to_specs(filters)
        if not valid_filters:
            return None

        try:
            filter_specs = [FilterSpec(**f) for f in valid_filters]
        except Exception as exc:
            logger.error(f"Failed to create filter specs: {exc}")
            return None

        # Get filtered DataFrame
        df = _query_engine.get_filtered_dataframe(filter_specs)

        if df.empty:
            return None

        # Convert to JSON-serializable format
        return df.to_json(orient="split")

    @app.callback(
        Output("distinguishing-columns-display", "children"),
        Input("filtered-df-store", "data"),
        State("grid-config-store", "data"),
    )
    def display_distinguishing_columns(df_json, grid_config):
        """Display columns that still vary in the filtered data."""
        if not df_json or _query_engine is None:
            return html.P("No data", className="text-muted")

        df = pd.read_json(df_json, orient="split")

        # Exclude grid dimensions if configured
        exclude_cols = set()
        if grid_config:
            if grid_config.get("row_col"):
                exclude_cols.add(grid_config["row_col"])
            if grid_config.get("col_col"):
                exclude_cols.add(grid_config["col_col"])

        distinguishing = _query_engine.get_distinguishing_columns(df, exclude_cols)

        if not distinguishing:
            return html.P("All columns are constant", className="text-muted")

        # Create table with separate Table column
        rows = []
        for col_info in distinguishing[:20]:  # Limit to 20 columns
            samples_str = ", ".join(str(v)[:30] for v in col_info.sample_values[:3])
            if len(col_info.sample_values) > 3:
                samples_str += ", ..."

            rows.append(
                html.Tr(
                    [
                        html.Td(col_info.table),
                        html.Td(col_info.column),
                        html.Td(str(col_info.n_unique)),
                        html.Td(samples_str, style={"font-size": "0.9em"}),
                    ]
                )
            )

        return dbc.Table(
            [
                html.Thead(html.Tr([html.Th("Table"), html.Th("Column"), html.Th("#"), html.Th("Samples")])),
                html.Tbody(rows),
            ],
            size="sm",
            striped=True,
        )

    @app.callback(
        Output("row-dimension-dropdown", "options"),
        Output("col-dimension-dropdown", "options"),
        Input("filtered-df-store", "data"),
        State("grid-config-store", "data"),
    )
    def update_grid_dimension_options(df_json, grid_config):
        """Update grid dimension dropdown options based on distinguishing columns."""
        if not df_json or _query_engine is None:
            return [], []

        df = pd.read_json(df_json, orient="split")

        # Exclude grid dimensions if already configured
        exclude_cols = set()
        if grid_config:
            if grid_config.get("row_col"):
                exclude_cols.add(grid_config["row_col"])
            if grid_config.get("col_col"):
                exclude_cols.add(grid_config["col_col"])

        distinguishing = _query_engine.get_distinguishing_columns(df, exclude_cols)

        # Format: "table.column (n values)" but still need to keep "table.column" as value
        # This is fine as-is since the dropdowns need the combined value for indexing
        options = [
            {"label": f"{c.table}.{c.column} ({c.n_unique} values)", "value": f"{c.table}.{c.column}"}
            for c in distinguishing
        ]

        return options, options

    @app.callback(
        Output("grid-config-store", "data"),
        Output("grid-validation-msg", "children"),
        Input("generate-grid-btn", "n_clicks"),
        State("row-dimension-dropdown", "value"),
        State("col-dimension-dropdown", "value"),
        State("filtered-df-store", "data"),
        prevent_initial_call=True,
    )
    def configure_grid(n_clicks, row_col, col_col, df_json):
        """Validate and configure grid dimensions."""
        if not df_json or not col_col or _query_engine is None:
            return None, dbc.Alert("Please select a column dimension", color="warning", className="mt-2")

        df = pd.read_json(df_json, orient="split")

        # Validate uniqueness
        is_valid, error_msg = _query_engine.validate_grid_uniqueness(df, row_col, col_col)

        if not is_valid:
            return None, dbc.Alert(error_msg, color="danger", className="mt-2")

        grid_config = {"row_col": row_col, "col_col": col_col}

        return grid_config, dbc.Alert(
            f"Grid configured: {df.shape[0]} figures to display", color="success", className="mt-2"
        )

    @app.callback(
        Output("figure-grid-display", "children"),
        Input("grid-config-store", "data"),
        State("filtered-df-store", "data"),
    )
    def generate_figure_grid(grid_config, df_json):
        """Generate and display the figure grid."""
        if not grid_config or not df_json or _query_engine is None or _figure_loader is None:
            return html.P("Configure grid to display figures", className="text-muted")

        df = pd.read_json(df_json, orient="split")

        # Build grid data
        grid_data = _query_engine.build_grid_data(
            df,
            grid_config.get("row_col"),
            grid_config["col_col"],
        )

        if not grid_data.records:
            return html.P("No figures to display", className="text-muted")

        # Create grid layout
        grid_children = []

        # Header row (column labels)
        header_row = [html.Div("", style={"grid-column": 1, "font-weight": "bold"})]  # Empty corner
        for i, col_val in enumerate(grid_data.col_values, start=2):
            header_row.append(
                html.Div(
                    str(col_val)[:50],
                    style={
                        "grid-column": i,
                        "font-weight": "bold",
                        "padding": "10px",
                        "text-align": "center",
                    },
                )
            )
        grid_children.extend(header_row)

        # Data rows
        for row_idx, row_val in enumerate(grid_data.row_values or [None], start=2):
            # Row label
            if row_val is not None:
                grid_children.append(
                    html.Div(
                        str(row_val)[:50],
                        style={
                            "grid-column": 1,
                            "grid-row": row_idx,
                            "font-weight": "bold",
                            "padding": "10px",
                            "text-align": "right",
                        },
                    )
                )

            # Figure cells
            for col_idx, col_val in enumerate(grid_data.col_values, start=2):
                record_dict = grid_data.records.get((row_val, col_val))

                if record_dict:
                    # Create FigureRecord from dict (need hash and evaluation_hash)
                    fig_hash = record_dict.get("figures.hash")
                    eval_hash = record_dict.get("figures.evaluation_hash")

                    # Create a minimal FigureRecord-like object
                    class MinimalFigureRecord:
                        def __init__(self, hash, evaluation_hash):
                            self.hash = hash
                            self.evaluation_hash = evaluation_hash

                        def get_path(self, format):
                            from feedbax.config import PATHS

                            return PATHS.figures / self.evaluation_hash / f"{self.hash}.{format}"

                    fig_record = MinimalFigureRecord(fig_hash, eval_hash)

                    # Load figure
                    fig_data = _figure_loader.load_figure(fig_record, format="png")

                    if fig_data:
                        cell_content = html.Img(
                            src=fig_data.content,
                            style={"max-width": "100%", "height": "auto"},
                        )
                    else:
                        cell_content = html.Div(
                            "⚠️ File not found",
                            style={"color": "orange", "padding": "20px", "text-align": "center"},
                        )
                else:
                    cell_content = html.Div(
                        "❌",
                        style={"color": "red", "font-size": "2em", "text-align": "center", "padding": "20px"},
                    )

                grid_children.append(
                    dbc.Card(
                        dbc.CardBody(cell_content),
                        style={"grid-column": col_idx, "grid-row": row_idx},
                    )
                )

        n_cols = len(grid_data.col_values) + 1  # +1 for row labels
        n_rows = len(grid_data.row_values or [None]) + 1  # +1 for column labels

        return html.Div(
            grid_children,
            style={
                "display": "grid",
                "grid-template-columns": f"auto repeat({len(grid_data.col_values)}, 1fr)",
                "grid-template-rows": f"auto repeat({len(grid_data.row_values or [None])}, auto)",
                "gap": "10px",
                "padding": "20px",
            },
        )


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
