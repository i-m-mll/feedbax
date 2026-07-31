from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec, acausal_interior_content_hash
from feedbax.contracts.domain import (
    DomainCompileReport,
    DomainDiagnostic,
    derive_compile_status,
)
from feedbax.contracts.graph import ComponentSpec, GraphProject, GraphSpec
from feedbax.contracts.graphs.acausal_compiler import compile_acausal_authoring_report
from feedbax.contracts.graphs.mechanics_templates import two_link_arm_6muscle_template_graph
from feedbax.web.app import create_app
from feedbax.web.services.graph_service import GraphService


pytestmark = pytest.mark.feedbax_contract


def _registry() -> ComponentRegistry:
    return ComponentRegistry(load_user_components=False)


def _report(graph: AcausalGraphSpec) -> DomainCompileReport:
    return compile_acausal_authoring_report(
        graph,
        node_path=["plant"],
        component_registry=_registry(),
    )


def _msd_interior() -> AcausalGraphSpec:
    return AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="Ground"),
            "mass": ComponentSpec(type="Mass", params={"mass": 1.0}),
            "spring": ComponentSpec(type="LinearSpring", params={"stiffness": 10.0}),
            "damper": ComponentSpec(type="LinearDamper", params={"damping": 0.5}),
            "act": ComponentSpec(
                type="ActuationInput",
                params={"port_name": "u", "source_kind": "force"},
            ),
            "sense": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "pos", "quantity": "position"},
            ),
        },
        connections=[
            {"a": ("wall", "flange"), "b": ("spring", "flange_a")},
            {"a": ("wall", "flange"), "b": ("damper", "flange_a")},
            {"a": ("spring", "flange_b"), "b": ("mass", "flange")},
            {"a": ("damper", "flange_b"), "b": ("mass", "flange")},
            {"a": ("act", "flange"), "b": ("mass", "flange")},
            {"a": ("sense", "flange"), "b": ("mass", "flange")},
        ],
        solver={"solver_type": "euler", "dt": 0.001},
    )


def _codes(report: DomainCompileReport) -> set[str]:
    return {diagnostic.code for diagnostic in report.diagnostics}


def _diagnostic(report: DomainCompileReport, code: str) -> DomainDiagnostic:
    matches = [diagnostic for diagnostic in report.diagnostics if diagnostic.code == code]
    assert matches, f"{code} missing from {[diagnostic.code for diagnostic in report.diagnostics]}"
    return matches[0]


def test_domain_compile_report_rejects_ok_with_error_diagnostic() -> None:
    with pytest.raises(ValueError, match="status 'ok' cannot include error diagnostics"):
        DomainCompileReport(
            status="ok",
            interior_content_hash="abc",
            diagnostics=[
                DomainDiagnostic(
                    severity="error",
                    code="acausal.unbalanced",
                    message="network has equations=0 and unknowns=2",
                )
            ],
            summary={},
        )


def test_analyze_system_lives_in_no_jax_import_module() -> None:
    source = Path("feedbax/acausal/analysis.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert "jax" not in imported_roots


def test_valid_acausal_compile_report_is_ok_and_has_interface_and_hash() -> None:
    report = _report(_msd_interior())

    assert report.status == "ok"
    assert report.diagnostics == []
    assert report.interior_content_hash == acausal_interior_content_hash(_msd_interior())
    assert report.derived_interface is not None
    assert sorted(report.derived_interface["inputs"]) == ["u"]
    assert sorted(report.derived_interface["outputs"]) == ["pos"]
    assert report.summary["n_elements"] == 6
    assert report.summary["n_networks"] == 1


def test_planar_multibody_report_is_ok_with_dof_summary() -> None:
    graph = two_link_arm_6muscle_template_graph()
    report = _report(graph)

    assert report.status == "ok"
    diagnostic = _diagnostic(report, "mechanics.dof_summary")
    assert diagnostic.severity == "info"
    assert diagnostic.counts == {"n_dof": 2, "n_links": 2, "n_muscles": 6}
    assert report.derived_interface is not None
    assert sorted(report.derived_interface["inputs"]) == ["excitation"]
    assert sorted(report.derived_interface["outputs"]) == ["effector", "state"]
    assert report.summary["n_dof"] == 2


def test_planar_multibody_reports_unanchored_chain() -> None:
    graph = two_link_arm_6muscle_template_graph()
    graph = graph.model_copy(
        update={
            "nodes": {
                key: value for key, value in graph.nodes.items() if key not in {"world", "anchor"}
            }
        }
    )

    diagnostic = _diagnostic(_report(graph), "mechanics.unanchored_chain")

    assert "WorldFrame" in diagnostic.message
    assert diagnostic.counts == {"n_world_frames": 0, "n_anchors": 0}


def test_planar_multibody_reports_missing_muscle_path_frame() -> None:
    graph = two_link_arm_6muscle_template_graph()
    nodes = dict(graph.nodes)
    muscle = nodes["muscle_0"]
    nodes["muscle_0"] = muscle.model_copy(
        update={
            "params": {
                **muscle.params,
                "path_points": [{"frame": "upper.nope"}, {"frame": "forearm.distal"}],
            }
        }
    )
    graph = graph.model_copy(update={"nodes": nodes})

    diagnostic = _diagnostic(_report(graph), "mechanics.muscle_path_missing_frame")

    assert "upper.nope" in diagnostic.message
    assert diagnostic.node_ids == ["muscle_0"]


@pytest.mark.parametrize(
    ("code", "graph", "must_include"),
    [
        (
            "acausal.empty_interior",
            AcausalGraphSpec(
                physical_domain="translational",
                nodes={},
                connections=[],
                solver={"solver_type": "euler", "dt": 0.001},
            ),
            ("no elements", "n_elements"),
        ),
        (
            "acausal.unknown_element_type",
            AcausalGraphSpec(
                physical_domain="translational",
                nodes={"mystery": ComponentSpec(type="NoSuchAcausalElement")},
                connections=[],
                solver={"solver_type": "euler", "dt": 0.001},
            ),
            ("mystery", "NoSuchAcausalElement"),
        ),
        (
            "acausal.duplicate_boundary_port_name",
            AcausalGraphSpec(
                physical_domain="translational",
                nodes={
                    "a": ComponentSpec(type="BoundaryPort", params={"port_name": "flange"}),
                    "b": ComponentSpec(type="BoundaryPort", params={"port_name": "flange"}),
                },
                connections=[],
                solver={"solver_type": "euler", "dt": 0.001},
            ),
            ("flange", "a", "b"),
        ),
        (
            "acausal.dangling_conserving_port",
            AcausalGraphSpec(
                physical_domain="translational",
                nodes={"free": ComponentSpec(type="BoundaryPort", params={"port_name": "flange"})},
                connections=[],
                solver={"solver_type": "euler", "dt": 0.001},
            ),
            ("free", "flange"),
        ),
        (
            "acausal.adapter_unit_mismatch",
            AcausalGraphSpec(
                physical_domain="translational",
                nodes={
                    "act": ComponentSpec(
                        type="ActuationInput",
                        params={"source_kind": "torque", "port_name": "u"},
                    )
                },
                connections=[],
                solver={"solver_type": "euler", "dt": 0.001},
            ),
            ("act", "torque", "translational"),
        ),
        (
            "acausal.domain_mismatch",
            AcausalGraphSpec(
                physical_domain="translational",
                nodes={
                    "act": ComponentSpec(
                        type="ActuationInput",
                        params={"source_kind": "torque", "port_name": "u"},
                    )
                },
                connections=[],
                solver={"solver_type": "euler", "dt": 0.001},
            ),
            ("act", "rotational", "translational"),
        ),
        (
            "acausal.unbalanced",
            AcausalGraphSpec(
                physical_domain="translational",
                nodes={"spring": ComponentSpec(type="LinearSpring")},
                connections=[],
                solver={"solver_type": "euler", "dt": 0.001},
            ),
            ("spring", "equations", "unknowns"),
        ),
        (
            "acausal.ungrounded_network",
            AcausalGraphSpec(
                physical_domain="translational",
                nodes={
                    "mass": ComponentSpec(type="Mass"),
                    "act": ComponentSpec(type="ActuationInput", params={"source_kind": "force"}),
                },
                connections=[{"a": ("act", "flange"), "b": ("mass", "flange")}],
                solver={"solver_type": "euler", "dt": 0.001},
            ),
            ("mass", "act", "Ground"),
        ),
        (
            "acausal.parallel_across_sources",
            AcausalGraphSpec(
                physical_domain="translational",
                nodes={
                    "motion_a": ComponentSpec(type="PrescribedMotion"),
                    "motion_b": ComponentSpec(type="PrescribedMotion"),
                },
                connections=[{"a": ("motion_a", "flange"), "b": ("motion_b", "flange")}],
                solver={"solver_type": "euler", "dt": 0.001},
            ),
            ("motion_a", "motion_b", "sources"),
        ),
        (
            "acausal.series_through_sources",
            AcausalGraphSpec(
                physical_domain="translational",
                nodes={
                    "wall": ComponentSpec(type="Ground"),
                    "force_a": ComponentSpec(
                        type="ActuationInput", params={"source_kind": "force"}
                    ),
                    "force_b": ComponentSpec(
                        type="ActuationInput", params={"source_kind": "force"}
                    ),
                },
                connections=[
                    {"a": ("wall", "flange"), "b": ("force_a", "flange")},
                    {"a": ("force_a", "flange"), "b": ("force_b", "flange")},
                ],
                solver={"solver_type": "euler", "dt": 0.001},
            ),
            ("force_a", "force_b", "sources"),
        ),
    ],
)
def test_acausal_diagnostic_catalog_messages_name_offenders(
    code: str,
    graph: AcausalGraphSpec,
    must_include: tuple[str, ...],
) -> None:
    report = _report(graph)
    diagnostic = _diagnostic(report, code)
    haystack = " ".join(
        [
            diagnostic.message,
            " ".join(diagnostic.node_ids),
            " ".join(diagnostic.variables),
            str(diagnostic.counts),
        ]
    )
    for expected in must_include:
        assert expected in haystack


def test_nested_unbalanced_composite_reports_local_offender() -> None:
    local_bad = AcausalGraphSpec(
        physical_domain="translational",
        nodes={"spring": ComponentSpec(type="LinearSpring")},
        connections=[],
        solver={"solver_type": "euler", "dt": 0.001},
    )
    parent = AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="Ground"),
            "composite": ComponentSpec(type="AcausalSystem", input_ports=["flange"]),
            "mass": ComponentSpec(type="Mass"),
        },
        connections=[
            {"a": ("wall", "flange"), "b": ("mass", "flange")},
            {"a": ("composite", "flange"), "b": ("mass", "flange")},
        ],
        subgraphs={"composite": local_bad},
        solver={"solver_type": "euler", "dt": 0.001},
    )

    diagnostic = _diagnostic(_report(parent), "acausal.unbalanced")

    assert "spring" in diagnostic.message
    assert diagnostic.counts is not None
    assert diagnostic.counts["equations"] != diagnostic.counts["unknowns"]


def test_graph_project_drops_old_compile_report_versions() -> None:
    project = GraphProject.model_validate(
        {
            "metadata": {
                "name": "demo",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            "graph": GraphSpec().model_dump(mode="json"),
            "compile_reports": {
                "plant": {
                    "schema_id": "feedbax.spec.domain_compile_report",
                    "schema_version": "feedbax.spec.domain_compile_report.v0",
                    "status": "ok",
                    "interior_content_hash": "abc",
                    "diagnostics": [],
                    "summary": {},
                }
            },
        }
    )

    assert project.compile_reports is None


def test_graph_service_persists_report_and_derives_stale_status(tmp_path: Path) -> None:
    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(GraphSpec(), None)
    interior = _msd_interior()

    report = service.compile_node(
        record.graph_id,
        node_path=["plant"],
        interior=interior,
        component_registry=_registry(),
    )
    loaded = service.get_graph(record.graph_id).project
    cached = (loaded.compile_reports or {})["plant"]

    edited = interior.model_copy(
        update={
            "nodes": {
                **interior.nodes,
                "mass": ComponentSpec(type="Mass", params={"mass": 2.0}),
            }
        }
    )
    assert cached == report
    assert (
        derive_compile_status(
            cached,
            current_interior_hash=acausal_interior_content_hash(edited),
        )
        == "stale"
    )


def test_internal_compile_exception_becomes_error_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import feedbax.contracts.graphs.acausal_compiler as compiler

    def fail_flatten(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom during structural compile")

    monkeypatch.setattr(compiler, "_flatten_acausal_graph", fail_flatten)

    report = _report(_msd_interior())

    diagnostic = _diagnostic(report, "acausal.internal")
    assert report.status == "error"
    assert "boom during structural compile" in diagnostic.message


def test_compile_endpoint_returns_report_and_malformed_body_422(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import feedbax.web.api.graphs as graphs_api

    service = GraphService(storage_dir=tmp_path)
    record = service.create_graph(GraphSpec(), None)
    monkeypatch.setattr(graphs_api, "service", service)

    with TestClient(create_app()) as client:
        response = client.post(
            f"/api/graphs/{record.graph_id}/nodes/compile",
            json={
                "node_path": ["plant"],
                "interior": _msd_interior().model_dump(mode="json"),
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert service.get_graph(record.graph_id).project.compile_reports is not None

        malformed = client.post(
            f"/api/graphs/{record.graph_id}/nodes/compile",
            json={"node_path": ["plant"]},
        )
    assert malformed.status_code == 422
