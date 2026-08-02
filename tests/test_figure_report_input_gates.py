"""Input gates for figure piece bytes and id-addressed figure context.

Each test states one accidental-corruption scenario the gate exists to refuse:
a path-addressed piece whose stored bytes drifted from its declared profile,
two different manifests sharing one identifier inside a single figure, and two
panels sharing one name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from feedbax.analysis.figures import (
    FigureInputAuthorityError,
    FigureSpecExecutionError,
    ResolvedFigureInput,
    _figure_expression_context,
    _piece_data,
    execute_figure_spec,
    figure_manifest_plotly_json,
)
from feedbax.contracts.figures import FigurePiece, FigureSpec
from feedbax.contracts.manifest import ArtifactRef, ParentRef
from feedbax.persistence.artifact_custody import ArtifactBlobIntegrityError
from feedbax.plot.constructors import (
    FigureRegistry,
    register_default_figure_constructors,
    register_figure_piece,
)

pytestmark = [pytest.mark.feedbax_contract]

_PIECE_PAYLOAD = {"payload": {"x": [0, 1, 2], "y": [[1, 2, 3], [2, 3, 4]]}}


@pytest.fixture
def figure_registry() -> FigureRegistry:
    registry = FigureRegistry()
    register_default_figure_constructors(registry)
    return registry


def _piece(
    name: str,
    uri: Path,
    *,
    sha256: str | None,
    size_bytes: int | None,
) -> FigurePiece:
    return FigurePiece(
        name=name,
        description="Path-addressed piece under test",
        artifact_ref=ArtifactRef(
            role="figure_piece",
            logical_name="piece.json",
            media_type="application/json",
            uri=str(uri),
            sha256=sha256,
            size_bytes=size_bytes,
        ),
        data_path="payload",
        label="Piece",
        constructor="feedbax.profile_band",
    )


def _piece_spec(name: str, piece_name: str) -> FigureSpec:
    return FigureSpec(name=name, assembler="feedbax.grid_figure", pieces=[piece_name])


def test_path_addressed_piece_refuses_bytes_that_drifted_from_declared_digest(
    tmp_path: Path, figure_registry
) -> None:
    """The declared digest is the only authentication a path-addressed piece has."""
    payload_path = tmp_path / "piece.json"
    authored = json.dumps(_PIECE_PAYLOAD).encode("utf-8")
    payload_path.write_bytes(authored)
    piece = _piece(
        "feedbax.test_drifted_piece",
        payload_path,
        sha256=hashlib.sha256(authored).hexdigest(),
        size_bytes=None,
    )
    register_figure_piece(piece, registry=figure_registry)

    # The location still resolves; only the bytes stored there changed.
    replaced = json.dumps({"payload": {"x": [9, 9, 9], "y": [[0, 0, 0], [1, 1, 1]]}})
    payload_path.write_text(replaced, encoding="utf-8")

    with pytest.raises(FigureSpecExecutionError) as excinfo:
        execute_figure_spec(
            _piece_spec("drifted-piece", piece.name),
            root=tmp_path,
            registry=figure_registry,
        )

    cause = excinfo.value.__cause__
    assert isinstance(cause, ArtifactBlobIntegrityError)
    assert "sha256 mismatch" in str(cause)
    assert excinfo.value.manifest.status == "failed"


def test_path_addressed_piece_refuses_bytes_that_drifted_from_declared_size(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "piece.json"
    authored = json.dumps(_PIECE_PAYLOAD).encode("utf-8")
    payload_path.write_bytes(authored)
    piece = _piece(
        "feedbax.test_sized_piece",
        payload_path,
        sha256=None,
        size_bytes=len(authored) + 1,
    )

    with pytest.raises(ArtifactBlobIntegrityError, match="size mismatch"):
        _piece_data(piece, tmp_path)


def test_path_addressed_piece_accepts_bytes_matching_its_declared_profile(
    tmp_path: Path, figure_registry
) -> None:
    payload_path = tmp_path / "piece.json"
    authored = json.dumps(_PIECE_PAYLOAD).encode("utf-8")
    payload_path.write_bytes(authored)
    piece = _piece(
        "feedbax.test_profiled_piece",
        payload_path,
        sha256=hashlib.sha256(authored).hexdigest(),
        size_bytes=len(authored),
    )
    register_figure_piece(piece, registry=figure_registry)

    manifest, _path = execute_figure_spec(
        _piece_spec("profiled-piece", piece.name),
        root=tmp_path,
        registry=figure_registry,
    )

    assert manifest.status == "completed"
    assert figure_manifest_plotly_json(manifest) is not None


def test_path_addressed_piece_without_a_declared_profile_is_unchanged(tmp_path: Path) -> None:
    """Verification checks a claim; it does not invent one for a bare locator."""
    payload_path = tmp_path / "piece.json"
    payload_path.write_text(json.dumps(_PIECE_PAYLOAD), encoding="utf-8")
    piece = _piece("feedbax.test_bare_piece", payload_path, sha256=None, size_bytes=None)

    assert _piece_data(piece, tmp_path) == _PIECE_PAYLOAD["payload"]


def _resolved_input(kind: str, manifest_id: str, role: str) -> ResolvedFigureInput:
    return ResolvedFigureInput(
        ref=ParentRef(kind=kind, id=manifest_id, role=role),
        manifest=None,
        path=None,
    )


def test_figure_expression_context_refuses_two_manifests_sharing_one_id() -> None:
    spec = FigureSpec(name="duplicate-manifest-id", assembler="feedbax.grid_figure")
    resolved = [
        _resolved_input("AnalysisRunManifest", "feedbax-shared:id", "analysis_run"),
        _resolved_input("EvaluationRunManifest", "feedbax-shared:id", "evaluation_run"),
    ]

    with pytest.raises(FigureInputAuthorityError, match="two different manifests"):
        _figure_expression_context(spec, resolved)


def test_figure_expression_context_admits_one_manifest_bound_under_two_roles() -> None:
    """The same manifest twice is not ambiguous; the id still names one record."""
    spec = FigureSpec(name="repeated-manifest-id", assembler="feedbax.grid_figure")
    resolved = [
        _resolved_input("AnalysisRunManifest", "feedbax-shared:id", "baseline"),
        _resolved_input("AnalysisRunManifest", "feedbax-shared:id", "comparison"),
    ]

    context = _figure_expression_context(spec, resolved)

    item = context.items["manifest:feedbax-shared:id"]
    assert item.payload["kind"] == "AnalysisRunManifest"
    assert context.items["baseline"].payload["id"] == "feedbax-shared:id"
    assert context.items["comparison"].payload["id"] == "feedbax-shared:id"


def test_execute_figure_spec_refuses_duplicate_panel_names(
    tmp_path: Path, figure_registry
) -> None:
    spec = FigureSpec(
        name="duplicate-panel-names",
        assembler="feedbax.grid_figure",
        panels=[
            {"name": "left", "row": 1, "col": 1},
            {"name": "left", "row": 1, "col": 2},
        ],
        traces=[
            {
                "name": "left-profile",
                "constructor": "feedbax.profile_band",
                "panel": "left",
                "data": {"y": [[1, 2, 3], [2, 3, 4]]},
            }
        ],
    )

    with pytest.raises(FigureSpecExecutionError) as excinfo:
        execute_figure_spec(spec, root=tmp_path, registry=figure_registry)

    cause = excinfo.value.__cause__
    assert isinstance(cause, ValueError)
    assert "duplicate panel names: ['left']" in str(cause)
