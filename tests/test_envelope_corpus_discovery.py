"""The corpus as a corpus: recursive discovery, one output address, contained aliases.

Three rules meet here, and they only make sense together. An envelope's alias is
its path stem, so a project may file its envelopes in subdirectories — which is
only true if the engine *enumerates* subdirectories, because the enumeration is
what the choke point compares the tracked tree against. A subdirectory makes two
envelopes with one basename natural, so the authored ``name`` that addresses both
compiled outputs has to be proved unique across the whole corpus before anything
is written. And a path-shaped alias is exactly where a directory traversal hides,
so containment is proved on the joined path rather than on the authored string.

Everything here runs against ``quillon``, the invented fake project, and writes
only inside ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.__main__ import main
from feedbax.contracts.experiment_envelope import (
    ExperimentEnvelopeRejection,
    ExperimentEnvelopeRejectionCategory,
)
from feedbax.contracts.experiment_envelope_dialect import ExperimentEnvelopeLayer
from feedbax.envelope import (
    ChokeFinding,
    DuplicateOutputAddressError,
    EnvelopeLayout,
    UncontainedEnvelopeAliasError,
    compare_tracked_outputs,
    kernel_for,
)

from tests.fake_project_experiment import (
    ENVELOPE_DIRECTORY,
    OUTPUT_DIRECTORY,
    PROJECT_DECLARATION,
    TRAINING_ENVELOPE,
    envelope_path,
    write_envelope,
    write_repo,
)

#: The subdirectory a namespaced corpus files part of itself in.
STUDY = "sisu"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    write_repo(tmp_path)
    return tmp_path


def kernel() -> Any:
    """Return the one compiler bound to the fake project's data declaration."""
    return kernel_for(PROJECT_DECLARATION)


def layout() -> EnvelopeLayout:
    return EnvelopeLayout.of(PROJECT_DECLARATION)


# -- helpers -------------------------------------------------------------------


def _read(repo: Path, ref: str) -> dict[str, Any]:
    return json.loads((repo / ref).read_text(encoding="utf-8"))


def _nest(repo: Path, alias: str, study: str) -> Path:
    """Move one authored envelope into a subdirectory, keeping its bytes."""
    source = envelope_path(repo, alias)
    target = repo / ENVELOPE_DIRECTORY / study / f"{alias}.envelope.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    source.unlink()
    return target


def _repoint_subject(repo: Path, alias: str, subject_alias: str) -> None:
    """Point one evaluation envelope's cross-layer subject at another alias."""
    document = json.loads(envelope_path(repo, alias).read_text(encoding="utf-8"))
    document["evaluation"] = {
        **document["evaluation"],
        "subject": {"kind": "envelope", "alias": subject_alias},
    }
    write_envelope(envelope_path(repo, alias), document)


def _regenerate(repo: Path) -> None:
    compiler = kernel()
    out_dir = repo / OUTPUT_DIRECTORY
    for path in compiler.envelopes(repo):
        compiler.write_outputs(compiler.compile_envelope_file(path, repo_root=repo), out_dir)


def _refs(paths: list[Path], repo: Path) -> list[str]:
    return [path.relative_to(repo).as_posix() for path in paths]


def _plant_envelope_in_output_directory(repo: Path, name: str) -> str:
    """Write a compilable envelope *inside the output directory*.

    Nothing legitimate does this. It exists so an escaping alias has something
    real to reach: the hole this closes is only observable when the target the
    traversal names actually resolves.
    """
    document = {**TRAINING_ENVELOPE, "name": name}
    ref = f"{OUTPUT_DIRECTORY}/{name}.envelope.json"
    write_envelope(repo / ref, document)
    return ref


# -- recursive discovery -------------------------------------------------------


def test_a_flat_corpus_enumerates_exactly_as_a_flat_glob_does(repo: Path) -> None:
    """The behaviour every landed flat corpus depends on is unchanged."""
    directory = repo / ENVELOPE_DIRECTORY

    enumerated = kernel().envelopes(repo)

    assert enumerated == sorted(directory.glob("*.envelope.json"))
    assert enumerated == sorted(directory.glob("*.envelope.json"), key=lambda p: p.as_posix())
    assert len(enumerated) == 6


def test_an_envelope_in_a_subdirectory_is_enumerated(repo: Path) -> None:
    nested = _nest(repo, "widened", STUDY)

    enumerated = kernel().envelopes(repo)

    assert nested in enumerated
    assert len(enumerated) == 6


def test_enumeration_is_deterministic_repo_relative_path_order(repo: Path) -> None:
    _nest(repo, "widened", STUDY)
    _nest(repo, "widened-plot", "wave1")

    enumerated = _refs(kernel().envelopes(repo), repo)

    assert enumerated == sorted(enumerated)
    assert enumerated == _refs(kernel().envelopes(repo), repo)
    assert f"{ENVELOPE_DIRECTORY}/{STUDY}/widened.envelope.json" in enumerated
    assert f"{ENVELOPE_DIRECTORY}/wave1/widened-plot.envelope.json" in enumerated


def test_a_directory_holding_no_envelopes_contributes_nothing(repo: Path) -> None:
    """Only the authored suffix is discovered, at any depth."""
    (repo / ENVELOPE_DIRECTORY / STUDY).mkdir(parents=True)
    (repo / ENVELOPE_DIRECTORY / STUDY / "README.md").write_text("notes\n", encoding="utf-8")
    (repo / ENVELOPE_DIRECTORY / STUDY / "budgets.json").write_text("{}\n", encoding="utf-8")

    assert len(kernel().envelopes(repo)) == 6


def test_a_nested_envelope_compiles_and_the_choke_point_sees_its_outputs(repo: Path) -> None:
    """End to end: nested envelope, nested alias, flat outputs, clean choke pass."""
    _nest(repo, "widened", STUDY)
    _repoint_subject(repo, "widened-probe", f"{STUDY}/widened")

    _regenerate(repo)
    report = compare_tracked_outputs(kernel(), repo)

    assert report.ok
    assert report.by_finding(ChokeFinding.ORPHANED) == ()
    assert len(report.by_finding(ChokeFinding.IDENTICAL)) == 12
    # Outputs stay flat and keep their name-derived addresses: the envelope's
    # directory namespaces the alias that reaches it, not the address it writes.
    assert (repo / OUTPUT_DIRECTORY / "widened.training_run_matrix.json").is_file()
    assert not (repo / OUTPUT_DIRECTORY / STUDY).exists()
    lock = _read(repo, f"{OUTPUT_DIRECTORY}/widened.compile-lock.json")
    assert lock["envelope"]["ref"] == f"{ENVELOPE_DIRECTORY}/{STUDY}/widened.envelope.json"
    # The nested alias is what the dependant envelope's lock records reaching.
    probe = _read(repo, f"{OUTPUT_DIRECTORY}/widened-probe.compile-lock.json")
    assert any(
        reference.get("envelope_ref")
        == f"{ENVELOPE_DIRECTORY}/{STUDY}/widened.envelope.json"
        for reference in probe["references"]
    )


def test_recursion_is_what_keeps_a_nested_corpus_visible(repo: Path) -> None:
    """Why discovery has to recurse, stated as the failure it prevents.

    A flat enumeration returns *nothing* for a wholly nested corpus, so every
    tracked output would be unclaimed: the choke point would report the entire
    output directory as orphaned while checking nothing at all.
    """
    for alias in ("widened", "widened-probe", "widened-summary"):
        _nest(repo, alias, STUDY)
    _repoint_subject(repo, f"{STUDY}/widened-probe", f"{STUDY}/widened")

    flat = sorted((repo / ENVELOPE_DIRECTORY).glob("*.envelope.json"))
    enumerated = kernel().envelopes(repo)

    assert len(flat) == 3
    assert len(enumerated) == 6
    assert set(flat) < set(enumerated)


# -- one output address per name -----------------------------------------------


def _duplicate(repo: Path, ref: str, name: str = "widened") -> str:
    """File a second envelope stating an existing envelope's authored name."""
    document = {**TRAINING_ENVELOPE, "name": name}
    write_envelope(repo / ref, document)
    return ref


def test_two_root_envelopes_stating_one_name_are_refused(repo: Path) -> None:
    ref = _duplicate(repo, f"{ENVELOPE_DIRECTORY}/copied.envelope.json")

    with pytest.raises(DuplicateOutputAddressError) as excinfo:
        kernel().refuse_duplicate_output_addresses(repo)

    rejection = excinfo.value
    assert rejection.category is ExperimentEnvelopeRejectionCategory.DUPLICATE_KEY
    assert rejection.name == "widened"
    assert set(rejection.envelope_refs) == {
        ref,
        f"{ENVELOPE_DIRECTORY}/widened.envelope.json",
    }
    assert rejection.output_path == f"{OUTPUT_DIRECTORY}/widened.compile-lock.json"
    assert rejection.output_path in str(rejection)
    for envelope_ref in rejection.envelope_refs:
        assert envelope_ref in str(rejection)


def test_a_nested_envelope_colliding_with_a_root_one_is_refused(repo: Path) -> None:
    """The collision directory namespacing makes reachable: same name, two homes."""
    ref = _duplicate(repo, f"{ENVELOPE_DIRECTORY}/{STUDY}/widened.envelope.json")

    with pytest.raises(DuplicateOutputAddressError) as excinfo:
        kernel().refuse_duplicate_output_addresses(repo)

    assert ref in excinfo.value.envelope_refs
    assert excinfo.value.output_path == f"{OUTPUT_DIRECTORY}/widened.compile-lock.json"


def test_two_nested_envelopes_in_different_studies_collide_on_the_flat_address(
    repo: Path,
) -> None:
    _nest(repo, "widened", STUDY)
    _duplicate(repo, f"{ENVELOPE_DIRECTORY}/wave1/widened.envelope.json")

    with pytest.raises(DuplicateOutputAddressError) as excinfo:
        kernel().refuse_duplicate_output_addresses(repo)

    assert set(excinfo.value.envelope_refs) == {
        f"{ENVELOPE_DIRECTORY}/{STUDY}/widened.envelope.json",
        f"{ENVELOPE_DIRECTORY}/wave1/widened.envelope.json",
    }


def test_distinct_names_in_one_directory_tree_are_admitted(repo: Path) -> None:
    _nest(repo, "widened", STUDY)
    _duplicate(repo, f"{ENVELOPE_DIRECTORY}/wave1/other.envelope.json", name="widened-other")

    kernel().refuse_duplicate_output_addresses(repo)

    claims = kernel().output_claims(repo)
    assert claims["widened"] == (f"{ENVELOPE_DIRECTORY}/{STUDY}/widened.envelope.json",)
    assert claims["widened-other"] == (f"{ENVELOPE_DIRECTORY}/wave1/other.envelope.json",)


def test_an_unreadable_envelope_is_not_reported_as_a_collision(repo: Path) -> None:
    """A broken envelope fails its own compile; it is not a corpus collision."""
    (repo / ENVELOPE_DIRECTORY / "broken.envelope.json").write_text("{ not json", encoding="utf-8")
    (repo / ENVELOPE_DIRECTORY / "nameless.envelope.json").write_text("{}\n", encoding="utf-8")

    kernel().refuse_duplicate_output_addresses(repo)

    assert f"{ENVELOPE_DIRECTORY}/broken.envelope.json" not in {
        ref for refs in kernel().output_claims(repo).values() for ref in refs
    }


def test_the_choke_point_refuses_a_colliding_corpus_rather_than_reporting_it(
    repo: Path,
) -> None:
    _regenerate(repo)
    _duplicate(repo, f"{ENVELOPE_DIRECTORY}/{STUDY}/widened.envelope.json")

    with pytest.raises(DuplicateOutputAddressError):
        compare_tracked_outputs(kernel(), repo)


def test_the_duplicate_refusal_fires_before_any_output_is_written(repo: Path) -> None:
    """The whole point: no compiled byte is written on the way to the refusal."""
    _duplicate(repo, f"{ENVELOPE_DIRECTORY}/{STUDY}/widened.envelope.json")

    code = main(
        [
            "preflight-experiment-envelope",
            str(envelope_path(repo, "widened-summary")),
            "--repo-root",
            str(repo),
            "--out-dir",
            OUTPUT_DIRECTORY,
        ]
    )

    assert code == 2
    assert not (repo / OUTPUT_DIRECTORY).exists()


def test_an_uncolliding_corpus_still_compiles_through_the_entrypoint(repo: Path) -> None:
    code = main(
        [
            "preflight-experiment-envelope",
            str(envelope_path(repo, "widened")),
            "--repo-root",
            str(repo),
            "--out-dir",
            OUTPUT_DIRECTORY,
        ]
    )

    assert code == 0
    assert (repo / OUTPUT_DIRECTORY / "widened.compile-lock.json").is_file()


# -- alias containment ---------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "reason"),
    [
        ("../compiled/widened", "'..' segment"),
        ("../../escape", "repeated '..' segments"),
        (f"{STUDY}/../../escape", "'..' after a legitimate segment"),
        ("/absolute/widened", "absolute path"),
        (f"{STUDY}\\widened", "backslash separator"),
        (f"{STUDY}//widened", "empty segment"),
        (f"{STUDY}/", "trailing separator"),
        ("./widened", "'.' segment"),
        (" widened", "leading whitespace"),
        ("widened ", "trailing whitespace"),
        ("", "empty alias"),
    ],
)
def test_an_alias_that_is_not_a_contained_stem_is_refused(alias: str, reason: str) -> None:
    with pytest.raises(UncontainedEnvelopeAliasError) as excinfo:
        layout().alias_ref(alias, field="envelope.base")

    rejection = excinfo.value
    assert rejection.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert rejection.alias == alias
    assert rejection.envelope_directory == ENVELOPE_DIRECTORY
    assert rejection.field == "envelope.base"
    assert isinstance(rejection, ExperimentEnvelopeRejection)
    assert reason  # the case's own label, kept so a failure names which form broke


@pytest.mark.parametrize(
    "alias", ["widened", f"{STUDY}/widened", f"{STUDY}/wave1/leaf", "a.b/c-d_e"]
)
def test_a_contained_alias_joins_to_a_path_inside_the_envelope_directory(alias: str) -> None:
    ref = layout().alias_ref(alias)

    assert ref == f"{ENVELOPE_DIRECTORY}/{alias}.envelope.json"
    assert ref.startswith(f"{ENVELOPE_DIRECTORY}/")
    assert ".." not in ref.split("/")


def test_an_escaping_base_alias_cannot_reach_a_document_in_the_output_directory(
    repo: Path,
) -> None:
    """The hole this closes: the raw string evades the compiled-output rule.

    ``refuse_compiled_output_base`` tests the base as written, and
    ``../compiled/sneaky`` is not under ``compiled/`` as written — it only
    becomes so once joined. With a real envelope planted at the escaped address,
    the traversal resolves, so this is the difference between refusing and
    inheriting from the engine's own output directory.
    """
    _plant_envelope_in_output_directory(repo, "sneaky")

    with pytest.raises(UncontainedEnvelopeAliasError) as excinfo:
        kernel().resolve_parent(
            repo,
            f"../{OUTPUT_DIRECTORY}/sneaky",
            (),
            expected_layer=ExperimentEnvelopeLayer.TRAINING,
        )

    assert excinfo.value.resolved is None or ".." not in str(excinfo.value.resolved)


def test_an_escaping_cross_layer_alias_is_refused_at_compile(repo: Path) -> None:
    _plant_envelope_in_output_directory(repo, "sneaky")
    _repoint_subject(repo, "widened-probe", f"../{OUTPUT_DIRECTORY}/sneaky")

    with pytest.raises(UncontainedEnvelopeAliasError) as excinfo:
        kernel().compile_envelope_file(envelope_path(repo, "widened-probe"), repo_root=repo)

    assert excinfo.value.category is ExperimentEnvelopeRejectionCategory.INVALID_VALUE
    assert excinfo.value.alias == f"../{OUTPUT_DIRECTORY}/sneaky"


def test_a_legitimate_nested_alias_still_resolves_as_a_parent(repo: Path) -> None:
    """Containment refuses traversal without refusing the nesting it protects."""
    _nest(repo, "widened", STUDY)

    parent = kernel().resolve_parent(
        repo, f"{STUDY}/widened", (), expected_layer=ExperimentEnvelopeLayer.TRAINING
    )

    assert parent.ref == f"{ENVELOPE_DIRECTORY}/{STUDY}/widened.envelope.json"
    assert parent.kind == "envelope_alias"
