"""The public cold-start conformance fixture.

This is the honesty gate for the whole upstreaming program. Every other test in
this repository imports Feedbax from the source tree it is testing, so none of
them can tell whether a *project* — a directory with literally nothing in it —
can reach the framework at all. This one builds the wheel, installs it into an
environment with no ``PYTHONPATH`` and no source checkout, and then does the
entire cold start through that installation and nothing else:

1. ``feedbax init`` in an empty directory, and the exact inventory it creates;
2. tens of lines of hand-authored content — one science plugin, one base spec,
   one envelope varying it — and nothing else;
3. ``feedbax preflight-experiment-envelope`` to compile, the public planning
   API to derive the plan, and ``feedbax fulfill-experiment-envelope`` to run it;
4. ``feedbax check-project-science-surface`` to prove the project grew no
   machinery, read from a ratified baseline ref rather than the working tree.

The project vocabulary is invented. ``vantry`` names nothing and belongs to
nobody: if any of this needed a real project's words, the framework would not be
generic and the fixture would be measuring the wrong thing.

The wheel build requires a clean Git checkout — Feedbax refuses to stamp
distribution provenance onto uncommitted bytes — so this module, like
``test_feedbax_wheel_provenance``, asserts that before it starts. Installing the
wheel's dependencies needs either a network or a warm ``uv`` cache; there is no
way to prove a clean-environment install without paying for one.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tomllib
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from feedbax.contracts.experiment_compile_lock import (
    EXPERIMENT_COMPILE_LOCK_SCHEMA_ID,
    EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION,
)
from feedbax.contracts.experiment_envelope_dialect import (
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
)
from feedbax.contracts.project_experiment import (
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID,
    PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION,
)
from feedbax.governance.project_init import (
    AUTHORING_BUDGET_SCHEMA_ID,
    AUTHORING_BUDGET_SCHEMA_VERSION,
    DEFAULT_AUTHORING_BUDGET_LAYERS,
)
from feedbax.governance.science_surface import POLICY_SCHEMA_VERSION

pytestmark = [pytest.mark.slow, pytest.mark.feedbax_contract]

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The template a cold start cannot happen without: the agent instructions are
#: loaded from package resources, so an absent one is an unusable wheel.
TEMPLATE_MEMBER = "feedbax/governance/templates/agent_instructions.v1.md"

#: Extras the installed framework needs to run its own CLI. The bare wheel's
#: declared dependencies are enough for ``feedbax init`` but not for
#: ``feedbax --help``, because the engine parser imports the plugin bootstrap,
#: which imports ``feedbax.analysis``, which imports plotly at module scope.
#: That is a packaging defect in the framework, recorded here rather than
#: hidden: a project that pins what ``feedbax init`` writes into its
#: ``pyproject.toml`` cannot run the commands this fixture runs.
INSTALL_EXTRAS = "analysis"

PROJECT_NAME = "vantry-study"
PACKAGE_NAME = "vantry_study"

#: Exactly what ``feedbax init`` is responsible for in an empty directory.
EXPECTED_INIT_INVENTORY: tuple[str, ...] = (
    "AGENTS.md",
    "CLAUDE.md",
    "ci/feedbax-science-surface.toml",
    "feedbax.project.json",
    "pyproject.toml",
    "specs/experiment/authoring_budget.json",
    f"src/{PACKAGE_NAME}/__init__.py",
)

#: Where the cold-start author puts the three things they write.
SCIENCE_REF = f"src/{PACKAGE_NAME}/science.py"
BASE_REF = "specs/base/tally.evaluation_run_matrix.json"
ENVELOPE_REF = "specs/experiment/wide-tally.envelope.json"

#: The whole cold start must fit inside the budget the framework itself states
#: for one authored evaluation document. Three hand-written files that together
#: exceed what one envelope may be is not a cold start in tens of lines.
COLD_START_LINE_BUDGET: int = DEFAULT_AUTHORING_BUDGET_LAYERS["evaluation"]["max_lines"]

# --------------------------------------------------------------------------
# The three things a cold-start author writes, and nothing else
# --------------------------------------------------------------------------

SCIENCE_SOURCE = '''"""The one thing this project owns: a tally over an authored span."""

from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.contracts.manifest import store_bytes_artifact
from feedbax.plugins import (
    EVALUATION_RECIPES,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
)

TALLY = "vantry.tally"


def tally(run_spec, root, states_path, execution_context):
    """Count the squares across the span this run was authored with."""
    span = int(run_spec.params["span"])
    artifact = store_bytes_artifact(
        "".join(f"{step * step}\\n" for step in range(span)).encode(),
        root=root,
        role="evaluation_states",
        logical_name="tally.txt",
    )
    return EvaluationRecipeResult(
        states=None,
        summary_metrics={"span": span},
        artifacts=[artifact],
        metadata={"states_schema": "vantry.tally.v1"},
    )


PLUGIN_REGISTRATION = PluginRegistration(
    PluginDeclaration(
        "vantry_study.science", "1.0", 1,
        families=(FamilyRequirement("evaluation_recipes"),),
    ),
    lambda context: context.registry(EVALUATION_RECIPES).register(TALLY, tally),
)
'''

BASE_SOURCE = """{
  "schema_id": "feedbax.spec.evaluation_run_matrix",
  "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
  "base": {
    "schema_id": "feedbax.spec.evaluation_run",
    "schema_version": "feedbax.spec.evaluation_run.v1",
    "evaluation_type": "vantry.tally",
    "params": {"span": 4}
  },
  "rows": [{"row_id": "baseline"}]
}
"""

ENVELOPE_SOURCE = """{
  "schema": "feedbax.experiment_envelope.v2",
  "name": "wide-tally",
  "base": "specs/base/tally.evaluation_run_matrix.json",
  "reason": "The four-step span closes before the tally stops growing.",
  "assert": [{"path": "base.evaluation_type", "equals": "vantry.tally"}],
  "evaluation": {
    "subject": {"kind": "not_applicable", "reason": "a tally reads only its own span"},
    "subject_id": "wide",
    "delta": {
      "layer_id": "wide-span",
      "patches": [{"path": "base.params.span", "op": "replace", "value": 9}]
    }
  }
}
"""

AUTHORED_SOURCES: dict[str, str] = {
    SCIENCE_REF: SCIENCE_SOURCE,
    BASE_REF: BASE_SOURCE,
    ENVELOPE_REF: ENVELOPE_SOURCE,
}

#: The science entry the project's owner ratifies for the one file above. The
#: policy `feedbax init` writes authorizes the package marker only, so this is
#: the whole of what a cold start adds to its own science surface.
RATIFIED_SCIENCE_ENTRY = f"""
[[allowed_file]]
path = "{SCIENCE_REF}"
symbols = ["TALLY", "tally", "PLUGIN_REGISTRATION"]
reason = "the one science this project owns"
"""

BASELINE_REF = "baseline"


# --------------------------------------------------------------------------
# Running things outside this checkout
# --------------------------------------------------------------------------


def _clean_env(**extra: str) -> dict[str, str]:
    """Return an environment with every path back into this checkout removed."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"}
    }
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
            **extra,
        }
    )
    return env


def _run(
    args: Sequence[str | Path], *, cwd: Path, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        env=_clean_env(),
    )


def _git(args: Sequence[str | Path], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "git",
            "-c",
            "user.email=cold-start@example.invalid",
            "-c",
            "user.name=Cold Start",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
    )


def _git_source_requirements() -> list[str]:
    """Return direct-reference requirements for every Git-sourced dependency.

    A wheel installed from a file cannot see ``[tool.uv.sources]``, so the Git
    dependencies the project declares there have to be supplied to the resolver
    explicitly. Reading them out of the same table keeps this from drifting into
    a second, stale pin.
    """
    document = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sources = document.get("tool", {}).get("uv", {}).get("sources", {})
    requirements: list[str] = []
    for name, source in sorted(sources.items()):
        if isinstance(source, dict) and "git" in source:
            rev = source.get("rev")
            suffix = f"@{rev}" if rev else ""
            requirements.append(f"{name} @ git+{source['git']}{suffix}")
    return requirements


def _tree_digest(root: Path) -> dict[str, str]:
    """Return one digest per entry below *root*, symlinks included as links."""
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            digests[relative] = f"link:{os.readlink(path)}"
        elif path.is_file():
            digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


# --------------------------------------------------------------------------
# The installed framework
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InstalledFramework:
    """One Feedbax wheel, installed where no source checkout is reachable."""

    wheel: Path
    environment: Path

    @property
    def feedbax(self) -> Path:
        return self.environment / "bin" / "feedbax"

    @property
    def python(self) -> Path:
        return self.environment / "bin" / "python"

    def cli(
        self, args: Sequence[str | Path], *, cwd: Path, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Run one ``feedbax`` subcommand through the installed console script."""
        return _run([self.feedbax, *args], cwd=cwd, check=check)


@pytest.fixture(scope="module")
def framework(tmp_path_factory: pytest.TempPathFactory) -> InstalledFramework:
    """Build the wheel and install it into an environment with no source path."""
    root = tmp_path_factory.mktemp("feedbax-cold-start")
    status = _run(["git", "status", "--porcelain=v1"], cwd=REPO_ROOT).stdout
    assert status == "", (
        "the cold-start fixture builds a real wheel, and Feedbax refuses to stamp "
        f"distribution provenance onto an uncommitted checkout:\n{status}"
    )

    wheel_dir = root / "wheel"
    _run(
        ["uv", "build", "--wheel", "--directory", REPO_ROOT, "--out-dir", wheel_dir],
        cwd=root,
    )
    wheel = next(wheel_dir.glob("feedbax-*.whl"))

    environment = root / "environment"
    _run(
        [
            "uv",
            "venv",
            "--python",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            environment,
        ],
        cwd=root,
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            environment / "bin" / "python",
            f"feedbax[{INSTALL_EXTRAS}] @ {wheel}",
            *_git_source_requirements(),
        ],
        cwd=root,
    )

    installed = InstalledFramework(wheel=wheel, environment=environment)
    package_root = Path(
        _run(
            [
                installed.python,
                "-c",
                "import pathlib, feedbax; print(pathlib.Path(feedbax.__file__).parent)",
            ],
            cwd=root,
        ).stdout.strip()
    )
    assert package_root.is_relative_to(environment), (
        f"the installed framework resolved to {package_root}, which is not the wheel"
    )
    assert not package_root.is_relative_to(REPO_ROOT)
    return installed


@dataclass(frozen=True)
class ColdStartProject:
    """One project initialized and authored entirely through the installation."""

    root: Path
    framework: InstalledFramework
    init_stdout: str

    def cli(
        self, args: Sequence[str | Path], *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return self.framework.cli(args, cwd=self.root, check=check)

    @property
    def declaration(self) -> dict[str, object]:
        return json.loads((self.root / "feedbax.project.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def initialized(
    framework: InstalledFramework, tmp_path_factory: pytest.TempPathFactory
) -> ColdStartProject:
    """Run ``feedbax init`` once in a genuinely empty directory."""
    root = tmp_path_factory.mktemp("cold-start-init") / PROJECT_NAME
    root.mkdir()
    completed = framework.cli(["init", "."], cwd=root)
    return ColdStartProject(root=root, framework=framework, init_stdout=completed.stdout)


@pytest.fixture(scope="module")
def authored(
    framework: InstalledFramework, tmp_path_factory: pytest.TempPathFactory
) -> ColdStartProject:
    """Initialize a project, author the cold-start content, and ratify it.

    Separate from :func:`initialized` on purpose: the inventory assertions must
    see a project nobody has added anything to.
    """
    root = tmp_path_factory.mktemp("cold-start-authored") / PROJECT_NAME
    root.mkdir()
    completed = framework.cli(["init", "."], cwd=root)
    for relative, source in AUTHORED_SOURCES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    policy = root / "ci" / "feedbax-science-surface.toml"
    policy.write_text(policy.read_text(encoding="utf-8") + RATIFIED_SCIENCE_ENTRY, encoding="utf-8")
    _git(["init", "-q", "-b", BASELINE_REF, "."], cwd=root)
    _git(["add", "-A"], cwd=root)
    _git(["commit", "-q", "-m", "cold start"], cwd=root)
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            framework.python,
            "--no-deps",
            root,
        ],
        cwd=root.parent,
    )
    return ColdStartProject(root=root, framework=framework, init_stdout=completed.stdout)


# --------------------------------------------------------------------------
# The wheel is the whole framework surface
# --------------------------------------------------------------------------


def test_the_wheel_ships_the_agent_instruction_template(
    framework: InstalledFramework,
) -> None:
    """The instruction block is loaded from package resources, so it must ship."""
    with zipfile.ZipFile(framework.wheel) as archive:
        members = set(archive.namelist())
        assert TEMPLATE_MEMBER in members
        template = archive.read(TEMPLATE_MEMBER).decode("utf-8")
    assert template.strip()
    assert template == (
        REPO_ROOT / "feedbax" / "governance" / "templates" / "agent_instructions.v1.md"
    ).read_text(encoding="utf-8")
    assert framework.feedbax.is_file(), "the wheel installs no `feedbax` console script"


# --------------------------------------------------------------------------
# `feedbax init` is the tested entrance
# --------------------------------------------------------------------------


def test_init_creates_exactly_the_seven_entries(initialized: ColdStartProject) -> None:
    """An empty directory gets seven entries, and no eighth."""
    created = sorted(
        path.relative_to(initialized.root).as_posix()
        for path in initialized.root.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    assert created == sorted(EXPECTED_INIT_INVENTORY)
    assert (initialized.root / "CLAUDE.md").is_symlink()
    assert os.readlink(initialized.root / "CLAUDE.md") == "AGENTS.md"
    assert not (initialized.root / "generated").exists()
    assert not (initialized.root / ".git").exists()


def test_init_states_the_schema_identity_of_everything_durable_it_writes(
    initialized: ColdStartProject,
) -> None:
    declaration = initialized.declaration
    assert declaration["schema_id"] == PROJECT_EXPERIMENT_DECLARATION_SCHEMA_ID
    assert declaration["schema_version"] == PROJECT_EXPERIMENT_DECLARATION_SCHEMA_VERSION
    assert declaration["project"] == PROJECT_NAME
    assert declaration["envelope_directory"] == "specs/experiment"
    assert declaration["output_directory"] == "generated"
    assert declaration["authoring_budget"] == "specs/experiment/authoring_budget.json"

    budget = json.loads(
        (initialized.root / "specs/experiment/authoring_budget.json").read_text(encoding="utf-8")
    )
    assert budget["schema_id"] == AUTHORING_BUDGET_SCHEMA_ID
    assert budget["schema_version"] == AUTHORING_BUDGET_SCHEMA_VERSION
    assert set(budget["layers"]) == set(DEFAULT_AUTHORING_BUDGET_LAYERS)

    policy = (initialized.root / "ci/feedbax-science-surface.toml").read_text(encoding="utf-8")
    assert f"schema_version = {POLICY_SCHEMA_VERSION}" in policy


def test_init_says_out_loud_that_the_gate_is_not_yet_authoritative(
    initialized: ColdStartProject,
) -> None:
    """A green first run that meant nothing would be worse than no run at all."""
    assert "is not authoritative until it is committed and ratified" in (initialized.init_stdout)


def test_a_second_init_and_instructions_install_change_nothing(
    framework: InstalledFramework, tmp_path: Path
) -> None:
    root = tmp_path / PROJECT_NAME
    root.mkdir()
    framework.cli(["init", "."], cwd=root)
    before = _tree_digest(root)

    framework.cli(["init", "."], cwd=root)
    framework.cli(["instructions", "install", "."], cwd=root)

    assert _tree_digest(root) == before
    assert framework.cli(["instructions", "check", "."], cwd=root).returncode == 0


# --------------------------------------------------------------------------
# Tens of lines, and no machinery
# --------------------------------------------------------------------------


def test_the_whole_cold_start_is_tens_of_hand_authored_lines(
    authored: ColdStartProject,
) -> None:
    """Science plus base plus envelope fits inside one envelope's line budget."""
    counts = {
        relative: len((authored.root / relative).read_text(encoding="utf-8").splitlines())
        for relative in AUTHORED_SOURCES
    }
    assert sum(counts.values()) <= COLD_START_LINE_BUDGET, counts
    for relative, count in counts.items():
        assert count > 0, relative


def test_the_project_contains_no_machinery(authored: ColdStartProject) -> None:
    """Nothing downstream compiles, parses, emits, discovers, or fulfills.

    Generated custody and Git's own directory are excluded: a compiled document,
    a lock, and a receipt are the engine's products, not the project's source.
    """
    ignored = (".git/", "generated/", "receipts/")
    tracked = sorted(
        relative
        for path in authored.root.rglob("*")
        if path.is_file() or path.is_symlink()
        for relative in [path.relative_to(authored.root).as_posix()]
        if not relative.startswith(ignored)
    )
    assert tracked == sorted((*EXPECTED_INIT_INVENTORY, *AUTHORED_SOURCES))

    python_files = [name for name in tracked if name.endswith(".py")]
    assert python_files == sorted((f"src/{PACKAGE_NAME}/__init__.py", SCIENCE_REF))
    science = (authored.root / SCIENCE_REF).read_text(encoding="utf-8")
    for machinery in (
        "argparse",
        "compile",
        "lower",
        "emit",
        "parse",
        "walk",
        "dispatch",
    ):
        assert machinery not in science, machinery


def test_the_science_surface_gate_admits_the_cold_start_and_refuses_machinery(
    authored: ColdStartProject,
) -> None:
    """The checker itself, run through the installed CLI, is the enforcement."""
    clean = authored.cli(
        [
            "check-project-science-surface",
            "--root",
            ".",
            "--policy",
            "ci/feedbax-science-surface.toml",
            "--baseline-ref",
            BASELINE_REF,
        ]
    )
    assert clean.returncode == 0
    assert "passed" in clean.stdout

    intruder = authored.root / f"src/{PACKAGE_NAME}/compiler.py"
    intruder.write_text("def compile_envelope(document):\n    return document\n", "utf-8")
    try:
        refused = authored.cli(
            [
                "check-project-science-surface",
                "--root",
                ".",
                "--policy",
                "ci/feedbax-science-surface.toml",
                "--baseline-ref",
                BASELINE_REF,
            ],
            check=False,
        )
    finally:
        intruder.unlink()
    assert refused.returncode == 1
    assert "unlisted-file" in refused.stdout
    assert f"src/{PACKAGE_NAME}/compiler.py" in refused.stdout


def test_the_science_policy_is_not_authoritative_before_it_is_ratified(
    framework: InstalledFramework, tmp_path: Path
) -> None:
    """The working tree's own policy never authorizes anything."""
    root = tmp_path / PROJECT_NAME
    root.mkdir()
    framework.cli(["init", "."], cwd=root)
    _git(["init", "-q", "-b", BASELINE_REF, "."], cwd=root)

    check = [
        "check-project-science-surface",
        "--root",
        ".",
        "--policy",
        "ci/feedbax-science-surface.toml",
        "--baseline-ref",
        BASELINE_REF,
    ]
    before = framework.cli(check, cwd=root, check=False)
    assert before.returncode == 1
    assert "has no ratified science-surface policy" in before.stdout

    _git(["add", "-A"], cwd=root)
    _git(["commit", "-q", "-m", "ratify"], cwd=root)

    after = framework.cli(check, cwd=root, check=False)
    assert after.returncode == 0
    assert "passed" in after.stdout


# --------------------------------------------------------------------------
# One complete compile, plan, and fulfillment cycle
# --------------------------------------------------------------------------


def test_one_cycle_compiles_plans_and_fulfils_through_the_installed_cli(
    authored: ColdStartProject,
) -> None:
    output_directory = str(authored.declaration["output_directory"])

    compiled = authored.cli(
        [
            "preflight-experiment-envelope",
            ENVELOPE_REF,
            "--out-dir",
            output_directory,
            "--repo-root",
            ".",
        ]
    )
    assert compiled.returncode == 0
    result = json.loads(compiled.stdout)
    assert result["envelope_schema"] == EXPERIMENT_ENVELOPE_SCHEMA_VERSION
    assert result["name"] == "wide-tally"
    assert result["family"] == "evaluation_run_matrix"

    generated = authored.root / output_directory
    lock_path = generated / result["compile_lock_path"]
    document_path = generated / result["document_path"]
    assert lock_path.is_file() and document_path.is_file()

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["schema_id"] == EXPERIMENT_COMPILE_LOCK_SCHEMA_ID
    assert lock["schema_version"] == EXPERIMENT_COMPILE_LOCK_SCHEMA_VERSION
    validated = _run(
        [
            authored.framework.python,
            "-c",
            (
                "import json, sys; "
                "from feedbax.contracts.experiment_compile_lock import load_compile_lock; "
                "load_compile_lock(json.load(open(sys.argv[1])), field=sys.argv[1]); "
                "print('lock ok')"
            ),
            lock_path,
        ],
        cwd=authored.root,
    )
    assert validated.stdout.strip() == "lock ok"

    document = json.loads(document_path.read_text(encoding="utf-8"))
    assert document["base"]["params"]["span"] == 9, "the envelope's delta did not apply"

    planned = _run(
        [
            authored.framework.python,
            "-c",
            (
                "import json, sys; "
                "from feedbax.analysis.fulfillment_experiment import "
                "plan_experiment_envelope; "
                "plan, index = plan_experiment_envelope("
                "sys.argv[1], output_directory=sys.argv[2]); "
                "print(json.dumps({'target': plan.target.text, "
                "'envelopes': len(index.envelopes)}))"
            ),
            "wide-tally",
            output_directory,
        ],
        cwd=authored.root,
    )
    assert json.loads(planned.stdout) == {
        "target": "evaluation:wide-tally",
        "envelopes": 1,
    }

    fulfil = [
        "fulfill-experiment-envelope",
        "wide-tally",
        "--out-dir",
        output_directory,
        "--repo-root",
        ".",
        "--receipt-root",
        "receipts",
        "--plugin",
        f"{PACKAGE_NAME}.science",
    ]
    first = authored.cli(fulfil)
    assert first.returncode == 0
    summary = json.loads(first.stdout)
    assert summary["target"] == "evaluation:wide-tally"
    assert summary["executed"] == ["evaluation:wide-tally"]
    assert summary["reused"] == []

    manifests = sorted((authored.root / "receipts").rglob("*.json"))
    assert manifests, "fulfillment wrote no receipt"
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"

    again = authored.cli(fulfil)
    assert json.loads(again.stdout)["executed"] == []
    assert json.loads(again.stdout)["reused"] == ["evaluation:wide-tally"]


def test_an_unknown_target_is_a_stable_refusal_rather_than_a_crash(
    authored: ColdStartProject,
) -> None:
    refused = authored.cli(
        [
            "fulfill-experiment-envelope",
            "no-such-artifact",
            "--out-dir",
            str(authored.declaration["output_directory"]),
            "--repo-root",
            ".",
            "--receipt-root",
            "receipts",
            "--plugin",
            f"{PACKAGE_NAME}.science",
        ],
        check=False,
    )
    assert refused.returncode == 2
    assert "CompiledOutputError" in refused.stderr


def test_installed_wheel_compiles_and_materializes_both_v3_training_root_families(
    framework: InstalledFramework, tmp_path: Path
) -> None:
    """The public v3 roots work without this checkout or ``PYTHONPATH``."""
    root = tmp_path / "root-envelope-proof"
    root.mkdir()
    framework.cli(["init", "."], cwd=root)
    script = r"""
import json
import pathlib
import sys

from feedbax.contracts.authored_canonical import canonical_sha256, emit_text
from feedbax.contracts.run_composition import (
    CompositionNode,
    InlineIntentParent,
    ResolvedOutputParent,
    authored_envelope_hash,
)
from feedbax.contracts.run_matrix import TrainingRowParentProvenance
from feedbax.contracts.training import (
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.contracts.project_experiment import load_project_declaration
from feedbax.envelope import kernel_for
from feedbax.training.row_lowering import (
    GovernedTrainingRowParent,
    TrainingRowLoweringContext,
)
from feedbax.training.run_matrix import materialize_adapted_run_matrix

repo = pathlib.Path(sys.argv[1])
envelope_dir = repo / "specs" / "experiment"
intent_dir = repo / "specs" / "intent"
intent_dir.mkdir(parents=True)

run = TrainingRunSpec(
    graph={
        "inline": {
            "nodes": {
                "gain": {
                    "type": "Gain",
                    "params": {"gain": 1.0},
                    "input_ports": ["input"],
                    "output_ports": ["output"],
                }
            },
            "wires": [],
            "input_ports": ["input"],
            "output_ports": ["output"],
            "input_bindings": {"input": ["gain", "input"]},
            "output_bindings": {"output": ["gain", "output"]},
        }
    },
    task=TaskSpec(type="GenericTask", params={"n_steps": 1}),
    training_config=TrainingConfig(n_batches=1, batch_size=1),
    objective=ObjectiveSlotSpec(
        loss=LossTermSpec(
            type="target_state",
            label="target",
            selector="port:gain.output",
            target_value=[0.0],
        )
    ),
    method_ref=standard_supervised_method_ref(),
    method_payload=standard_supervised_method_payload(),
    worker_execution=WorkerExecutionSpec(
        method_contract=standard_supervised_method_contract(),
        effective_phase=standard_supervised_effective_phase_spec(),
    ),
).model_dump(mode="json", exclude_none=True)

run_ref = "specs/intent/generic.training_run.json"
(repo / run_ref).write_text(emit_text(run), encoding="utf-8")
source_ref = "specs/intent/generic.items.json"
(repo / source_ref).write_text(
    emit_text({"items": [[1.25, -0.0], [1.25, -0.0], [2.5, 3.75]]}),
    encoding="utf-8",
)
authority = {
    "schema_id": "feedbax.spec.root_training_authority",
    "schema_version": "feedbax.spec.root_training_authority.v1",
    "sources": [{"alias": "items", "kind": "json", "uri": source_ref}],
    "derivations": [
        {
            "output_path": "method_payload.payload.metadata.records",
            "query": {
                "kind": "map_object_list",
                "items": {"item": "items", "path": "items"},
                "template": {"fixed": {"shape": [2]}},
                "item_output_path": "value",
            },
        }
    ],
}
authority_ref = "specs/intent/generic.root_training_authority.json"
(repo / authority_ref).write_text(emit_text(authority), encoding="utf-8")
composition = CompositionNode(
    name="generic-composition",
    parent=InlineIntentParent(
        payload=run,
        schema_id=run["schema_id"],
        schema_version=run["schema_version"],
    ),
)
composition_document = composition.model_dump(mode="json", exclude_none=True)
composition_ref = "specs/intent/generic.composition.json"
(repo / composition_ref).write_text(emit_text(composition_document), encoding="utf-8")

envelopes = {
    "composition-root": {
        "schema": "feedbax.experiment_envelope.v3",
        "name": "composition-root",
        "training": {
            "root": {
                "kind": "composition",
                "parent": {
                    "kind": "authored_intent",
                    "ref": composition_ref,
                    "content_hash": authored_envelope_hash(composition),
                },
                "deltas": [{
                    "layer_id": "matrix",
                    "patches": [{
                        "op": "replace",
                        "path": "training_config.n_batches",
                        "value": 2,
                    }],
                }],
                "rows": [{
                    "id": "condition-a",
                    "delta": {
                        "layer_id": "condition-a",
                        "patches": [{
                            "op": "replace",
                            "path": "training_config.batch_size",
                            "value": 2,
                        }],
                    },
                }],
            }
        },
    },
    "resolved-root": {
        "schema": "feedbax.experiment_envelope.v3",
        "name": "resolved-root",
        "training": {
            "root": {
                "kind": "composition",
                "parent": {
                    "kind": "resolved_output",
                    "ref": "artifact-blob:generic-terminal",
                    "resolved_root_hash": canonical_sha256(run),
                    "row_id": "source-row",
                    "checkpoint_transaction_id": "source-transaction",
                },
                "selected_checkpoint": {
                    "source_run_id": "source-run",
                    "checkpoint_root_hash": "9" * 64,
                    "source_barrier": "after_segment",
                },
                "rows": [{"id": "condition-b"}],
                "fork": {
                    "lr_continuation": "continue",
                    "parity": "require",
                    "absolute_lr_tolerance": 1e-12,
                },
            }
        },
    },
    "training-run-root": {
        "schema": "feedbax.experiment_envelope.v3",
        "name": "training-run-root",
        "training": {
            "root": {
                "kind": "training_run",
                "ref": run_ref,
                "content_hash": canonical_sha256(run),
                "rows": [{"id": "condition-c"}],
                "authority": {"ref": authority_ref, "sha256": canonical_sha256(authority)},
            }
        },
    },
}
kernel = kernel_for(load_project_declaration(repo))
outcomes = {}
for name, envelope in envelopes.items():
    path = envelope_dir / f"{name}.envelope.json"
    path.write_text(emit_text(envelope), encoding="utf-8")
    outcomes[name] = kernel.compile_envelope_file(path, repo_root=repo)

composed = materialize_adapted_run_matrix(
    outcomes["composition-root"].document,
    repo_root=repo,
    row_validator=lambda _payload, _row_id: None,
)
assert composed.rows[0].authored_payload["training_config"]["n_batches"] == 2
assert composed.rows[0].authored_payload["training_config"]["batch_size"] == 2

resolved_parent = ResolvedOutputParent(
    ref="artifact-blob:generic-terminal",
    resolved_root_hash=canonical_sha256(run),
    row_id="source-row",
    checkpoint_transaction_id="source-transaction",
)
context = TrainingRowLoweringContext((GovernedTrainingRowParent(
    provenance=TrainingRowParentProvenance(
        role="terminal",
        parent_kind="resolved_output",
        ref=resolved_parent.ref,
        semantic_hash=resolved_parent.resolved_root_hash,
        artifact_id="generic-terminal",
        artifact_sha256=canonical_sha256(run),
        schema_id=run["schema_id"],
        schema_version=run["schema_version"],
    ),
    parent=resolved_parent,
    payload=run,
),))
resolved = materialize_adapted_run_matrix(
    outcomes["resolved-root"].document,
    repo_root=repo,
    row_validator=lambda _payload, _row_id: None,
    row_lowering_context=context,
)
assert resolved.rows[0].authored_payload["schema_id"] == run["schema_id"]
assert resolved.rows[0].authored_payload["schema_version"] == run["schema_version"]
assert resolved.rows[0].authored_payload["training_config"] == run["training_config"]
dependency = outcomes["resolved-root"].document["execution_dependencies"][0]
assert dependency["source_authority"] == {
    "kind": "resolved_output_root",
    "source_run_id": "source-run",
    "resolved_root_hash": canonical_sha256(run),
}
assert dependency["source_row_id"] == "source-row"
assert dependency["checkpoint_transaction_id"] == "source-transaction"
assert dependency["source_barrier"] == "after_segment"
assert "execution_hash" not in json.dumps(dependency)
assert outcomes["training-run-root"].document["base"]["content_hash"] == canonical_sha256(run)
training_run = materialize_adapted_run_matrix(
    outcomes["training-run-root"].document,
    repo_root=repo,
    row_validator=lambda _payload, _row_id: None,
)
assert training_run.rows[0].authored_payload["method_payload"]["payload"]["metadata"][
    "records"
] == [
    {"fixed": {"shape": [2]}, "value": [1.25, -0.0]},
    {"fixed": {"shape": [2]}, "value": [1.25, -0.0]},
    {"fixed": {"shape": [2]}, "value": [2.5, 3.75]},
]
source_pin = next(
    item
    for item in outcomes["training-run-root"].compile_lock["references"]
    if item.get("ref") == source_ref
)
assert source_pin["content_hash"] == canonical_sha256(
    {"items": [[1.25, -0.0], [1.25, -0.0], [2.5, 3.75]]}
)
authority_pin = next(
    item
    for item in outcomes["training-run-root"].compile_lock["references"]
    if item.get("ref") == authority_ref
)
assert authority_pin["content_hash"] == canonical_sha256(authority)
print(json.dumps({
    "schemas": sorted(outcome.envelope_schema for outcome in outcomes.values()),
    "families": sorted(outcome.family for outcome in outcomes.values()),
}))
"""
    completed = _run([framework.python, "-c", script, root], cwd=root)
    proof = json.loads(completed.stdout)
    assert proof == {
        "schemas": ["feedbax.experiment_envelope.v3"] * 3,
        "families": ["training_run_matrix"] * 3,
    }


def test_the_fixture_never_reached_back_into_this_checkout(
    framework: InstalledFramework,
) -> None:
    """Whatever the cold start proved, it proved about the installed wheel."""
    observed = _run(
        [framework.python, "-c", "import sys, json; print(json.dumps(sys.path))"],
        cwd=framework.environment,
    ).stdout
    assert str(REPO_ROOT) not in observed
    assert "PYTHONPATH" not in _clean_env()
    assert sys.executable != str(framework.python)
