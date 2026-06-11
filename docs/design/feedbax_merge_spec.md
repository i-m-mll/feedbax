# Feedbax + Feedbax-Experiments Merge Specification

## Target Package Structure

```
feedbax/
├── __init__.py
├── _model.py
├── _staged.py                    # REFACTOR: remove Iterator dependency assumption
├── _tree.py
├── _mapping.py
├── _io.py
├── _logging.py                   # MERGE: combine with experiments logging
│
├── bodies.py
├── dynamics.py
├── channel.py
├── filters.py
├── noise.py
├── state.py
├── iterate.py                    # DEPRECATE: keep for backwards compat, mark for removal
│
├── mechanics/
│   ├── __init__.py
│   ├── mechanics.py
│   ├── muscle.py
│   ├── plant.py
│   └── skeleton/
│
├── intervene/
│   ├── __init__.py
│   ├── intervene.py
│   ├── remove.py
│   └── schedule.py
│
├── nn.py
├── loss.py
├── task.py                       # REFACTOR: task owns iteration via eval methods
├── train.py
│
├── analysis/                     # FROM: feedbax_experiments.analysis
│   ├── __init__.py
│   ├── base.py                   # AbstractAnalysis, AbstractAnalysisPorts, Data proxy
│   ├── dependencies.py           # _dependencies.py renamed
│   ├── execution.py              # run_analysis_module, setup functions
│   ├── fig_ops.py                # _FigOp, _apply_fig_ops, figure iteration
│   ├── transforms.py             # Transformed, ExpandTo, LiteralInput
│   │
│   ├── builtin/                  # Generic, reusable analyses
│   │   ├── __init__.py
│   │   ├── func.py               # ApplyFns
│   │   ├── violins.py
│   │   ├── pca.py
│   │   ├── regression.py
│   │   ├── grad.py
│   │   └── tangling.py
│   │
│   └── motor_control/            # Feedbax-specific (assumes mechanics.effector structure)
│       ├── __init__.py
│       ├── aligned.py
│       ├── effector.py
│       ├── profiles.py
│       ├── activity.py
│       ├── network.py
│       └── state_utils.py
│
├── database/                     # FROM: feedbax_experiments.database
│   ├── __init__.py
│   ├── models.py                 # ModelRecord, EvaluationRecord, FigureRecord
│   ├── session.py                # db_session, get_db_session
│   └── operations.py             # add_evaluation, query functions
│
├── config/                       # FROM: feedbax_experiments.config
│   ├── __init__.py
│   ├── paths.py                  # PATHS singleton
│   ├── strings.py                # STRINGS singleton
│   └── yaml.py                   # YAML loader utilities
│
├── dashboard/                    # FROM: feedbax_experiments.dashboard (optional extra)
│   └── ...
│
├── plugins/                      # FROM: feedbax_experiments.plugins
│   ├── __init__.py
│   ├── registry.py
│   └── discovery.py
│
├── plot/                         # MERGE: both repos' plotting
│   ├── __init__.py
│   ├── colors.py                 # merge feedbax_experiments.colors
│   ├── mpl.py
│   ├── plotly.py
│   ├── profiles.py
│   ├── trajectories.py
│   └── utils.py                  # FROM: feedbax_experiments.plot_utils
│
├── types.py                      # MERGE: combine both, export from here
├── tree_utils.py                 # FROM: feedbax_experiments.tree_utils
├── misc.py                       # MERGE: deduplicate
├── hyperparams.py                # FROM: feedbax_experiments.hyperparams
├── constants.py                  # FROM: feedbax_experiments.constants
│
├── training/                     # FROM: feedbax_experiments.training
│   ├── __init__.py
│   └── post_training.py
│
├── bin/                          # CLI entry points
│   ├── __init__.py
│   ├── run.py
│   ├── analysis.py
│   └── train.py
│
└── xabdeef/                      # Keep as-is (convenience constructors)
    └── ...
```

## Module Mapping

| Source (feedbax_experiments) | Target (feedbax) | Notes |
|------------------------------|------------------|-------|
| `analysis/analysis.py` | Split → `analysis/{base,fig_ops,transforms}.py` | 3000 lines → ~3 files |
| `analysis/_dependencies.py` | `analysis/dependencies.py` | Rename only |
| `analysis/execution.py` | `analysis/execution.py` | Direct move |
| `analysis/{func,violins,pca,regression,grad,tangling}.py` | `analysis/builtin/` | Generic analyses |
| `analysis/{aligned,effector,profiles,activity,network,state_utils}.py` | `analysis/motor_control/` | Domain-specific |
| `analysis/fps.py`, `fp_finder.py` | `analysis/builtin/fixed_points.py` | Consolidate |
| `analysis/fps_tmp2.py`, `tmp-map.py`, `nn_utils.py` | DELETE | Dead code |
| `database.py` | Split → `database/{models,session,operations}.py` | 1600 lines → ~3 files |
| `config/` | `config/` | Direct move |
| `dashboard/` | `dashboard/` | Direct move, optional extra |
| `plugins/` | `plugins/` | Direct move |
| `types.py` | Merge → `types.py` | Combine with existing feedbax.types |
| `tree_utils.py` | `tree_utils.py` | Direct move |
| `misc.py` | Merge → `misc.py` | Deduplicate with existing |
| `hyperparams.py` | `hyperparams.py` | Direct move |
| `colors.py` | `plot/colors.py` | Merge with existing plot.colors |
| `plot.py`, `plot_utils.py` | `plot/utils.py` | Consolidate |
| `training/` | `training/` | Direct move |
| `bin/` | `bin/` | Direct move |
| `setup_utils.py` | `training/setup.py` or `analysis/setup.py` | Decide based on usage |
| `perturbations.py` | `intervene/perturbations.py` | Domain helper |

## Key Refactoring Tasks

### 1. Remove Iterator Layer Assumption

**Current:** Models are `Iterator[StagedModel]`, accessed via `model.step.net`

**Target:** Models are `StagedModel` directly, iteration handled by task

**Changes:**

```python
# task.py - AbstractTask.eval() handles iteration internally
def eval(self, model, *, key) -> StateT:
    """Evaluate model for n_steps, return full trajectory."""
    keys = jr.split(key, self.n_steps)
    def step(state, args):
        input_, k = args
        return model(input_, state, key=k), state
    _, states = lax.scan(step, self.init_state(model), (self.inputs, keys))
    return states

# Deprecation path for iterate.py
class Iterator(AbstractIterator):
    """DEPRECATED: Use task.eval() directly. Will be removed in v0.3."""
    ...
```

**Migration:** 
- Add `n_steps` to `AbstractTask` (already there)
- Move `lax.scan` logic from `Iterator.__call__` to `AbstractTask.eval`
- Update `TaskTrainer` to work with non-Iterator models
- Grep for `model.step.` and update to `model.`

### 2. Split analysis/analysis.py

Target file breakdown:

| New File | Contents | Approx Lines |
|----------|----------|--------------|
| `base.py` | `AbstractAnalysis`, `AbstractAnalysisPorts`, `NoPorts`, `SinglePort`, `Data` proxy, `AnalysisRef`, `InputOf` type | ~800 |
| `fig_ops.py` | `_FigOp`, `_PrepOp`, `_FinalOp`, `_apply_fig_ops`, `_apply_final_ops`, `_combine_figures`, aggregators | ~600 |
| `transforms.py` | `_DataField`, `ExpandTo`, `Transformed`, `LiteralInput`, `CallWithDeps` | ~400 |
| `base.py` (continued) | `compute` orchestration, `make_figs`, caching, hashing | ~1200 |

### 3. Generalize State Path References

**Current:** Hardcoded `states.mechanics.effector.vel`

**Target:** Use `VarSpec` pattern consistently

```python
# analysis/motor_control/specs.py
EFFECTOR_POS = VarSpec(
    where=lambda data: data.states.mechanics.effector.pos,
    labels=Labels("Position", "Pos", "p"),
    time_axis=-2,
)

# Usage in analysis
class EffectorProfiles(AbstractAnalysis):
    var_spec: VarSpec = EFFECTOR_POS  # configurable
```

### 4. Merge Overlapping Modules

**misc.py conflicts to resolve:**
- `camel_to_snake` — likely duplicated, keep one
- `is_module` — feedbax version canonical
- `indent_str` — feedbax version canonical
- Check for name collisions, prefer feedbax versions

**plot/ conflicts:**
- `colors.py` exists in both — merge color definitions
- `feedbax.plot.mpl` vs `feedbax_experiments.plot` — different purposes, both needed

**_logging.py:**
- feedbax version is simpler
- experiments version has more features (rich integration)
- Keep experiments version, ensure backwards compat

### 5. Update Import Paths

All internal imports in moved modules need updating:

```python
# Before (in feedbax_experiments)
from feedbax_experiments.analysis.analysis import AbstractAnalysis
from feedbax_experiments.types import LDict

# After (in feedbax)
from feedbax.analysis.base import AbstractAnalysis
from feedbax.types import LDict
```

## pyproject.toml Changes

```toml
[project]
name = "feedbax"
version = "0.2.0"  # Major version bump for breaking changes
dependencies = [
    # Existing
    "equinox>=0.11.12",
    "jax>=0.5.2",
    "diffrax>=0.7.0",
    # ... existing deps ...
    
    # New from feedbax_experiments
    "alembic>=1.15.1",
    "dill>=0.3.9",
    "pyexiv2>=2.15.3",
    "ruamel-yaml>=0.18.0",
    "rich>=13.9.4",
    "scikit-learn>=1.6.1",
]

[project.optional-dependencies]
dashboard = [
    "dash>=2.14.0",
    "dash-bootstrap-components>=1.5.0",
]
notebook = [
    "ipyfilechooser>=0.6.0",
    "ipywidgets>=8.1.5",
    "notebook>=7.4.1",
]

[project.scripts]
feedbax = "feedbax.bin.run:main"
feedbax-analysis = "feedbax.bin.analysis:main"
feedbax-train = "feedbax.bin.train:main"
```

## Testing Strategy

### Priority 1: Tree Utilities
```
tests/
├── test_tree_utils.py          # Property tests for tree operations
├── test_ldict.py               # Already exists, expand
└── test_tree_mapping.py        # _tree.py functions
```

### Priority 2: Analysis Graph
```
tests/analysis/
├── test_dependency_resolution.py   # Graph correctly resolves deps
├── test_caching.py                 # Hash stability, cache hits
├── test_fig_ops.py                 # Figure iteration logic
└── test_builtin_analyses.py        # ApplyFns, Violins work correctly
```

### Priority 3: Integration
```
tests/integration/
├── test_simple_reach_workflow.py   # Train → eval → analyze pipeline
└── test_database_roundtrip.py      # Save/load model records
```

### Test Fixtures Needed
```python
# conftest.py
@pytest.fixture
def simple_feedback_model():
    """Minimal SimpleFeedback for testing."""
    ...

@pytest.fixture  
def reach_task():
    """SimpleReaches task with 8 directions."""
    ...

@pytest.fixture
def mock_db_session():
    """In-memory SQLite for testing."""
    ...
```

## Migration Checklist

1. [ ] Create new directory structure in feedbax
2. [ ] Move files according to mapping table
3. [ ] Update all internal imports
4. [ ] Merge overlapping modules (misc, logging, types, plot)
5. [ ] Split analysis.py into multiple files
6. [ ] Split database.py into multiple files
7. [ ] Update pyproject.toml dependencies
8. [ ] Add optional extras (dashboard, notebook)
9. [ ] Deprecate Iterator class
10. [ ] Refactor task.eval to handle iteration
11. [ ] Update TaskTrainer for non-Iterator models
12. [ ] Delete dead code (fps_tmp2, tmp-map, empty nn_utils)
13. [ ] Run existing tests, fix breakage
14. [ ] Add new tests per priority list
15. [ ] Update documentation/docstrings
16. [ ] Update rlrmp imports to use new feedbax paths
17. [ ] Tag release, update changelog

## Breaking Changes Summary

For CHANGELOG.md:

```markdown
## v0.2.0 - BREAKING

### Changed
- `feedbax_experiments` merged into `feedbax` — install `feedbax>=0.2.0` instead
- Import paths changed: `feedbax_experiments.X` → `feedbax.X`
- `Iterator` class deprecated — use `task.eval()` directly
- Model access pattern: `model.step.net` → `model.net`

### Added
- `feedbax.analysis` — lazy computation graph for analyses
- `feedbax.database` — SQLite experiment catalog
- `feedbax.plugins` — experiment registration system
- CLI commands: `feedbax`, `feedbax-analysis`, `feedbax-train`

### Removed
- `feedbax_experiments` package (merged)
```

## Notes for Implementer

- The `jax-cookbook` dependency is shared and unchanged
- `rlrmp` becomes a pure "experiment plugin" with no structural changes needed beyond import updates
- The `xabdeef` subpackage is a backwards-compat convenience layer — leave as-is
- `TrialTimeline` in task.py is already well-designed for the iteration refactor
- Consider keeping `Iterator` functional but deprecated for 1-2 minor versions
- The `VarSpec` pattern exists but isn't used consistently — standardizing this is optional but recommended
