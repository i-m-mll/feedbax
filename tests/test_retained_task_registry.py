import asyncio

import pytest

from feedbax.contracts.migrations import UnsupportedComponentMigration
from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD
from feedbax.plugins.composition import compose_application
from feedbax.plugins.retained import compose_retained_application
from feedbax.studio.authoring import authoring_surface


def test_retained_task_history_is_available_only_in_retained_bootstrap():
    managed = asyncio.run(compose_application(entry_points=()))
    retained = asyncio.run(compose_retained_application(entry_points=()))
    catalog = authoring_surface(managed, catalog=True)
    published_types = {item.name for item in catalog.registries["components"].entries}

    task_names = ("SimpleReaches", "DelayedReaches", "Stabilization")
    assert set(retained.bundle.components.names()) - published_types == {
        f"feedbax.{task}" for task in task_names
    }
    for name in task_names:
        params = {"n_steps": 200}
        with pytest.raises(UnsupportedComponentMigration):
            managed.bundle.components.resolve_component_spec(name, params, legacy_token="1")
        resolution = retained.bundle.components.resolve_component_spec(
            name, params, legacy_token="1", complete_params=False,
        )
        assert (resolution.type_id, resolution.revision, resolution.params) == (
            f"feedbax.{name}", 1, params,
        )

    managed_method = managed.bundle.training_programs.program(STUDIO_GRAPH_METHOD)
    retained_method = retained.bundle.training_programs.program(STUDIO_GRAPH_METHOD)
    assert managed_method is not None and retained_method is not None
    assert managed_method.preparation_provider is None
    assert retained_method.preparation_provider is not None
