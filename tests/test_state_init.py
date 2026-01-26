from feedbax.graph import init_state_from_component
from feedbax.intervene import AddNoise, AddNoiseParams


def test_init_state_collects_param_indices():
    params = AddNoiseParams(scale=2.0, active=True)
    component = AddNoise(params=params)
    state = init_state_from_component(component)

    stored = state.get(component.params_index)
    assert stored.scale == params.scale
    assert stored.active == params.active
