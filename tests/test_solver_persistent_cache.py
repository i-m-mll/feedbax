"""Sampled solver inputs and dynamics survive persistent-cache reuse across processes."""

import json
import subprocess
import sys
from textwrap import dedent


_PROBE = dedent(
    """
    import json
    import logging
    import sys
    from scripts.full_suite import configure_native_threads
    configure_native_threads()
    import diffrax as dfx
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.tree as jt
    import numpy as np
    from feedbax.runtime.filters import FirstOrderFilter

    class CacheHits(logging.Handler):
        count = 0
        def emit(self, record):
            if 'Persistent compilation cache hit' in record.getMessage():
                self.count += 1
    hits = CacheHits()
    logger = logging.getLogger('jax._src.compiler')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers = [hits]
    jax.config.update('jax_compilation_cache_dir', sys.argv[1])
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
    with jax.experimental.enable_x64():
        prototype = {'x': jnp.zeros(2, dtype=jnp.float32),
                     'y': jnp.zeros(3, dtype=jnp.float64)}
        values = {}
        for solver in [dfx.Heun, dfx.Tsit5]:
            for compiled in [False, True]:
                model = FirstOrderFilter(tau_rise=.2, tau_decay=.3, dt=.001,
                                         solver_type=solver, init_value=2)
                model = model.change_input(prototype).prepare()
                state = model.init_state(key=jax.random.PRNGKey(0))
                step = eqx.filter_jit(model) if compiled else model
                for target, tau in [(0., .3), (4., .2)]:
                    previous = state.get(model.state_index).output
                    inputs = jt.map(lambda x: jnp.full_like(x, target), prototype)
                    outputs, state = step({'input': inputs}, state,
                                          key=jax.random.PRNGKey(0))
                    for name, value in outputs['output'].items():
                        expected = target + (np.asarray(previous[name]) - target) * np.exp(-.001/tau)
                        np.testing.assert_allclose(value, expected, atol=1e-7)
                        assert value.dtype == prototype[name].dtype
                values[f'{solver.__name__}-{compiled}'] = float(outputs['output']['y'][0])

        def response(tau):
            model = FirstOrderFilter(tau_rise=tau, dt=.001, solver_type=dfx.Tsit5,
                                     input_proto=jnp.zeros(1), init_value=2).prepare()
            outputs, _ = model({'input': jnp.full(1, 4.)},
                               model.init_state(key=jax.random.PRNGKey(0)),
                               key=jax.random.PRNGKey(0))
            return outputs['output'][0]
        for tau in [.2, .4]:
            static_value = response(tau)
            np.testing.assert_allclose(static_value, 4-2*np.exp(-.001/tau), atol=1e-12)
            values[f'static-tau-{tau}'] = float(static_value)
            value, grad = jax.jit(jax.value_and_grad(response))(jnp.array(tau))
            np.testing.assert_allclose(value, 4-2*np.exp(-.001/tau), atol=1e-12)
            np.testing.assert_allclose(grad, -2*np.exp(-.001/tau)*.001/tau**2, atol=1e-12)
            values[f'tau-{tau}'] = [float(value), float(grad)]
    print(json.dumps({'values': values, 'cache_hits': hits.count}))
    """
)


def test_solver_response_survives_warm_persistent_cache(tmp_path):
    results = []
    for _ in range(2):
        process = subprocess.run(
            [sys.executable, "-c", _PROBE, str(tmp_path / "cache")],
            capture_output=True,
            text=True,
            timeout=25,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        results.append(json.loads(process.stdout))
    assert results[1]["cache_hits"] > 0
    assert results[1]["values"] == results[0]["values"]
