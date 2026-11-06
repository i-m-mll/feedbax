# Force and signal transforms

Use force fields and signal transforms as graph components. Connect a task trajectory to a component input port to vary the control over time. Broadcast a trial-constant control across the steps if it uses that same port. To initialize owned state at reset, set the component's node and StateIndex path in `TaskTrialSpec.inits`.

::: feedbax.mechanics.force_fields

::: feedbax.runtime.signal_transforms
