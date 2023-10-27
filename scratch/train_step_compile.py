# This was a method I had written to separate out the compilation 
# of `TaskTrainer.train_step`, so that I could try to force it to 
# compile in a notebook before starting a training run. The idea was 
# to stop the compilation from being included in the profiling of 
# the run as a whole. This didn't work, and I'm not sure why.

# There are two loop steps because `opt_state` is reshaped after the 
# first step by `optimizer.update`, for some reason, and I assumed
# that this might be causing a recompilation on the second step.

def _train_step_compile(
    self,
    key,
    task,
    model,
    batch_size,
    trainable_leaves_func,
):
    """Forces JIT compilation of the `jit`-decorated `train_step`.
    
    This is useful because we 
    
        1) want to time the compilation separately,
        2) in some (internal) use cases we want to force the compilation 
            to happen before profiling an entire run, in which case this
            serves as a helper to avoid needing to repeat a bunch of steps
            elsewhere (e.g. flattening `model` and `opt_state`). So
            when `TaskTrainer._train` calls this, a few operations it has
            already performed will be repeated, but only once, and they are
            much less expensive than the JIT compilation itself. 
    """
    get_batch = jax.vmap(task.get_train_trial)
    filter_spec = filter_spec_leaves(model, trainable_leaves_func)
    loss_func_wrapped = grad_wrap_loss_func(task.loss_func)
    opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
    
    flat_model, treedef_model = jax.tree_util.tree_flatten(model)
    flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)
    
    keys = jr.split(key, batch_size)
    init_state, target_state, task_input = get_batch(keys)
    
    for _ in range(2):
        _, _, flat_model, flat_opt_state, treedef_opt_state = self.train_step(
            flat_model, 
            treedef_model,
            flat_opt_state,
            treedef_opt_state,
            filter_spec, 
            init_state, 
            target_state, 
            task_input,
            loss_func_wrapped,
            key, 
        )  