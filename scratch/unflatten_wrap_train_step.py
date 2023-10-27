def unflatten_wrap_train_step(
    train_step,
):
    @wraps(train_step)
    def wrapper(
        flat_model, 
        treedef_model,
        flat_opt_state, 
        treedef_opt_state,
        *args,
    ):
        model = jax.tree_util.tree_unflatten(treedef_model, 
                                             flat_model)
        opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, 
                                                 flat_opt_state)
        
        loss, loss_terms, model, opt_state = train_step(
            model, opt_state, *args
        )
    
        flat_model = jax.tree_util.tree_leaves(model)
        flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)
        
        return loss, loss_terms, flat_model, flat_opt_state, treedef_opt_state

    return wrapper