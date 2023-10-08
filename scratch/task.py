def get_sequences(key, n_steps, epoch_len_ranges, init, target):
    """Convert static task inputs to sequences, and make hold signal.
    
    TODO: 
    - this could be part of a `Task` subclass
    """
    stim_epochs = jnp.array((1,))
    hold_epochs = jnp.array((0, 1, 2))
    move_epochs = jnp.array((3,))  
    
    epoch_lengths = gen_epoch_lengths(key, epoch_len_ranges)
    epoch_idxs = jnp.pad(jnp.cumsum(epoch_lengths), (1, 0), constant_values=(0, -1))
    epoch_masks = jnp.ones((len(epoch_lengths) + 1, n_steps), dtype=bool)
    for i in range(epoch_lengths.shape[0]):
        update = np.zeros((1, epoch_lengths[i],), dtype=bool)
        epoch_masks = jax.lax.dynamic_update_slice(
            epoch_masks, 
            update, 
            (i, epoch_idxs[i],)
        )
    epoch_masks = jax.lax.dynamic_update_slice_in_dim(
        epoch_masks, 
        jnp.logical_not(jnp.prod(epoch_masks, axis=0))[None, :],
        -1,
        axis=0,
    )
    
    # epoch_masks = jnp.array([jnp.ones((n_steps,), dtype=bool).at[idx0:idx1].set(False)
    #                          for idx0, idx1 in zip(epoch_idxs[:-1], epoch_idxs[1:])])
    
    stim_seqs = get_masked_seqs(target, epoch_masks[stim_epochs])
    target_seqs = jax.tree_map(
        lambda x, y: x + y, 
        get_masked_seqs(target, epoch_masks[move_epochs]),
        get_masked_seqs(init, epoch_masks[hold_epochs]),
    )
    stim_on_seq = get_scalar_epoch_seq(epoch_idxs, n_steps, 1., stim_epochs)
    hold_seq = get_scalar_epoch_seq(epoch_idxs, n_steps, 1., hold_epochs)
    
    task_input = stim_seqs + (hold_seq, stim_on_seq)
    target = target_seqs
    
    return task_input, target, epoch_idxs