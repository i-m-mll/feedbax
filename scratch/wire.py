class Wire(eqx.Module):
    """Connection delay implemented as a queue, with added noise.
    
    A list implementation is faster than modifying a JAX array.
    
    TODO: 
    - Infer delay steps from time.
    """
    delay: int 
    noise_std: float 
    
    def __call__(self, input, state, key):
        # state = jax.tree_map(lambda x: jnp.roll(x, -1, axis=0), state)
        # state = tree_set_idx(state, self._add_noise(input, key), -1)
        _, queue = state
        queue.append(input)
        output = self._add_noise(queue.pop(0), key)
        return input, (output, queue), key 
    
    def init_state(self, input, state, key):
        # return jax.tree_map(
        #     lambda x: jnp.zeros((self.delay, *x.shape), dtype=x.dtype),
        #     input
        # )
        input_zeros = jax.tree_map(jnp.zeros_like, input)
        return (input_zeros, 
                (self.delay - 1) * [input_zeros] + [input])
        
    @cached_property
    def _add_noise(self):
        if self.noise_std is None:
            return lambda x: x
        else:
            return self.__add_noise 
    
    def __add_noise(self, x, key):
        return x + self.noise_std * jr.normal(key, x.shape) 