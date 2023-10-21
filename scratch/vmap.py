
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

class Test(eqx.Module):
    weights: dict
    
    def __call__(self, n_batches, key):
        losses = jnp.zeros((n_batches,))
        losses_terms = dict(zip(
            self.weights.keys(),
            [jnp.empty((n_batches,)) for _ in self.weights]
        ))
        return losses, losses_terms 

key = jrandom.PRNGKey(0)

tt = Test(dict(a=1, b=2))
eqx.tree_pprint(tt(5, key))

jax.vmap(tt, in_axes=(None, 0))(5, jrandom.split(key, 5))
    