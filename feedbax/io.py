"""Saving to and loading from disk."""


import json
from pathlib import Path
from typing import Optional, Tuple, Union

import equinox as eqx
from jaxtyping import PyTree 


#! this is currently unused and doesn't work because json can't serialize jnp.ndarray
#! equinox needs a skeleton to restore a pytree, which I don't want to provide for `extra`.
#! could use orbax-checkpoint? or provide a skeleton for extra inside of `Trainer` or something
def save_checkpoint(
    filepath: Union[str, Path],
    model: PyTree,
    opt_state: Optional[PyTree] = None,
    extra: Optional[PyTree] = None,
) -> None:
    with open(filepath, 'wb') as f:
        other_str = json.dumps((opt_state, extra))
        f.write((other_str + '\n').encode())
        eqx.tree_serialise_leaves(f, model)
            
            
def load_checkpoint(
    filepath: Union[str, Path],
    like: PyTree,
) -> Tuple[PyTree, Optional[PyTree], Optional[PyTree]]:
    with open(filepath, 'rb') as f:
        return eqx.tree_deserialise_leaves(f, like)