from collections.abc import Callable, Mapping, Sequence
from rlrmp.analysis.analysis import AbstractAnalysis, AbstractAnalysisPorts, InputOf, NoPorts
from rlrmp.tree_utils import ldict_level_to_bottom, rearrange_ldict_levels, tree_level_labels
from rlrmp.types import AnalysisInputData, LDict, TreeNamespace

import equinox as eqx
from equinox import Module
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, PyTree
from sklearn.decomposition import PCA

from feedbax.bodies import SimpleFeedbackState
from feedbax.misc import batch_reshape, nan_bypass
from jax_cookbook import is_type, is_module
import jax_cookbook.tree as jtree

from types import MappingProxyType
from typing import Literal, Optional


class PCAResults(Module):
    pca: PCA 
    states_pc: Optional[PyTree[Array]] = None  
    
    def batch_transform(self, tree: PyTree[Array]) -> PyTree[Array]:
        """Transform a PyTree's arrays using the PCA transform, retaining batch axes."""
        return jt.map(
            batch_reshape(nan_bypass(self.pca.transform)),  # type: ignore
            tree,
        )


class StatesPCA(AbstractAnalysis[NoPorts]):
    """Perform principle component analysis on evaluated states, with sklearn.
    
    Assumes that `data.states` is an array PyTree of shape `(*batch, n_vars)`.
    After flattening the batch axes, PCA is fit on the concatenated leaves of the PyTree;
    thus the user should ensure that `n_vars` is the same for all leaves, either by 
    passing an appropriate value for the `where_states` field, or by transforming the 
    instance of this class with an appropriate prep op.

    Fields:
        n_components: Number of principal components to keep.
        where_states: Optional[Callable[[SimpleFeedbackState], PyTree[Array]]] = None

    Returns: A `TreeNamespace` with the following attributes:
        pca: The fitted sklearn PCA object.
        batch_transform: A function that applies the PCA transform to a PyTree of arrays of shape
            `(*batch, n_vars)`, retaining the batch axes in the output. 
    """
    
    n_components: Optional[int] = None
    where_states: Optional[Callable[[SimpleFeedbackState], PyTree[Array]]] = None
    return_data: bool = True
    aggregate_over_labels: Sequence[str] | Literal['all'] = ()

    def compute(
        self,
        data: AnalysisInputData,
        **kwargs,
    ):
        states = data.states
        if self.aggregate_over_labels == 'all':
            is_leaf = is_type(type(states))
        else:
            if self.aggregate_over_labels:
                states = rearrange_ldict_levels(
                    states, [...] + list(self.aggregate_over_labels), is_leaf=is_module
                )
                is_leaf = LDict.is_of(self.aggregate_over_labels[0])
            else:
                is_leaf = is_module     
        
        def get_pca(states: PyTree[Array]) -> PCAResults:
            if self.where_states is not None: 
                states = jt.map(self.where_states, states, is_leaf=is_module)

            # flatten into samples; assume last dimension is features
            states_reshaped = jt.map(lambda arr: arr.reshape(-1, arr.shape[-1]), states)

            #! TODO: Use JAX-native PCA
            pca = PCA(n_components=self.n_components).fit(
                jnp.concatenate(jt.leaves(states_reshaped))  # type: ignore
            )
        
            if self.return_data:
                states_pc = jt.map(
                    batch_reshape(pca.transform),  # type: ignore
                    states,
                ) 
            else:
                states_pc = None

            return PCAResults(
                pca=pca,
                states_pc=states_pc,
            )
        
        return jt.map(get_pca, states, is_leaf=is_leaf)


