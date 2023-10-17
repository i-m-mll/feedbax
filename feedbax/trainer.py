"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
import sys
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jrandom
from jaxtyping import Array, Float, PyTree
import optax
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from feedbax.task import AbstractTask
from feedbax.utils import (
    delete_contents,
    filter_spec_leaves, 
    tree_set_idx,
)


logger = logging.getLogger(__name__)


class Trainer(eqx.Module):
    """
    
    
    """
    task: AbstractTask 
    model: eqx.Module
    optimizer: optax.Optimizer
    writer: Optional[SummaryWriter]
    
    def __init__(
        self,
        task: AbstractTask,
        model: eqx.Module,
        optimizer: optax.Optimizer,    
        tensorboard_logdir: Optional[str] = None,
    ):
        self.task = task
        self.model = model 
        self.optimizer = optimizer 
        self._use_tb = tensorboard_logdir is not None
        if self._use_tb:
            self.writer = SummaryWriter(tensorboard_logdir)
            # display loss terms in the same figure under "Custom Scalars"
            # layout = {
            #     "Loss terms": {
            #         "Training loss": ["Multiline", ["Loss/train"] + [f'Loss/train/{term}'
            #                                         for term in term_weights.keys()]],
            #         "Evaluation loss": ["Multiline", ["Loss/eval"] + [f'Loss/eval/{term}'
            #                                         for term in term_weights.keys()]],
            #     },
            # }
            # writer.add_custom_scalars(layout)    
    
    def __call__(
        self, 
        n_batches, 
        n_replicates,  #?
        batch_size, 
        trainable_leaves_func= lambda tree: (tree.step.net.cell.weight_hh, 
                                             tree.step.net.cell.weight_ih, 
                                             tree.step.net.cell.bias),
        log_step=100, 
        restore_checkpoint=False,
        chkpt_dir=None,
        *,
        key,
    ):
        filter_spec = filter_spec_leaves(models, trainable_leaves_func)
        
        losses = jnp.empty((n_batches, n_replicates))
        losses_terms = dict(zip(
            self.task.term_weights.keys(), 
            [jnp.empty((n_batches, n_replicates)) for _ in self.task.term_weights]
        ))
        
        if restore_checkpoint:
            with open(chkpt_dir / "last_batch.txt", 'r') as f:
                last_batch = int(f.read()) 
                
            start_batch = last_batch + 1
            
            chkpt_path = chkpt_dir / f'model{last_batch}.eqx'
            models = eqx.tree_deserialise_leaves(chkpt_path, models)
            losses, losses_terms = eqx.tree_deserialise_leaves(
                chkpt_dir / f'losses{last_batch}.eqx', 
                (losses, losses_terms),
            )

            logger.info(f"Restored checkpoint {chkpt_path} from training step {last_batch}")
            
        else:
            start_batch = 0
            delete_contents(chkpt_dir)  
            
        # TODO: should also restore this from checkpoint
        opt_states = eqx.filter_vmap(
            lambda model: self.optimizer.init(eqx.filter(model, eqx.is_array))
        )(
            models
        )    
        
        keys = jrandom.split(key, n_batches)
        get_batch = jax.vmap(self.task.get_trial)
        
        #for _ in range(epochs): #! assume 1 epoch (no fixed dataset)
        for batch in tqdm(range(start_batch, n_batches), 
                          desc='batch', initial=start_batch, total=n_batches):
            trials_keys = jrandom.split(keys[batch], batch_size)
            init_state, target_state = get_batch(trials_keys)
            
            loss, loss_terms, models, opt_states = self.train_step(
                models, filter_spec, init_state, target_state, opt_states
            )
            
            losses = losses.at[batch].set(loss)
            losses_terms = tree_set_idx(losses_terms, loss_terms, batch)
            
            # tensorboard losses on every iteration
            #! just report for one of the replicates
            if self._use_tb:
                self.writer.add_scalar('Loss/train', loss[0].item(), batch)
                for term, loss_term in loss_terms.items():
                    self.writer.add_scalar(f'Loss/train/{term}', loss_term[0].item(), batch)
            
            if jnp.sum(jnp.isnan(loss)) > n_replicates // 2:
                raise ValueError(f"\nNaN loss on more than 50% of replicates at batch {batch}!")
            
            # checkpointing and evaluation occasionally
            if (batch + 1) % log_step == 0:
                # model checkpoint
                eqx.tree_serialise_leaves(chkpt_dir / f'models{batch}.eqx', models)
                eqx.tree_serialise_leaves(chkpt_dir / f'losses{batch}.eqx', 
                                        (losses, losses_terms))
                with open(chkpt_dir / "last_batch.txt", 'w') as f:
                    f.write(str(batch)) 
                
                # tensorboard
                loss_eval, loss_eval_terms, states = eqx.filter_vmap(self.task.evaluate)(models)
                
                if self._use_tb:
                    # TODO: register plots
                    # fig, _ = plot_states_forces_2d(
                    #         states[1][0][0], states[1][1][0], states[2][0], eval_endpoints[..., :2], 
                    #         cmap='plasma', workspace=workspace
                    # )
                    # self.writer.add_figure('Eval/centerout', fig, batch)
                    self.writer.add_scalar('Loss/eval', loss_eval[0].item(), batch)
                    for term, loss_term in loss_eval_terms.items():
                        self.writer.add_scalar(f'Loss/eval/{term}', loss_term[0].item(), batch)
                    
                # TODO: https://stackoverflow.com/a/69145493
                tqdm.write(f"step: {batch}, training loss: {loss[0]:.4f}", file=sys.stderr)
                tqdm.write(f"step: {batch}, center out loss: {loss_eval[0]:.4f}", file=sys.stderr)
        
        return models, losses, loss_terms
    
    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None, eqx.if_array(0)))
    def train_step(
        self, 
        model, 
        filter_spec, 
        init_state, 
        target_state, 
        task_input, 
        opt_state, 
        key
    ):
        diff_model, static_model = eqx.partition(model, filter_spec)
        (loss, loss_terms), grads = eqx.filter_value_and_grad(self.task.loss_fn, has_aux=True)(
            diff_model, static_model, init_state, target_state, task_input, key, 
            term_weights=term_weights, # discount=position_error_discount
        )
        updates, opt_state = self.optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, loss_terms, model, opt_state