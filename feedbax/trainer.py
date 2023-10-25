"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from datetime import datetime
from functools import wraps
import json 
import logging
from pathlib import Path
import sys
from typing import Callable, Optional, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jrandom
from jaxtyping import Array, Float, PyTree
import matplotlib.pyplot as plt
import optax
from tqdm.auto import tqdm

from feedbax.loss import AbstractLoss
from feedbax.task import AbstractTask
from feedbax.utils import (
    delete_contents,
    filter_spec_leaves, 
    git_commit_id,
    tree_set_idx,
)

if TYPE_CHECKING:
    # this is sloow so we'll actually import it only when needed
    from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class TaskTrainer(eqx.Module):
    """A class 
    
    NOTE: 
    I don't think it makes sense to use this to train (say) jPCA or linear
    regressions, which might not use a `AbstractTask` but operate directly
    on an entire dataset/single array at once. I should try writing another
    training for those cases, and see if perhaps an `AbstractTrainer` makes
    sense to capture the common aspects. 
    
    TODO:
    - Should be vmappable; i.e. for ensemble training.
    - Should work as a component of a larger model; say, if we want to train
      a smaller model during each step of a larger one.    
    """
    optimizer: optax.GradientTransformation
    checkpointing: bool
    chkpt_dir: Optional[Path]
    writer: Optional[SummaryWriter]
    _use_tb: bool
    
    def __init__(
        self,
        optimizer: optax.GradientTransformation,    
        checkpointing=True,
        chkpt_dir='.ckpts',
        tensorboard_logdir: Optional[str] = None,
    ):
        self.optimizer = optimizer 
        self._use_tb = tensorboard_logdir is not None
        if self._use_tb:
            from torch.utils.tensorboard import SummaryWriter
            
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
        else:
            self.writer = None
    
        self.checkpointing = checkpointing
        self.chkpt_dir = Path(chkpt_dir)
        if self.checkpointing:
            self.chkpt_dir.mkdir(exist_ok=True)
            
    def model_ensemble(
        self,
        task: AbstractTask,
        get_model: Callable[[jrandom.PRNGKeyArray], eqx.Module],
        n_model_replicates: int, 
        n_batches: int, 
        batch_size: int, 
        trainable_leaves_func: Callable = lambda model: model,
        log_step:int = 100, 
        restore_checkpoint:bool = False,
        save_dir: Optional[str] = None,
        *,
        key: jrandom.PRNGKeyArray,
    ):
        """Trains an ensemble of models.
        
        Instead of taking a single model like `__call__`, this takes a function
        that generates a single model from a random key, along with a single key.
        It uses these to generate `n_model_replicates` models from different keys,
        then trains them in parallel on the same task data.
        
        NOTE: 
        - I'm not sure what to do about tensorboard/print statements. We could log all replicates 
          (probably too messy), log a single replicate, or log some statistics.
          But I'm not sure how to implement this from within the vmapped function...
        
        TODO: 
        - Allow variation in the task as well as the model.
        """
        key1, key2 = jrandom.split(key)
        keys_model = jrandom.split(key1, n_model_replicates)
        keys_train = jrandom.split(key2, n_model_replicates)
        models = eqx.filter_vmap(get_model)(keys_model)
        models_arrays, models_other = eqx.partition(models, eqx.is_array)
        
        # only map over model arrays and training keys
        in_axes = (None, 0, None, None, None, None, None, None, None, 0, None)
        
        return eqx.filter_vmap(self._train, in_axes=in_axes)(
            task, 
            models_arrays, 
            models_other,
            n_batches, 
            batch_size, 
            trainable_leaves_func,
            log_step,
            restore_checkpoint,
            save_dir,
            keys_train,
            True,
        )
        
    
    def __call__(
        self, 
        task: AbstractTask,
        model: eqx.Module,
        n_batches: int, 
        batch_size: int, 
        trainable_leaves_func: Callable = lambda model: model,
        log_step: int = 100, 
        restore_checkpoint: bool = False,
        save_dir: Optional[str] = None,
        *,
        key: jrandom.PRNGKeyArray,
    ):
        """Train a model on a task for a fixed number of batches of trials."""
        
        return self._train(
            task, 
            model, 
            model, 
            n_batches, 
            batch_size, 
            trainable_leaves_func,
            log_step,
            restore_checkpoint,
            save_dir,
            key,
            False,
        )
        
    def _train(
        self,
        task,
        model_arrays,
        model_other,
        n_batches,
        batch_size,
        trainable_leaves_func,
        log_step,
        restore_checkpoint,
        save_dir,
        key,
        ensembled,
    ):
        """Implementation of the training procedure. 
        
        This is a private kwargless method for the purpose of vmapping, and
        altering control flow based on whether the models are ensembled. 
        
        TODO:
        - Try a different approach to ensembling, where the vmapping
        is only performed on appropriate subsets of the training procedure,
        and other stuff (e.g. logging, NaN control flow) is handled outside 
        of JAX transformations.
        """
        model = eqx.combine(model_arrays, model_other)
        
        filter_spec = filter_spec_leaves(model, trainable_leaves_func)
        
        losses = jnp.empty((n_batches,))
        losses_terms = dict(zip(
            task.loss_func.weights.keys(), 
            [jnp.empty((n_batches,)) for _ in task.loss_func.weights]
        ))
        loss_func_wrapped = grad_wrap_loss_func(task.loss_func)
        
        start_batch = 0  # except when we load a checkpoint
        
        if restore_checkpoint:
            chkpt_path, last_batch, model, losses, losses_terms = \
                self._load_last_checkpoint(model, losses, losses_terms)
            start_batch = last_batch + 1
            logger.info(f"Restored checkpoint {chkpt_path} from training step {last_batch}.")
            
        elif self.checkpointing:
            # delete old checkpoints if checkpointing is on
            # TODO: keep old checkpoints for past N runs (env variable?)
            delete_contents(self.chkpt_dir)  
            
        # TODO: should also restore this from checkpoint
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        
        keys = jrandom.split(key, n_batches - start_batch)
        get_batch = jax.vmap(task.get_trial)
        
        # assume 1 epoch (i.e. batch iterations only; no fixed dataset)
        for batch in tqdm(range(start_batch, n_batches), 
                          desc='batch', initial=start_batch, total=n_batches):
            key_train, key_eval, key_batch = jrandom.split(keys[batch], 3)
            keys_trials = jrandom.split(key_batch, batch_size)
            init_state, target_state, task_input = get_batch(keys_trials)

            loss, loss_terms, model, opt_state = self.train_step(
                model, 
                filter_spec, 
                init_state, 
                target_state, 
                task_input,
                loss_func_wrapped, 
                opt_state,
                key_train, 
            )
            
            losses = losses.at[batch].set(loss)
            losses_terms = tree_set_idx(losses_terms, loss_terms, batch)
            
            if ensembled: 
                # ensemble training skips the logging and NaN control flow stuff
                continue
            
            # tensorboard losses on every iteration
            #! just report for one of the replicates
            if self._use_tb:
                self.writer.add_scalar('Loss/train', loss[0].item(), batch)
                for term, loss_term in loss_terms.items():
                    self.writer.add_scalar(f'Loss/train/{term}', loss_term[0].item(), batch)
        
            if jnp.isnan(loss):
                msg = f"NaN loss at batch {batch}! "
                if (checkpoint := self._load_last_checkpoint(
                    model, losses, losses_terms
                )) is not None:
                    _, last_batch, model, losses, losses_terms = checkpoint
                    msg += f"Returning checkpoint from batch {last_batch}."
                else:
                    msg += "No checkpoint found, returning model from final iteration."
                
                logger.warning(msg)
                
                return model, losses, losses_terms
            

            # checkpointing and evaluation occasionally
            if batch % log_step == 0:
                # model checkpoint
                if self.checkpointing:
                    eqx.tree_serialise_leaves(self.chkpt_dir / f'model{batch}.eqx', model)
                    eqx.tree_serialise_leaves(self.chkpt_dir / f'losses{batch}.eqx', 
                                              (losses, losses_terms))
                    with open(self.chkpt_dir / "last_batch.txt", 'w') as f:
                        f.write(str(batch)) 
                
                # tensorboard
                loss_eval, loss_eval_terms, states = task.eval(model, key_eval)
                
                if self._use_tb:
                    # TODO: register plots
                    # fig = make_eval_fig(states.effector, states.control, workspace)
                    # self.writer.add_figure('Eval/centerout', fig, batch)
                    self.writer.add_scalar('Loss/eval', loss_eval.item(), batch)
                    for term, loss_term in loss_eval_terms.items():
                        self.writer.add_scalar(f'Loss/eval/{term}', loss_term.item(), batch)
                    
                # TODO: https://stackoverflow.com/a/69145493
                tqdm.write(f"step: {batch}", file=sys.stdout)
                tqdm.write(f"\ttraining loss: {loss:.4f}", file=sys.stdout)
                tqdm.write(f"\tevaluation loss: {loss_eval:.4f}", file=sys.stdout)
        
        if save_dir is not None:
            # could probably just concatenate 
            #   1) arguments to this method, 
            #   2) optimizer hyperparameters,
            #   3) `model` and `task pytrees, filtered of callables and jax arrays
            hyperparams = dict(
                commit_hash=git_commit_id(),
                key=key.tolist(),
                batch_size=batch_size,
                n_batches=n_batches,
                workspace=task.workspace.tolist(),
                term_weights=task.loss_func.weights,
                # epochs=n_epochs,
                opt_hyperparams=opt_state.hyperparams,
                # weight_decay=weight_decay,
            )        
            
            save_model(model, hyperparams, Path(save_dir))
         
        return model, losses, losses_terms
    
    @eqx.filter_jit
    #? @eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None, eqx.if_array(0)))
    def train_step(
        self, 
        model, 
        filter_spec, 
        init_state, 
        target_state, 
        task_input, 
        loss_func_wrapped,
        opt_state, 
        key
    ):
        """Executes a single training step of the model.
        
        This assumes that:
        
            - The (wrapped) loss function returns an auxiliary `loss_terms`,
              which is intended to be a dictionary of the loss components 
              prior to their final aggregation (i.e. already weighted).
        """
        diff_model, static_model = eqx.partition(model, filter_spec)
        (loss, loss_terms), grads = eqx.filter_value_and_grad(loss_func_wrapped, has_aux=True)(
            diff_model, 
            static_model, 
            init_state, 
            target_state, 
            task_input, 
            key,
        )
        updates, opt_state = self.optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, loss_terms, model, opt_state

    def _load_last_checkpoint(self, model, losses, losses_terms):
        try:
            with open(self.chkpt_dir / "last_batch.txt", 'r') as f:
                last_batch = int(f.read()) 
        except FileNotFoundError:
            #! this will just raise another error at checkpoint restore
            return
            
        chkpt_path = self.chkpt_dir / f'model{last_batch}.eqx'
        model = eqx.tree_deserialise_leaves(chkpt_path, model)
        losses, losses_terms = eqx.tree_deserialise_leaves(
            self.chkpt_dir / f'losses{last_batch}.eqx', 
            (losses, losses_terms),
        )
        return chkpt_path, last_batch, model, losses, losses_terms
    
    
def grad_wrap_loss_func(
    loss_func: AbstractLoss
):
    """Wraps a loss function taking state to a `grad`-able one taking a model.
    
    It is convenient to first define the loss function in terms of a 
    mapping from states to scalars, because sometimes we want to evaluate the 
    loss on states without re-evaluating the model itself. It also helps to
    separate the mathematical logic of the loss function from the training
    logic of passing a model as an argument to a `grad`-able function.
    
    Note that we are assuming that
    
      1) `TaskTrainer` will manage a `filter_spec` on the trainable parameters. 
         When `jax.grad` is applied to the wrapper, the gradient will be 
         taken with respect to the first argument `diff_model` only, and the 
         `filter_spec` defines this split.
      2) Model modules will use a `target_state, init_state, key` signature.
      
    TODO: 
    - This is specific to `TaskTrainer`. Could it be more general?
      Note that the vmap here works the same as in `Task.eval`; perhaps if this
      is a `TaskTrainer`-specific function, then `Task` could provide an interface
    """
    @wraps(loss_func)
    def wrapper(
        diff_model, 
        static_model, 
        init_state, 
        target_state, 
        task_input, 
        key: jrandom.PRNGKeyArray,
    ):
        model = eqx.combine(diff_model, static_model)
        #? will `in_axes` ever change? currently assuming we will design it not to
        batched_model = jax.vmap(model, in_axes=(0, 0, None))
        states = batched_model(task_input, init_state, key)
        
        # TODO: loss_func should take task_input as well, probably
        return loss_func(states, target_state, task_input)
    
    return wrapper 


def save_model(
            model, 
            hyperparams: Optional[dict] = None, 
            save_dir: Path = Path('.'),
            # fig_funcs: Optional[dict]=None
    ):
    """Save a model to disk, along with hyperparameters and figures.
    
    TODO:
    - Don't save figures in this function. Move to a different one
      if we want this feature.
    """
        
    timestr = datetime.today().strftime("%Y%m%d-%H%M%S") 
    name = f"model_{timestr}"
    hyperparam_str = json.dumps(hyperparams)
    
    eqx.tree_serialise_leaves(save_dir / f'{name}.eqx', model)
    if hyperparams is not None:
        with open(save_dir / f'{name}.json', 'w') as f:
            hyperparams_str = json.dumps(hyperparams, indent=4)
            f.write(hyperparams_str)
    # if fig_funcs is not None:
    #     for label, fig_func in fig_funcs.items():
    #         fig = fig_func()
    #         fig.savefig(model_dir / f'{name}_{label}.png')
    #         plt.close(fig)    
    