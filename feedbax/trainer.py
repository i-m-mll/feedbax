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
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jr
from jaxtyping import Array, Float, PyTree
import optax
from tqdm.auto import tqdm

from feedbax.loss import AbstractLoss
from feedbax.task import AbstractTask, AbstractTaskTrialSpec
from feedbax.utils import (
    delete_contents,
    filter_spec_leaves, 
    git_commit_id,
    tree_set_idx,
)

if TYPE_CHECKING:
    # This is sloow so we'll actually import it only when needed.
    from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class TaskTrainer(eqx.Module):
    """A class 
    
    NOTE: 
    - I don't think it makes sense to use this to train (say) jPCA or linear
      regressions, which might not use a `AbstractTask` but operate directly
      on an entire dataset/single array at once. I should try writing another
      training for those cases, and see if perhaps an `AbstractTrainer` makes
      sense to capture the common aspects.  
    """
    optimizer: optax.GradientTransformation
    checkpointing: bool
    chkpt_dir: Optional[Path]
    writer: Optional[SummaryWriter]
    _use_tb: bool
    
    def __init__(
        self,
        optimizer: optax.GradientTransformation,    
        checkpointing: bool = True,
        chkpt_dir: str | Path ="/tmp/fbx-checkpoints",
        enable_tensorboard: bool = False,
        tensorboard_logdir: str | Path = "/tmp/fbx-tensorboard",
    ):
        self.optimizer = optimizer 
        self._use_tb = enable_tensorboard
        if self._use_tb:
            from torch.utils.tensorboard import SummaryWriter
            
            self.writer = SummaryWriter(tensorboard_logdir)
            # display loss terms in the same figure under "Custom Scalars"
            # layout = {
            #     "Loss terms": {
            #         "Training loss": ["Multiline", ["Loss/train"] + [f'Loss/train/{term}'
            #                                         for term in term_weights.keys()]],
            #         "Evaluation loss": ["Multiline", ["Loss/validation"] + [f'Loss/validation/{term}'
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
    
    def __call__(
        self, 
        task: AbstractTask,
        model: eqx.Module,
        n_batches: int, 
        batch_size: int, 
        key: jax.Array,
        trainable_leaves_func: Callable = lambda model: model,
        batch_callbacks: Optional[Dict[int, Sequence[Callable]]] = None,
        log_step: int = 100, 
        restore_checkpoint: bool = False,
        disable_tqdm=False,
    ):
        """Train a model on a task for a fixed number of batches of trials.
        
        NOTE:
        - Model checkpointing only saves model state, not the task or 
          hyperparameters. That is, it assumes that the model and task passed 
          to this method are, aside from their trainable state, identical to 
          those from the time of checkpointing. This is typically the case 
          when a checkpoint is used locally to resume training. However, trying
          to load a checkpoint as a model later may fail. Use `save` and `load`.
        """
        
        return self._train(
            task, 
            model, 
            model, 
            n_batches, 
            batch_size, 
            trainable_leaves_func,
            batch_callbacks,
            log_step,
            restore_checkpoint,
            key,
            False,
            disable_tqdm,
        )
    
    def train_ensemble(
        self,
        task: AbstractTask,
        models: eqx.Module,
        n_replicates: int,
        n_batches: int, 
        batch_size: int, 
        key: jax.Array,
        trainable_leaves_func: Callable = lambda model: model,
        batch_callbacks: Optional[Dict[int, Sequence[Callable]]] = None,
        log_step: int = 100, 
        restore_checkpoint:bool = False,
        disable_tqdm=False,
    ):
        """Trains an ensemble of models.
        
        This is essentially a helper that vmaps training over the first 
        dimension of `models`.
        
        NOTE: 
        - I'm not sure what to do about tensorboard/print statements. We could log all replicates 
          (probably too messy), log a single replicate, or log some statistics.
          But I'm not sure how to implement this from within the vmapped function...
        
        TODO: 
        - Allow variation in the task as well as the model.
        - In principle we could infer `n_replicates` from `models`, or else 
          have the user pass `keys` instead of `key`.
        """
        keys_train = jr.split(key, n_replicates)
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
            batch_callbacks,
            log_step,
            restore_checkpoint,
            keys_train,
            True,
            disable_tqdm,
        )
    
    @jax.named_scope("fbx.TaskTrainer")
    def _train(
        self,
        task,
        model_arrays,
        model_other,
        n_batches,
        batch_size,
        trainable_leaves_func,
        batch_callbacks,
        log_step,
        restore_checkpoint,
        key,
        ensembled,
        disable_tqdm,
    ):
        """Implementation of the training procedure. 
        
        This is a private kwargless method for the purpose of vmapping, and
        altering control flow based on whether the models are ensembled. 
        
        TODO:
        - Try a different approach to ensembling, where the vmapping
          is only performed on appropriate subsets of the training procedure,
          and other stuff (e.g. logging, NaN control flow) is handled outside 
          of JAX transformations.
        - Improve the handling of the flatten/unflatten operations around 
          `train_step`. See its docstring for details. 
        - The first iteration (or two) are much slower due to JIT compilation
          of `train_step`, which distorts the smoothed it/s estimate of tqdm. 
          Also, `opt_state` seems to change shape only on the first step call 
          to `optimizer.update`, which is why we need to recompute and return 
          `treedef_opt_state` from `train_step`. So maybe the first call 
          should be separated out from the loop.
        """
        model = eqx.combine(model_arrays, model_other)
        
        filter_spec = filter_spec_leaves(model, trainable_leaves_func)
        
        losses = jnp.empty((n_batches,))
        losses_terms = dict(zip(
            task.loss_func.weights.keys(), 
            [jnp.empty((n_batches,)) for _ in task.loss_func.weights]
        ))
        learning_rates = jnp.empty((n_batches,))
        
        loss_func_wrapped = grad_wrap_loss_func(task.loss_func)
        
        get_batch = jax.vmap(task.get_train_trial)
        
        start_batch = 0  # except when we load a checkpoint
        
        if restore_checkpoint:
            chkpt_path, last_batch, model, losses, losses_terms = \
                self._load_last_checkpoint(model, losses, losses_terms)
            start_batch = last_batch + 1
            logger.info(f"Restored checkpoint {chkpt_path} from training step {last_batch}.")
            
        elif self.checkpointing:
            # Delete old checkpoints if checkpointing is on.
            # TODO: keep old checkpoints for past N runs (env variable?)
            delete_contents(self.chkpt_dir)  
            
        # TODO: should also restore this from checkpoint
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))        

        # Passing the flattened pytrees through `train_step` gives a slight
        # performance improvement. See the docstring of `train_step`.
        flat_model, treedef_model = jax.tree_util.tree_flatten(model)
        flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)
         
        # Finish the JIT compilation before the first training iteration,
        # so the user will see it timed.
        for _ in tqdm(range(1), desc='compile', disable=disable_tqdm):
            keys = jr.split(key, batch_size)
            trial_specs, _ = get_batch(key=keys)
            self.train_step(  # doesn't alter model or opt_state
                flat_model, 
                treedef_model,
                flat_opt_state,
                treedef_opt_state,
                filter_spec, 
                trial_specs,
                loss_func_wrapped,
                keys, 
            )  
            tqdm.write(f"Training step compiled.", file=sys.stdout)
            task.eval(model, key)
            tqdm.write(f"Validation step compiled.", file=sys.stdout)

        keys = jr.split(key, n_batches)
        
        # Assume 1 epoch (i.e. batch iterations only; no fixed dataset).
        for batch in tqdm(
            range(start_batch, n_batches), 
            desc='train batch', 
            initial=start_batch, 
            total=n_batches,
            smoothing=0.1,
            disable=disable_tqdm,
        ):            
            key_train, key_eval, key_batch = jr.split(keys[batch], 3)
            # Sample 1) different model noise per trial, 2) different trials
            keys_train = jr.split(key_train, batch_size)  
            keys_trials = jr.split(key_batch, batch_size)  
            
            trial_specs, _ = get_batch(keys_trials)

            loss, loss_terms, flat_model, flat_opt_state, treedef_opt_state = self.train_step(
                flat_model, 
                treedef_model,
                flat_opt_state,
                treedef_opt_state,
                filter_spec, 
                trial_specs,
                loss_func_wrapped,
                keys_train, 
            )           
            
            if batch_callbacks is not None and batch in batch_callbacks:
                for func in batch_callbacks[batch]:
                    func()
            
            losses = losses.at[batch].set(loss)
            losses_terms = tree_set_idx(losses_terms, loss_terms, batch)
            try:
                # requires that the optimizer was wrapped in `optax.inject_hyperparameters`
                learning_rate = opt_state.hyperparams['learning_rate']
                learning_rates = learning_rates.at[batch].set(learning_rate)
            except (AttributeError, KeyError):
                learning_rate = None 
            
            if ensembled: 
                # ensemble training skips the logging and NaN control flow stuff
                continue
            
            # tensorboard losses on every iteration
            #! just report for one of the replicates
            if self._use_tb:
                self.writer.add_scalar('Loss/train', loss.item(), batch)
                for term, loss_term in loss_terms.items():
                    self.writer.add_scalar(f'Loss/train/{term}', loss_term.item(), batch)
        
            if jnp.isnan(loss):
                model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
                
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

            # checkpointing and validation occasionally
            if batch % log_step == 0:
                model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
                
                # model checkpoint
                if self.checkpointing:
                    eqx.tree_serialise_leaves(self.chkpt_dir / f'model{batch}.eqx', model)
                    eqx.tree_serialise_leaves(self.chkpt_dir / f'losses{batch}.eqx', 
                                              (losses, losses_terms))
                    with open(self.chkpt_dir / "last_batch.txt", 'w') as f:
                        f.write(str(batch)) 
                
                # tensorboard
                loss_val, loss_val_terms, states = task.eval(model, key_eval)
                
                if self._use_tb:
                    # TODO: register plots
                    # fig = make_eval_fig(states.effector, states.network.output, workspace)
                    # self.writer.add_figure('Eval/centerout', fig, batch)
                    self.writer.add_scalar('Loss/validation', loss_val.item(), batch)
                    for term, loss_term in loss_val_terms.items():
                        self.writer.add_scalar(f'Loss/validation/{term}', loss_term.item(), batch)
                    
                # TODO: https://stackoverflow.com/a/69145493
                tqdm.write(f"step: {batch}", file=sys.stdout)
                tqdm.write(f"\ttraining loss: {loss:.4f}", file=sys.stdout)
                tqdm.write(f"\tvalidation loss: {loss_val:.4f}", file=sys.stdout)
                if learning_rate is not None:                    
                    tqdm.write(f"\tlearning rate: {learning_rate:.4f}", file=sys.stdout)
         
        model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
         
        return model, losses, losses_terms, learning_rates
    
    @eqx.filter_jit
    @jax.named_scope("fbx.TaskTrainer.train_step")
    def train_step(
        self, 
        flat_model, 
        treedef_model,
        flat_opt_state, 
        treedef_opt_state,
        filter_spec, 
        trial_specs: AbstractTaskTrialSpec,
        loss_func_wrapped,
        keys: jax.Array,
    ):
        """Executes a single training step of the model.
        
        This assumes that the (wrapped) loss function returns an auxiliary
        `loss_terms`, which is intended to be a dictionary of the loss
        components prior to their final aggregation (i.e. already weighted).
              
        The wrapping calls to `tree_unflatten` and `tree_leaves`, and passing
        of flattened versions of `model` and `opt_state`, bring slight
        performance improvements because they cancel out the inverse tree
        operations that are performed by JIT compilation. 
        
        See https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops
        
        TODO: 
        - Use a wrapper to make the flatten/unflatten stuff less ugly.
        - The shape of `opt_state` changes due to `optimizer.update` on the 
          first step only. Why? 
        - Typing
        """
        
        model = jax.tree_util.tree_unflatten(treedef_model, 
                                             flat_model)
        opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, 
                                                 flat_opt_state)
        
        diff_model, static_model = eqx.partition(model, filter_spec)
        
        (loss, loss_terms), grads = eqx.filter_value_and_grad(
            loss_func_wrapped, has_aux=True
        )(
            diff_model, 
            static_model, 
            trial_specs,
            keys,
        )
        
        updates, opt_state = self.optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        flat_model = jax.tree_util.tree_leaves(model)
        flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)
        
        return loss, loss_terms, flat_model, flat_opt_state, treedef_opt_state

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
    - Typing
    - This is specific to `TaskTrainer`. Could it be more general?
      Note that the vmap here works the same as in `Task.eval`; perhaps if this
      is a `TaskTrainer`-specific function, then `Task` could provide an interface
    """
    @wraps(loss_func)
    def wrapper(
        diff_model, 
        static_model, 
        trial_specs: AbstractTaskTrialSpec,
        keys: jax.Array,
    ):
        model = eqx.combine(diff_model, static_model)
        #? will `in_axes` ever change? 
        states = jax.vmap(model)(trial_specs.input, trial_specs.init, keys)
        
        # TODO: loss_func should take task_input as well, probably
        return loss_func(states, trial_specs.target, trial_specs.input)
    
    return wrapper 


def save(
    tree: PyTree[eqx.Module],
    hyperparams: Optional[dict] = None, 
    path: Optional[Path] = None,
    save_dir: Path = Path('.'),
    suffix: Optional[str] = None,
):
    """Save a PyTree to disk along with hyperparameters.
    
    If a path is not specified, a filename will be generated from the current 
    time and the commit ID of the `feedbax` repository, and the file will be 
    saved in `save_dir`, which defaults to the current working directory.
    
    Assumes none of the hyperparameters are JAX arrays, as these are not 
    JSON serializable.
    
    Based on https://docs.kidger.site/equinox/examples/serialisation/
    
    TODO: 
    - If we leave in the git hash label, allow the user to specify the directory.
    """
    if path is None:      
        # TODO: move to separate function maybe
        timestr = datetime.today().strftime("%Y%m%d-%H%M%S") 
        commit_id = git_commit_id()
        name = f"model_{timestr}_{commit_id}"
        if suffix is not None:
            name += f"_{suffix}"
        path = save_dir / f'{name}.eqx'
    
    with open(path, 'wb') as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + '\n').encode())
        eqx.tree_serialise_leaves(f, tree)
    
    return path
    

def load(
    filename: Path | str, 
    setup_func: Callable[[Any], PyTree[eqx.Module]],
) -> PyTree[eqx.Module]:
    """Setup a PyTree of equinox modules from stored state and hyperparameters.
    
    TODO: 
    - Could provide a simple interface to show the most recently saved files in 
      a directory, according to the default naming rules from `save`.
    """
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        tree = setup_func(**hyperparams)
        tree = eqx.tree_deserialise_leaves(f, tree)
    
    return tree
    