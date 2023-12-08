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

from feedbax.loss import AbstractLoss, LossDict
import feedbax.plot as plot
from feedbax.task import AbstractTask, AbstractTaskTrialSpec
from feedbax.utils import (
    delete_contents,
    filter_spec_leaves, 
    git_commit_id,
    tree_get_idx,
    tree_set_idx,
)

if TYPE_CHECKING:
    # This is sloow so we'll actually import it only when needed.
    from torch.utils.tensorboard import SummaryWriter


LOSS_FMT = ".2e"


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
    
    @jax.named_scope("fbx.TaskTrainer")
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
        disable_tqdm = False,
        ensembled: bool = False,
    ):
        """Train a model on a task for a fixed number of batches of trials.
        
        NOTE:
        - Model checkpointing only saves model state, not the task or 
          hyperparameters. That is, it assumes that the model and task passed 
          to this method are, aside from their trainable state, identical to 
          those from the time of checkpointing. This is typically the case 
          when a checkpoint is used locally to resume training. However, trying
          to load a checkpoint as a model later may fail. Use `save` and `load`.
        
        TODO:
        - check that `trainable_leaves_func` contains only trainable stuff   
        - Improve the handling of the flatten/unflatten operations around 
          `train_step`. See its docstring for details. 
        - The first iteration (or two) are much slower due to JIT compilation
          of `train_step`, which distorts the smoothed it/s estimate of tqdm. 
          Also, `opt_state` seems to change shape only on the first step call 
          to `optimizer.update`, which is why we need to recompute and return 
          `treedef_opt_state` from `train_step`. So maybe the first call 
          should be separated out from the loop.
        """          
        if ensembled:
            # Infer the number of replicates from shape of trainable arrays
            n_replicates = jax.tree_leaves(
                eqx.filter(model, eqx.is_array)
            )[0].shape[0]
            loss_array_shape = (n_batches, n_replicates)
            opt_state = jax.vmap(self.optimizer.init)(
                eqx.filter(model, eqx.is_array)
            )
        else:
            loss_array_shape = (n_batches,)
            opt_state = self.optimizer.init(
                eqx.filter(model, eqx.is_array)
            )
        
        losses_history = LossDict(zip(
            task.loss_func.weights.keys(), 
            [jnp.empty(loss_array_shape) for _ in task.loss_func.weights],
        ))
        learning_rates = jnp.empty(loss_array_shape)
        
        start_batch = 0  # except when we load a checkpoint
        
        if restore_checkpoint:
            # TODO: should also restore opt_state, learning_rates...
            chkpt_path, last_batch, model, losses_history = \
                self._load_last_checkpoint(model, losses_history)
            start_batch = last_batch + 1
            logger.info(f"Restored checkpoint {chkpt_path} from training step {last_batch}.")
            
        elif self.checkpointing:
            # Delete old checkpoints if checkpointing is on.
            # TODO: keep old checkpoints for past N runs (env variable?)
            delete_contents(self.chkpt_dir)  

        # Passing the flattened pytrees through `train_step` gives a slight
        # performance improvement. See the docstring of `train_step`.
        flat_model, treedef_model = jax.tree_util.tree_flatten(model)
        flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)
         
        if ensembled:
            # We only vmap over axis 0 of the *array* components of the model.
            flat_model_array_spec = jax.tree_map(
                lambda x: 0 if eqx.is_array(x) else None, 
                flat_model,
            )
            train_step = eqx.filter_vmap(
                self.train_step, 
                in_axes=(None, None, flat_model_array_spec, None, 0, None, None, 0), 
                #out_axes=(0, 0, 0, None),
            )
            
            # We can't simply flatten this to get `flat_model_array_spec`,
            # even if we use `is_leaf=lambda x: x is None`.
            model_array_spec = jax.tree_map(
                lambda x: 0 if eqx.is_array(x) else None, 
                model,
            )
            evaluate = eqx.filter_vmap(task.eval, in_axes=(model_array_spec, 0)) 
        else:
            train_step = self.train_step
            evaluate = task.eval
        
        filter_spec = filter_spec_leaves(model, trainable_leaves_func)
           
        # Finish the JIT compilation before the first training iteration.
        if not jax.config.jax_disable_jit:
            for _ in tqdm(range(1), desc='compile', disable=disable_tqdm):
                if ensembled:
                    key_compile = jr.split(key, n_replicates)                
                else:
                    key_compile = key
                
                train_step(  # doesn't alter model or opt_state
                    task,
                    batch_size,
                    flat_model, 
                    treedef_model,
                    flat_opt_state,
                    treedef_opt_state,
                    filter_spec, 
                    key_compile, 
                )  
                if not disable_tqdm:
                    tqdm.write(f"Training step compiled.", file=sys.stdout)
                
                evaluate(model, key_compile)
                if not disable_tqdm:
                    tqdm.write(f"Validation step compiled.", file=sys.stdout)
        else:
            logger.debug("JIT globally disabled, skipping pre-run compilation")

        keys = jr.split(key, n_batches)
        
        #! The first step of a run is still slow, even with the above warmup.
        #! I suspect this might be due to re-compilation that is happening 
        #! due to the structure of `opt-state` changing *only on the first
        #! step*. Perhaps we can either 1) figure out why `opt_state` changes,
        #! and perhaps avoid that, or 2) do the first two actual steps inside 
        #! the warmup loop, and then continue here from step 3, to give the 
        #! user more appropriate timing estimates.
        # Assume 1 epoch (i.e. batch iterations only; no fixed dataset).
        for batch in tqdm(
            range(start_batch, n_batches), 
            desc='train batch', 
            initial=start_batch, 
            total=n_batches,
            smoothing=0.1,
            disable=disable_tqdm,
        ):            
            key_train, key_eval = jr.split(keys[batch], 2) 
            
            if ensembled:
                key_train = jr.split(key_train, n_replicates)

            losses, flat_model, flat_opt_state, treedef_opt_state = train_step(
                task,
                batch_size,
                flat_model, 
                treedef_model,
                flat_opt_state,
                treedef_opt_state,
                filter_spec, 
                key_train, 
            )           
            
            if batch_callbacks is not None and batch in batch_callbacks:
                for func in batch_callbacks[batch]:
                    func()
            
            losses_history = tree_set_idx(losses_history, losses, batch)
            try:
                # requires that the optimizer was wrapped in `optax.inject_hyperparameters`
                learning_rate = opt_state.hyperparams['learning_rate']
                learning_rates = learning_rates.at[batch].set(learning_rate)
            except (AttributeError, KeyError):
                learning_rate = None 
            
            # tensorboard losses on every iteration
            if ensembled:
                losses_mean = jax.tree_map(lambda x: jnp.mean(x, axis=-1), 
                                           losses)
                # This will be appended to user-facing labels
                # e.g. "mean training loss"
                ensembled_str = "mean "
            else:
                losses_mean = losses
                ensembled_str = ""
            
            
            if self._use_tb:
                self.writer.add_scalar(
                    f'loss/{ensembled_str}train', 
                    losses_mean.total.item(), 
                    batch
                )
                for loss_term_label, loss_term in losses_mean.items():
                    self.writer.add_scalar(
                        f'loss/{ensembled_str}train/{loss_term_label}', 
                        loss_term.item(), 
                        batch
                    )
        
            if jnp.isnan(losses_mean.total):
                model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
                
                msg = f"NaN loss at batch {batch}! "
                if (checkpoint := self._load_last_checkpoint(
                    model, losses_history
                )) is not None:
                    _, last_batch, model, losses_history = checkpoint
                    msg += f"Returning checkpoint from batch {last_batch}."
                else:
                    msg += "No checkpoint found, returning model from final iteration."
                
                logger.warning(msg)
                
                return model, losses_history, learning_rates

            # checkpointing and validation occasionally
            if batch % log_step == 0:
                model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
                
                # model checkpoint
                if self.checkpointing and batch > 0:
                    eqx.tree_serialise_leaves(self.chkpt_dir / f'model{batch}.eqx', model)
                    eqx.tree_serialise_leaves(self.chkpt_dir / f'losses{batch}.eqx', 
                                              losses_history)
                    with open(self.chkpt_dir / "last_batch.txt", 'w') as f:
                        f.write(str(batch)) 
                
                if ensembled:
                    key_eval = jr.split(key_eval, n_replicates)
                
                losses_val, states = evaluate(model, key_eval)
                
                if ensembled:
                    losses_val_mean = jax.tree_map(lambda x: jnp.mean(x, axis=-1), 
                                                   losses_val)
                    # Only log a validation plot for the first replicate.
                    states_plot = tree_get_idx(states, 0)            
                else:
                    losses_val_mean = losses_val
                    states_plot = states
                
                if self._use_tb:
                    # TODO: register plots instead of hard-coding
                    trial_specs, _ = task.trials_validation
                    fig, _ = plot.plot_pos_vel_force_2D(
                        states_plot,
                        endpoints=(
                            trial_specs.init['mechanics']['effector'].pos, 
                            trial_specs.goal.pos
                        ),
                        workspace=task.workspace,
                    )
                    self.writer.add_figure('validation/centerout', fig, batch)
                    self.writer.add_scalar(
                        f'loss/{ensembled_str}validation', 
                        losses_val_mean.total.item(), 
                        batch
                    )
                    for loss_term_label, loss_term in losses_val_mean.items():
                        self.writer.add_scalar(
                            f'loss/{ensembled_str}validation/{loss_term_label}', 
                            loss_term.item(), 
                            batch
                        )
                    
                # TODO: https://stackoverflow.com/a/69145493
                if not disable_tqdm:
                    tqdm.write(f"\nTraining iteration: {batch}", file=sys.stdout)
                    tqdm.write(f"\t{ensembled_str}training loss: "
                               + f"{losses_mean.total:{LOSS_FMT}}".capitalize(), 
                               file=sys.stdout)
                    tqdm.write(f"\t{ensembled_str}validation loss: "
                               + f"{losses_val_mean.total:{LOSS_FMT}}".capitalize(), 
                               file=sys.stdout)
                    # if learning_rate is not None:                    
                    #     tqdm.write(f"\tlearning rate: {learning_rate:.4f}", file=sys.stdout)
         
        model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
         
        return model, losses_history, learning_rates
    
    @eqx.filter_jit
    @jax.named_scope("fbx.TaskTrainer.train_step")
    def train_step(
        self, 
        task,
        batch_size,
        flat_model, 
        treedef_model,
        flat_opt_state, 
        treedef_opt_state,
        filter_spec, 
        key: jax.Array,
    ):
        """Executes a single training step of the model.
        
        Note that the primary output of `loss_func_wrapped` is tht scalar 
        `losses.total`, which is discarded because `losses` is itself returned
        as the auxiliary of `loss_func_wrapped`. This is necessary because 
        the gradient is computed with respect to the primary output, but we 
        only need to store the original `LossDict` containing all loss terms.
              
        The wrapping calls to `tree_unflatten` and `tree_leaves`, and passing
        of flattened versions of `model` and `opt_state`, bring slight
        performance improvements because they cancel out the inverse tree
        operations that would be performed by default during JIT compilation. 
        
        See https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops
        
        TODO: 
        - Use a wrapper to make the flatten/unflatten stuff less ugly.
        - The shape of `opt_state` changes due to `optimizer.update` on the 
          first step only. Why? 
        - Typing of arguments.
        """
        key_trials, key_model = jr.split(key, 2)
        keys_trials = jr.split(key_trials, batch_size)  
        keys_model = jr.split(key_model, batch_size)  
        
        trial_specs, _ = jax.vmap(task.get_train_trial)(keys_trials)
        
        model = jax.tree_util.tree_unflatten(treedef_model, 
                                             flat_model)
        diff_model, static_model = eqx.partition(model, filter_spec)
        
        opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, 
                                                 flat_opt_state)
        
        (_, losses), grads = eqx.filter_value_and_grad(
            grad_wrap_loss_func(task.loss_func), has_aux=True
        )(
            diff_model, 
            static_model, 
            trial_specs,
            keys_model,
        )
        
        updates, opt_state = self.optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        flat_model = jax.tree_util.tree_leaves(model)
        flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)
        
        return losses, flat_model, flat_opt_state, treedef_opt_state

    def _load_last_checkpoint(self, model, losses_history):
        try:
            with open(self.chkpt_dir / "last_batch.txt", 'r') as f:
                last_batch = int(f.read()) 
        except FileNotFoundError:
            #! this will just raise another error at checkpoint restore
            return
            
        chkpt_path = self.chkpt_dir / f'model{last_batch}.eqx'
        model = eqx.tree_deserialise_leaves(chkpt_path, model)
        losses_history = eqx.tree_deserialise_leaves(
            self.chkpt_dir / f'losses{last_batch}.eqx', 
            losses_history,
        )
        return chkpt_path, last_batch, model, losses_history
    
    
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
    ) -> Tuple[float, LossDict]:
        model = eqx.combine(diff_model, static_model)
        #? will `in_axes` ever change? 
        states = jax.vmap(model)(trial_specs.input, trial_specs.init, keys)
        
        losses = loss_func(states, trial_specs.target, trial_specs.input)
        
        return losses.total, losses 
    
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
    