# Learning with Differentiable Physics

Training neural controllers for biomechanical systems requires choosing (a) a learning paradigm, (b) a physics simulation backend, and (c) strategies for managing gradient flow through time. These choices interact: certain backends enable certain paradigms, and gradient management techniques expand the feasible horizon for backpropagation through physics. This article surveys the landscape of methods at the intersection of differentiable simulation and controller learning, with emphasis on biomechanical and musculoskeletal applications.

---

## Learning Paradigms

### Model-Free Reinforcement Learning

Model-free reinforcement learning (RL) treats the simulator as a black box. The policy gradient is estimated via the REINFORCE / likelihood ratio trick (a zeroth-order method): the agent collects trajectory rollouts, scores them by cumulative reward, and adjusts the policy to increase the probability of high-reward action sequences. No gradient flows through the physics — only through the policy network and the reward-to-log-probability product.

Representative algorithms include PPO (Schulman et al., 2017) and SAC (Haarnoja et al., 2018). Because the gradient estimator treats physics as opaque, it works with *any* simulator regardless of differentiability. This generality comes at a cost: REINFORCE-style estimators have high variance, requiring millions of environment steps for convergence on tasks of moderate complexity.

!!! info "Contact robustness"
    The zeroth-order gradient estimate is unaffected by discontinuities in the dynamics (contacts, friction transitions, joint limits). This makes model-free RL the default choice for contact-rich tasks — locomotion, dexterous manipulation — especially when the simulator is not differentiable or when stiff contacts would corrupt first-order gradients.

**Typical use**: locomotion and manipulation in standard simulators (MuJoCo, Isaac Gym, Brax). Model-free RL remains the dominant paradigm for tasks where contacts are frequent and the simulator does not expose gradients.

### Differentiable RL: Short-Horizon Actor-Critic Methods

When a differentiable simulator is available, it becomes possible to backpropagate the policy gradient *through* the physics for a short window, then use a learned value function to handle the remainder of the episode. This family of methods achieves substantially higher sample efficiency than model-free RL while avoiding the instability of backpropagating through hundreds of timesteps.

#### SHAC

Short-Horizon Actor-Critic (SHAC; Xu et al., 2022) divides the full episode (e.g., $H = 1000$ for locomotion) into short windows of length $h$ (typically 32). Within each window, the policy gradient is computed via first-order backpropagation through the differentiable simulator — gradients flow through `sim.step()` for $h$ consecutive steps. At the window boundary, a learned value function $V_\phi(s)$ bootstraps the return beyond the short horizon:

$$J(\theta) = \sum_{t=0}^{h-1} \gamma^t r_t + \gamma^h V_\phi(s_h)$$

The gradient $\nabla_\theta J$ flows backward through the reward computation, through each physics step, and through the policy network, for all $h$ steps. The value function provides the tail estimate for the remaining $H - h$ steps.

??? info "Extended: SHAC training details"
    **Critic training** proceeds separately using zeroth-order TD-style regression on collected rollouts (16 critic update iterations per batch). SHAC is therefore a *mixed-order* method: first-order (analytic) gradients for the actor over the short horizon, zeroth-order (sample-based) gradients for the critic.

    **Why the short horizon works**: The value function absorbs the noisy, discontinuous landscape over long horizons into a smooth function approximation. The short differentiable window avoids vanishing and exploding gradients from deeply nested autodiff chains. Up to 1000 parallel environments on GPU provide the data throughput needed for stable critic training.

    **Demonstrated tasks**: CartPole, Ant, Humanoid, and SNU Humanoid MTU (152 muscle-tendon actuators). On the muscle-tendon humanoid, SHAC achieves a $>17\times$ training time reduction compared to the best RL baseline.

#### AHAC

Adaptive Horizon Actor-Critic (AHAC; Georgiev et al., 2024) extends SHAC with an *adaptive* horizon per trajectory segment. Rather than fixing $h = 32$, AHAC truncates the differentiable rollout when it detects stiff dynamics (contact events), falling back to the value function earlier. In regions of smooth dynamics, the horizon can extend further, extracting more gradient signal before bootstrapping.

AHAC reports 40% more reward than PPO on locomotion tasks and three orders of magnitude better sample efficiency.

#### Theoretical Grounding

Suh et al. (2022) provide the analytical framework for understanding these trade-offs. First-order gradient estimators (backprop through physics) have lower variance than zeroth-order estimators (REINFORCE), but can be *biased* in the presence of stiff dynamics and discontinuities. They propose an $\alpha$-order estimator ($\alpha \in [0, 1]$) that interpolates continuously between first-order efficiency and zeroth-order robustness. The SHAC/AHAC approach can be understood as a practical instantiation of this trade-off: first-order within the window, zeroth-order (via the value function) beyond it.

### Analytic Policy Gradient

Analytic Policy Gradient (APG; Wiedemann et al., 2023) backpropagates through a differentiable simulator for a fixed horizon $T$, with no value function. The paper evaluates $T \in \{1, 3, 5, 8, 10, 12, 15\}$ and finds $T = 10$ optimal for the tested domains. Two action-generation strategies are compared:

- **Concurrent**: The policy outputs $T$ actions simultaneously; they are applied sequentially through the dynamics. Gradients pass $T$ times through the dynamics but only once through the policy network.
- **Autoregressive**: Actions are predicted one at a time with state feedback at each step. This produces longer gradient chains (through both dynamics and policy at every step), making it more prone to gradient instability.

!!! note "Curriculum learning for stability"
    APG relies on a curriculum over task difficulty rather than gradient clipping for training stability. A divergence threshold $\tau_\mathrm{div}$ controls the maximum allowable deviation from a reference trajectory. Training begins at $\tau_\mathrm{div} = 0.1\,\mathrm{m}$ and increases by $0.05\,\mathrm{m}$ every 5 epochs until reaching $2\,\mathrm{m}$. The reference speed also progresses through 50%, 75%, and 100% of the target. This gradual increase in difficulty is the primary mechanism preventing gradient explosion — the policy never encounters dynamics far from what it has already learned to handle.

APG is demonstrated on quadrotor and fixed-wing UAV control using analytical (PyTorch-coded) dynamics, not a general-purpose physics engine. The resulting controllers achieve MPC-level tracking accuracy with $>10\times$ faster inference.

### Full-Episode Supervised Backpropagation

MotorNet (Codol et al., 2024) takes the direct approach: the controller (typically a GRU with $\sim$50 hidden units) and the biomechanical plant form a single differentiable computation graph. At each timestep the GRU receives proprioceptive feedback and task information, outputs muscle excitations, and the plant integrates one step (Euler, $\mathrm{d}t = 0.01\,\mathrm{s}$). The new state feeds back to the GRU. This loop runs for the full episode (80--100 timesteps), and the total loss is backpropagated through the entire unrolled graph — through all RNN steps *and* all plant dynamics steps. This is standard backpropagation through time (BPTT) applied jointly to controller and plant.

??? info "Extended: Why full-episode BPTT is feasible here"
    Three properties make full-episode backpropagation tractable in MotorNet:

    1. **No contact dynamics.** The plant consists of smooth ODEs throughout — no collision detection, no constraint solving, no discontinuities.
    2. **Analytical dynamics.** The physics is purpose-built biomechanics (Lagrangian ODEs with Hill-type muscle models), not a general-purpose solver with iterative constraint resolution.
    3. **Short episodes.** At 80--100 timesteps, the computation graph remains shallow enough that gradients do not vanish or explode catastrophically.

??? info "Extended: MotorNet training configuration"
    For the arm26 model (2 DOF, 6 muscles):

    | Parameter | Value |
    |-----------|-------|
    | Loss | ClippedPositionLoss($\alpha=2$) + activation penalty($\beta=5$) + hidden state penalty($\gamma=0.1$) + weight regularization($\kappa=0.05$) |
    | Batch size | 64--1024 episodes |
    | Training duration | 7,680--38,400 batches (task-dependent) |
    | Learning rate | $10^{-4}$ (TensorFlow) or $10^{-3}$ (PyTorch), Adam |
    | Gradient clipping | `max_norm=1.0` |
    | Wall-clock time | ~13 minutes for the curl-field task (M1 Max) |

### Model Predictive Control with Differentiable Dynamics

DiffMJX (Paulus et al., 2025) demonstrates gradient-based shooting MPC using a differentiable physics backend. At each decision point:

1. **Initialize** a control sequence over a planning horizon (e.g., 256 steps), warm-started from the previous solution.
2. **Optimize** the action sequence over multiple iterations of Adam by backpropagating a cost function through the DiffMJX simulation.
3. **Execute** the first portion of the plan (e.g., 16 steps) in the environment.
4. **Replan** from the new observed state.

No policy network is involved — the action sequence itself is the optimization variable. This is online trajectory optimization, not amortized policy learning. Because the full computation budget is spent at runtime for each planning cycle, the approach is expensive at inference time, but it can handle complex dynamics (contacts, muscle-tendon actuators) because only a rough gradient direction is needed for each optimization step.

DiffMJX demonstrates this on MyoHand dexterous manipulation (29 bones, 23 joints, 39 muscle-tendon units) and bionic tennis (63 muscle-tendon units).

### System Identification via Differentiable Simulation

Differentiable simulators also enable gradient-based system identification. Given observed trajectories (e.g., a cube tossed onto a table), one can simulate forward through the differentiable simulator from an initial state, compute an $L_2$ loss between predicted and observed states, and backpropagate to update model parameters — masses, friction coefficients, geometry parameters. Because ground truth states are available at every timestep, very short prediction windows suffice (e.g., 4 steps in the ContactNets comparison from Paulus et al., 2025), and the optimization window can slide across the trajectory.

This is not controller learning but shares the same computational substrate: first-order gradients through a physics simulation.

### Additional Methods

#### SAPO

Smooth Analytic Policy Optimization (SAPO; Xing et al., 2025) adds maximum entropy regularization to the first-order model-based RL objective. The entropy term smooths the optimization landscape, stabilizing training across random seeds — a persistent challenge for first-order methods, which can be sensitive to initialization. SAPO is demonstrated on rigid bodies, soft bodies, and fluids.

#### DMO

Differentiable Model Optimization (DMO; Amigo et al., 2025) decouples trajectory generation from gradient computation. Trajectories are unrolled in a high-fidelity (potentially non-differentiable) simulator, while gradients are computed via backpropagation through a learned differentiable surrogate model. This enables first-order policy optimization even when the true simulator's gradients are unavailable or biased due to contacts and discontinuities.

---

## Physics Simulation Backends

### MuJoCo / MJX

MuJoCo (Multi-Joint dynamics with Contact; Todorov et al., 2012) is a general-purpose physics engine widely used in robotics and biomechanics. MJX is its JAX-compiled variant, enabling hardware acceleration (GPU/TPU) and JAX automatic differentiation.

**Integration schemes**: Semi-implicit Euler (default), RK4, and IMPLICITFAST (implicit-in-velocity, dropping Coriolis and centripetal terms). Semi-implicit Euler updates velocity first ($v_{t+h} = v_t + h \cdot a_t$), then position with the new velocity ($q_{t+h} = q_t + h \cdot v_{t+h}$). IMPLICITFAST offers similar cost to Euler with improved stability, and is recommended for most models.

**Constraint solver**: Newton (GPU) or CG (TPU). Contact forces, joint limits, and tendon constraints are resolved as a convex optimization problem. The solver is iterative, using `jax.lax.while_loop` with dynamic termination for convergence.

!!! note "Differentiability of the constraint solver"
    The iterative constraint solver's `while_loop` with dynamic termination is **not differentiable** in reverse mode by default. Setting `iterations=1` and `ls_iterations=0` (and disabling contacts) eliminates the `while_loop`, replacing it with a single Newton step and making `mjx.forward()` fully differentiable. Even with these settings, gradient quality degrades over long horizons due to the accumulated complexity of the forward computation.

**Contact model**: MuJoCo uses a soft constraint (penalty-based) contact model rather than hard complementarity. An impedance parameter $d \in (0, 1)$ interpolates between a nearly hard constraint ($R \to 0$) and an infinitely soft one ($R \to \infty$). Gradients exist everywhere, but they can be uninformative when stiffness is high (the gradient of a near-step-function is near-zero almost everywhere).

**Sub-stepping**: At a control timestep of $\mathrm{d}t = 0.01\,\mathrm{s}$, the single-Newton-step approximation is too coarse for stable musculoskeletal dynamics, requiring sub-stepping (e.g., $\mathrm{d}t = 0.002\,\mathrm{s}$ with 5 substeps per control step). This multiplies the backpropagation depth proportionally.

### Analytical Lagrangian Dynamics

Purpose-built ODEs derived from first-principles Lagrangian mechanics. For a two-link planar arm:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + B\dot{q} = \tau$$

where $M(q)$ is the configuration-dependent mass matrix, $C(q, \dot{q})$ captures Coriolis and centripetal terms, $B$ is joint viscous friction, and $\tau$ is the vector of applied joint torques.

**Muscle models**: Hill-type with rigid tendon, as in the MotorNet formulation. Muscle force is a function of activation $a$, fiber length $l_m$, and contraction velocity $v_m$. A moment arm matrix $R \in \mathbb{R}^{n_\mathrm{muscles} \times n_\mathrm{joints}}$ maps muscle forces to joint torques:

$$\tau = R^\top f_\mathrm{muscle}(a, l_m, v_m)$$

Every operation is a smooth, explicitly coded function — no solver iterations, no constraint resolution, no discontinuities. The Jacobian of acceleration with respect to state is the derivative of $M(q)^{-1}[\tau - C(q, \dot{q})\dot{q} - B\dot{q}]$, which is well-conditioned for typical musculoskeletal systems.

!!! info "Limitations"
    Analytical dynamics must be derived and implemented per body topology. There is no general contact handling. Scaling to many-DOF systems requires efficient mass matrix inversion ($O(n^3)$ for dense systems, though recursive $O(n)$ algorithms exist for tree topologies).

### Diffrax ODE Integration

Diffrax (Kidger, 2022) provides ODE and SDE solvers in JAX with full automatic differentiation support. It serves as either (a) the sole integrator for analytical dynamics (replacing Euler with higher-order methods), or (b) a wrapper around another simulator's `forward()` function to provide adaptive stepping and clean adjoint computation.

**Available solvers** (selected):

| Solver | Order | Notes |
|--------|-------|-------|
| `Euler` | 1 | Simplest; rarely recommended for production |
| `Heun` | 2 | Recommended for neural ODEs |
| `Tsit5` | 5(4) | Tsitouras method; slightly more efficient than Dopri5 |
| `Dopri5` | 5(4) | Dormand-Prince; the classic adaptive solver |
| `Dopri8` | 8(7) | For high-accuracy requirements |

**Adaptive step control**: `PIDController` adjusts step size to match a specified error tolerance. It automatically refines during stiff or fast-changing dynamics and coarsens in smooth regions. Adaptive stepping is almost always recommended over fixed stepping.

**Backpropagation**: The default strategy is `RecursiveCheckpointAdjoint` — discretize-then-optimize with binomial checkpointing for $O(\log n)$ memory and $O(n \log n)$ compute. An alternative, `BacksolveAdjoint` (optimize-then-discretize), solves the adjoint ODE backward in time, producing approximate gradients; it is generally not recommended due to numerical drift.

### DiffMJX: Adaptive Integration for MJX

DiffMJX (Paulus et al., 2025) wraps MJX with Diffrax adaptive ODE integration, addressing two problems:

**Adaptive integration**: DiffMJX automatically refines internal timesteps during contact-rich phases and coarsens during smooth dynamics. The integrator is extended to handle quaternion normalization and stateful actuators (muscle-tendon units). The external interface preserves a fixed-timestep API; only internal stepping is adaptive. Memory is managed via Diffrax's recursive checkpoint adjoint.

**Contacts From Distance (CFD)**: Standard penalty contacts produce zero gradient when bodies are not in contact — the signed distance function passes through a ReLU, whose gradient is zero for positive distances. CFD extends MuJoCo's impedance function to positive distances using a softplus activation (replacing the ReLU on signed distance). This extension is applied only in the backward pass via a straight-through estimator: forward simulation remains physically accurate, while the backward pass sees smooth gradients even before contact occurs.

---

## Gradient Management Across Time

### The Horizon Problem

Backpropagating through $T$ steps of physics simulation requires computing the product of $T$ Jacobians:

$$\frac{\partial s_T}{\partial s_0} = \prod_{t=1}^{T} \frac{\partial s_t}{\partial s_{t-1}}$$

For smooth dynamics, the spectral radius of each Jacobian determines whether this product vanishes (eigenvalues consistently $< 1$) or explodes (eigenvalues consistently $> 1$) exponentially with $T$. The practical limit depends on the conditioning of the dynamics:

| Dynamics type | Feasible horizon | Example |
|---------------|-----------------|---------|
| Contact-rich (locomotion, manipulation) | ~16--64 steps | SHAC uses $h = 32$ |
| Contact-free, analytical (musculoskeletal reaching) | ~80--200 steps | MotorNet uses 80--100 |
| Stiff constraints (MJX with high penalty stiffness) | Gradients can be uninformative even at short horizons | — |

### Truncation and Value Function Bootstrap

The SHAC/AHAC approach: backpropagate through a short differentiable window, use a learned value function to estimate the return beyond that window. This is a *mixed-order* estimator — first-order (analytic) within the window, zeroth-order (sample-based) beyond.

The trade-off is direct: shorter windows produce more stable gradients but rely more heavily on value function accuracy. Longer windows extract richer gradient signal but increase the risk of vanishing or exploding gradients. AHAC makes this trade-off adaptive, truncating earlier when dynamics are stiff and extending further when they are smooth.

### Gradient Clipping

Gradient clipping truncates the gradient vector when its norm exceeds a threshold:

$$\tilde{g} = \begin{cases} g & \text{if } \|g\| \leq c \\ c \cdot \frac{g}{\|g\|} & \text{if } \|g\| > c \end{cases}$$

This prevents explosion but introduces bias — clipped gradients may point in a different direction from the true gradient. In JAX, this is typically applied via `optax.clip_by_global_norm(max_norm)`. Gradient clipping is widely used in conjunction with other techniques (truncation, curriculum learning) rather than as a standalone solution.

### Curriculum on Horizon and Difficulty

Rather than directly increasing the backpropagation horizon during training, APG (Wiedemann et al., 2023) uses a curriculum on task difficulty: the divergence threshold $\tau_\mathrm{div}$ increases gradually, and the reference trajectory speed progresses from 50% to 100%. The effect is similar to a horizon curriculum — the policy first learns to handle short, easy trajectory segments, then progressively longer and more challenging ones — but the mechanism operates through task design rather than autodiff depth.

No published method applies an explicit curriculum on backpropagation depth itself (e.g., starting with $H = 4$ and increasing to $H = 64$), though the principle is sound and could be combined with value function bootstrapping.

### Checkpointing and Memory

Full-episode backpropagation through $T$ physics steps requires storing intermediate states for the backward pass. Three strategies are common:

| Strategy | Memory | Compute overhead | Mechanism |
|----------|--------|-----------------|-----------|
| No checkpointing | $O(T)$ | None | Store all intermediate states during forward pass |
| `jax.checkpoint` (remat) | Configurable | Recomputes selected segments | Manually annotate which forward segments to recompute during backward pass |
| Diffrax `RecursiveCheckpointAdjoint` | $O(\log T)$ | $O(T \log T)$ | Binomial checkpointing — optimal trade-off for long sequences |

For short episodes (80--100 steps), no checkpointing is typically sufficient. For longer horizons or memory-constrained settings (e.g., batched training across many parallel bodies), checkpointing becomes essential.

---

## Summary

| Method | Gradient source | Differentiable horizon | Contact handling | Produces amortized policy? | Sample efficiency |
|--------|----------------|----------------------|-----------------|---------------------------|-------------------|
| PPO / SAC | Zeroth-order (REINFORCE) | Full episode | Robust (black-box) | Yes | Low |
| SHAC / AHAC | First-order (short) + zeroth-order (value fn) | 16--64 steps | Limited (gradient bias near contacts) | Yes | High |
| APG | First-order (fixed) | 10--15 steps | N/A (smooth dynamics only) | Yes | High |
| Full-episode BPTT | First-order (full) | 80--100+ steps | None (smooth dynamics only) | Yes | Highest |
| Differentiable MPC | First-order (planning horizon) | ~256 steps | Good (with CFD) | No (online optimization) | N/A |
| System identification | First-order (short) | 4+ steps | Good (with CFD) | No | N/A |

The choice of paradigm depends on the dynamics regime. Contact-rich tasks with long horizons favor model-free RL or short-horizon differentiable RL (SHAC/AHAC). Smooth biomechanical dynamics with short episodes permit full-episode backpropagation (MotorNet-style). Mixed regimes — biomechanical models with intermittent contacts — can benefit from DiffMJX's CFD approach combined with adaptive horizon methods. In all cases, the simulation backend constrains what is possible: differentiable methods require a differentiable simulator, and the quality of that simulator's gradients determines the practical horizon for backpropagation.

---

## References

- Amigo, I., Surovik, D., Choi, C., & Coros, S. (2025). Differentiable Model Optimization for Robotics. *CoRL 2025*. [arXiv:2509.00215](https://arxiv.org/abs/2509.00215)
- Codol, O., Michaels, J. A., Bhagat, J., Cline, C., Bhargava, H., & Miller, L. E. (2024). MotorNet, a Python toolbox for controlling differentiable biomechanical effectors with artificial neural networks. *eLife*, 13, e88591. [doi:10.7554/eLife.88591](https://doi.org/10.7554/eLife.88591)
- Freeman, C. D., Frey, E., Raichuk, A., Girber, S., Mordatch, I., & Bachem, O. (2021). Brax — A Differentiable Physics Engine for Large Scale Rigid Body Simulation. *NeurIPS 2021*. [arXiv:2106.13281](https://arxiv.org/abs/2106.13281)
- Georgiev, V., Schmid, L., Mattamala, M., Dharmadhikari, M., & Hutter, M. (2024). Adaptive Horizon Actor-Critic for Policy Learning in Contact-Rich Differentiable Simulation. *ICML 2024*. [arXiv:2405.17784](https://arxiv.org/abs/2405.17784)
- Kidger, P. (2022). On Neural Differential Equations. *PhD thesis, University of Oxford*. [arXiv:2202.02435](https://arxiv.org/abs/2202.02435)
- Paulus, D., et al. (2025). DiffMJX: Differentiable Physics with MuJoCo and JAX. *Under review, ICLR 2026*. [arXiv:2506.14186](https://arxiv.org/abs/2506.14186)
- Suh, H. J., Simchowitz, M., Zhang, K., & Tedrake, R. (2022). Do Differentiable Simulators Give Better Policy Gradients? *ICML 2022*. [arXiv:2202.00817](https://arxiv.org/abs/2202.00817)
- Wiedemann, T., Gilhyun, R., Solowjow, F., & Trimpe, S. (2023). Training Efficient Controllers via Analytic Policy Gradient. *ICRA 2023*. [arXiv:2209.13052](https://arxiv.org/abs/2209.13052)
- Xing, J., Zheng, L., Xu, Z., Qiao, Y.-L., & Zhu, C. (2025). SAPO: Stabilized Analytic Policy Optimization via Maximum Entropy Regularization. *ICLR 2025 (Spotlight)*. [arXiv:2412.12089](https://arxiv.org/abs/2412.12089)
- Xu, J., Makoviichuk, V., Narang, Y., Ramos, F., Matusik, W., Garg, A., & Macklin, M. (2022). Accelerated Policy Learning with Parallel Differentiable Simulation. *ICLR 2022*. [arXiv:2204.07137](https://arxiv.org/abs/2204.07137)
