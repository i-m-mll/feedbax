# Authored MuJoCo models

Planar models use the acausal graph's `PlanarLink`, `RevoluteJoint`, `WorldFrame`, and `Anchor` nodes. The supported assembly is a single serial chain with one hinge per link and a world-rooted pivot. Connections, when present, must agree with the joint and anchor frame names. `planar_chain_graph` creates this representation from a body preset.

A link's proximal origin is local zero and its distal frame is at `(length, 0)`. Length, frame offsets and transverse COM offset are in metres, and mass is in kilograms. The `com` field measures the longitudinal COM as a fraction of link length. `inertia` is the moment about local Z through the COM, in kg·m². The native body uses explicit principal moments `(Izz/2, Izz/2, Izz)`. Those transverse moments complete the planar rigid-body representation for MuJoCo. The graph's hinge degrees of freedom restrict motion to the plane.

A revolute joint aligns its parent and child frames after applying the authored parent-frame offset and the zero-angle frame rotation. Its axis is local +Z. Joint limits, frame rotation and spring rest angle use radians. Viscous damping uses N·m·s/rad and stiffness uses N·m/rad. The spring rest angle is independent of the zero-angle frame alignment.

An optional `planar_dynamics` block sets XY gravity in m/s² and can add named joint dampers. A damper with coefficients `a` indexed by joint name and coefficient `d` contributes generalized torque `-d a (a · qvel)`. Each damper is a passive MuJoCo fixed tendon, so its contribution is evaluated at every integration stage. Joint damping and these coupled dampers add. Set `solver.planar_substeps` to divide the control period `solver.dt` into equal physics steps. Existing planar graphs default to vertical gravity and one physics step per control tick.

Spatial models use a `spatial_multibody` interior with a `mujoco_asset` binding. Its `ContentObjectRef` identifies the exact UTF-8 MJCF bytes from the caller's artifact provider. A spatial asset is self-contained. File references, external XML includes and plugins are rejected. Its anatomy, inertials and physics options come from the asset. The graph adds attachment sites in named bodies' local frames and tendon paths through named asset or overlay sites. Editing a site changes the overlay. Editing anatomy requires an asset with a new content identity.

The compiled descriptor owns the native model, exact authored identity, asset identity where applicable, workspace dimension, selected effector, and the native name-to-index bindings. Joint slices distinguish configuration coordinates from velocities. Actuator slices identify native activation state. Muscle path bindings retain the authored law, parameters and external state width for the consuming geometry and muscle layers.

The structural signature includes named topology and state and law layout. Numeric changes can preserve that signature while changing exact identity. Group candidates by structural signature, then check that their chosen provider's static data can share one compiled program. Resume against the same authored model uses exact identity.

## Muscle excitation

Set the `ActuationInput` source to `muscle_excitation_vector` to drive the paths in lexicographic name order. The default law is the bounded, zero-pennation rigid Hill law with one activation state. A path can instead name `feedbax.ReluMuscle` or a custom type registered with the same component registry used to compile the graph. A registered component either implements `MuscleLaw` or returns a pure law from `as_muscle_law()`. The law takes the current state, excitation, length and lengthening velocity, and returns its state derivative and shortening tension. The declared state size must match the law's fixed vector width.

Set law parameters in `law_parameters` on a planar `MusclePath` or in `parameters` on a spatial overlay path. An optional `law_revision` selects the component registry revision and its migrations. Exact continuation records the resolved law identities. Fixed paths use the tendon length from engine geometry. The older flat `reference_length` field applies to explicit constant moment arms, which retain their original rigid-tendon behavior. A graph using constant arms must use them for every path and cannot combine them with nested law parameters.

The continuous solver evaluates paths, muscle state derivatives and tension together at each stage. A native MuJoCo step accepts raw authored actuator controls. Geometry-driven excitation requires the continuous solver.

## Execution

`compile_graph` builds either interior through `Mechanics` and the compiled descriptor. Pass the spatial asset's artifact provider explicitly. The `actuator_control` input has a declared vector size that must match the compiled actuator count. The canonical planar muscle template uses its authored constant moment arms and the existing rigid-tendon muscle law. Authored muscle paths expose geometry for coupling.

Each authored muscle path is an ordered spatial tendon through fixed sites. The path reports length in metres, positive lengthening velocity in metres per second, and the Jacobian of length with respect to generalized displacements. The Jacobian has one column per generalized velocity coordinate, including for quaternion models. A generated stateless tendon motor takes positive shortening tension in newtons with unit gain and gear −1. The engine then applies generalized force `-J.T @ tension` and the corresponding power `-tension @ lengthening_velocity`. The public `actuator_control` port continues to address the other native actuators in their original order.

Coupled continuous stages evaluate position and velocity before computing tension and pass those same engine values to actuation and acceleration. A fixed segment whose sites coincide on one body is rejected at compilation. Geometry evaluation rejects a pose where a segment becomes coincident or nonfinite. Derivatives apply at regular fixed-path poses of a fixed model.


For planar wrapping, `wrap_geometries` defines named spheres or Z-aligned cylinders with a frame, XY offset and radius in metres. These geometries do not add contact or inertia. The cylinder's visual half-height equals its radius. Tendon routing uses MuJoCo's infinite-cylinder surface. A spatial overlay references named sphere or cylinder geometry from the asset. In either model, the path's optional `wraps` list names each wrapped segment by its zero-based `after_site` index, and names the `geom` and `side_site`. A planar side site is a named `PointMarker`. Geometry, site and segment references persist with the model.

The reverse-mode capability requires zero compiled constraint rows. For wrapped models every side site must be outside its primitive and fixed in the same body. Use derivatives in a regular pose domain, where the path endpoints are clear of the primitive, the selected side defines a stable route, and the pose stays away from grazing or route transitions. In the planar acceptance geometry, endpoint radii are 0.07 and 0.12 m, wrap radius is 0.062 m, and the tested active-wrap angle range is 1.65–2.55 rad. The near-grazing checks qualify forward routing and force transmission. Inside side sites remain available for forward execution, with reverse-mode capability disabled. `muscle_path_wrap_counts` reads the current engine route engagement per authored path.

| Solver | Execution |
| --- | --- |
| `mujoco` | MJX-JAX steps the authored MuJoCo integrator at the model timestep. Controls and COM wrenches are held through one native substep. |
| `euler`, `tsit5` | Diffrax advances continuous MJX forward dynamics at a fixed substep. Loads and actuator derivatives are evaluated at each stage. |

The control period must be an integer multiple of the model timestep. The continuous route admits zero compiled constraint rows and stage-local actuator dynamics without activation limits. History, plugins and plasticity are outside this continuation profile. Qualification uses the exact stock provider described in [MuJoCo numerical baseline](mujoco_numerical_baseline.md).

Spatial position and velocity are in world XYZ. The orientation matrix maps local coordinates to world axes. Spatial body wrenches supply three coordinates each for force, moment and application point. Planar wrenches supply XY force and point and a one-element Z moment. A body-frame application point is relative to the body origin. The backend transports the free moment and lever-arm contribution to the body COM.

Ball and free joint configurations use MuJoCo's scalar-first quaternion convention with body-local angular velocity. The continuous path normalizes quaternions at each dynamics stage, computes their tangent derivatives, and retracts accepted states to the unit sphere. Singular or nonfinite quaternions fail. The native path uses the integrator's own quaternion update.

Reset uses the authored initial configuration. Same-runtime continuation binds the physical model, runtime, solver and periods, and retains generalized coordinates, velocities, time, native actuator and external muscle state, held loads and solver buffers. An explicit transfer between execution modes restores common physical state and initializes the destination's solver and provider continuation buffers.
