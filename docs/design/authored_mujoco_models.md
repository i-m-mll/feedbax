# Authored MuJoCo models

Planar models use the acausal graph's `PlanarLink`, `RevoluteJoint`, `WorldFrame`, and `Anchor` nodes. The supported assembly is a single serial chain with one hinge per link and a world-rooted pivot. Connections, when present, must agree with the joint and anchor frame names. `planar_chain_graph` creates this representation from a body preset.

A link's proximal origin is local zero and its distal frame is at `(length, 0)`. Length, frame offsets and transverse COM offset are in metres, and mass is in kilograms. The `com` field measures the longitudinal COM as a fraction of link length. `inertia` is the moment about local Z through the COM, in kg·m². The native body uses explicit principal moments `(Izz/2, Izz/2, Izz)`. Those transverse moments complete the planar rigid-body representation for MuJoCo. The graph's hinge degrees of freedom restrict motion to the plane.

A revolute joint aligns its parent and child frames after applying the authored parent-frame offset and the zero-angle frame rotation. Its axis is local +Z. Joint limits, frame rotation and spring rest angle use radians. Viscous damping uses N·m·s/rad and stiffness uses N·m/rad. The spring rest angle is independent of the zero-angle frame alignment.

Spatial models use a `spatial_multibody` interior with a `mujoco_asset` binding. Its `ContentObjectRef` identifies the exact UTF-8 MJCF bytes from the caller's artifact provider. A spatial asset is self-contained. File references, external XML includes and plugins are rejected. Its anatomy, inertials and physics options come from the asset. The graph adds attachment sites in named bodies' local frames and tendon paths through named asset or overlay sites. Editing a site changes the overlay. Editing anatomy requires an asset with a new content identity.

The compiled descriptor owns the native model, exact authored identity, asset identity where applicable, workspace dimension, selected effector, and the native name-to-index bindings. Joint slices distinguish configuration coordinates from velocities. Actuator slices identify native activation state. Muscle path bindings retain the authored law, parameters and external state width for the consuming geometry and muscle layers.

The structural signature includes named topology and state and law layout. Numeric changes can preserve that signature while changing exact identity. Group candidates by structural signature, then check that their chosen provider's static data can share one compiled program. Resume against the same authored model uses exact identity.
