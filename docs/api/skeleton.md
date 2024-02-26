# Skeleton models

As subclasses ultimately of `AbstractDynamicalSystem`, skeleton models in Feedbax define the continuous skeletal dynamics with a `vector_field` method, which returns the time derivatives of the skeletal state.

---

::: feedbax.mechanics.skeleton.PointMass
    options:
        members: [
            'vector_field',
            'init',
            'effector',
            'forward_kinematics',
            'inverse_kinematics',
            'update_state_given_effector_force',
        ]

::: feedbax.mechanics.skeleton.TwoLinkState

::: feedbax.mechanics.skeleton.TwoLink
    options:
        members: [
            'vector_field',
            'init',
            'effector',
            'forward_kinematics',
            'inverse_kinematics',
            'update_state_given_effector_force',
            'bounds',
            'effector_jac',
            'workspace_test',
        ]


## Abstract base classes

::: feedbax.mechanics.skeleton.AbstractSkeleton
    options:
        members: [
            'vector_field',
            'init',
            'effector',
            'forward_kinematics',
            'inverse_kinematics',
            'update_state_given_effector_force',            
        ]
