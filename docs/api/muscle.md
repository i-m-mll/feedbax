# Muscle models

!!! Info  ""
    Currently, only variants of the Virtual Muscle Model (Brown et al., 1999) are implemented.

::: feedbax.mechanics.muscle.MuscleState

## Constructors for specific variants of the Virtual Muscle Model

::: feedbax.mechanics.muscle.brown_1999_virtualmuscle
    options:
        separate_signature: true

::: feedbax.mechanics.muscle.todorov_li_2004_virtualmuscle
    options:
        separate_signature: true

::: feedbax.mechanics.muscle.lillicrap_scott_2013_virtualmuscle
    options:
        separate_signature: true

## Virtual Muscle Model

::: feedbax.mechanics.muscle.VirtualMuscle

## Muscle input signal filters

::: feedbax.mechanics.muscle.ActivationFilter
    options:
        members: ['__call__', 'vector_field']

## Abstract base classes

::: feedbax.mechanics.muscle.AbstractMuscle

