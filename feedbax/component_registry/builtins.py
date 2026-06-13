from __future__ import annotations

from typing import Protocol

from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.graph import ParamSchema

from .cde_templates import register_cde_templates
from .meta import ComponentMeta
from .templates import register_builtin_graph_templates


class _Registry(Protocol):
    def register(self, meta: ComponentMeta) -> None: ...


def register_builtin_components(registry: _Registry) -> None:
    registry.register(
        ComponentMeta(
            name='Network',
            category='Neural Networks',
            description='Recurrent network. Enter to inspect and modify the internal architecture.',
            param_schema=[
                ParamSchema(name='input_size', type='int', default=6, min=1, required=True),
                ParamSchema(name='hidden_size', type='int', default=100, min=1, required=True),
                ParamSchema(name='out_size', type='int', default=2, min=1, required=False),
                ParamSchema(
                    name='hidden_type',
                    type='enum',
                    options=['GRUCell', 'LSTMCell', 'Linear'],
                    default='GRUCell',
                    required=False,
                ),
                ParamSchema(
                    name='hidden_nonlinearity',
                    type='enum',
                    options=['tanh', 'relu', 'identity'],
                    default='tanh',
                    required=False,
                ),
                ParamSchema(
                    name='out_nonlinearity',
                    type='enum',
                    options=['tanh', 'relu', 'sigmoid', 'identity'],
                    default='tanh',
                    required=False,
                ),
                ParamSchema(
                    name='hidden_noise_std',
                    type='float',
                    default=0.0,
                    min=0.0,
                    required=False,
                ),
                ParamSchema(
                    name='encoding_size',
                    type='int',
                    default=0,
                    min=0,
                    required=False,
                ),
            ],
            input_ports=['input', 'feedback'],
            output_ports=['output', 'hidden'],
            icon='CircuitBoard',
            is_composite=True,
            port_types=PortTypeSpec(
                inputs={
                    'input': PortType(dtype='vector'),
                    'feedback': PortType(dtype='vector'),
                },
                outputs={
                    'output': PortType(dtype='vector'),
                    'hidden': PortType(dtype='vector'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Subgraph',
            category='Structure',
            description='Nested graph container.',
            param_schema=[],
            input_ports=[],
            output_ports=[],
            icon='Layers',
            is_composite=True,
        )
    )
    register_builtin_graph_templates(registry)
    registry.register(
        ComponentMeta(
            name='PenzaiAdapter',
            category='Structure',
            description='Penzai neural network adapter (leaf). Wraps a trained Penzai model for inference.',
            param_schema=[
                ParamSchema(
                    name='builder_name',
                    type='enum',
                    options=[],  # Populated dynamically from registry
                    default='',
                    required=True,
                ),
                ParamSchema(name='input_port', type='str', default='input', required=False),
                ParamSchema(name='output_port', type='str', default='output', required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Hexagon',
            is_composite=False,
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Gain',
            category='Math',
            description='Multiply input by constant.',
            param_schema=[
                ParamSchema(name='gain', type='float', default=1.0, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='SlidersHorizontal',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Sum',
            category='Math',
            description='Add two inputs.',
            param_schema=[],
            input_ports=['a', 'b'],
            output_ports=['output'],
            icon='Sigma',
            port_types=PortTypeSpec(
                inputs={'a': PortType(dtype='any'), 'b': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Multiply',
            category='Math',
            description='Element-wise product.',
            param_schema=[],
            input_ports=['a', 'b'],
            output_ports=['output'],
            icon='X',
            port_types=PortTypeSpec(
                inputs={'a': PortType(dtype='any'), 'b': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Constant',
            category='Sources',
            description='Constant value output.',
            param_schema=[
                ParamSchema(name='value', type='float', default=0.0, required=True),
            ],
            input_ports=[],
            output_ports=['output'],
            icon='Circle',
            port_types=PortTypeSpec(
                inputs={},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Ramp',
            category='Sources',
            description='Linear ramp over time.',
            param_schema=[
                ParamSchema(name='slope', type='float', default=1.0, required=True),
                ParamSchema(name='intercept', type='float', default=0.0, required=True),
                ParamSchema(name='dt', type='float', default=0.01, required=True),
            ],
            input_ports=[],
            output_ports=['output'],
            icon='TrendingUp',
            port_types=PortTypeSpec(
                inputs={},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Sine',
            category='Sources',
            description='Sinusoidal signal.',
            param_schema=[
                ParamSchema(name='amplitude', type='float', default=1.0, required=True),
                ParamSchema(name='frequency', type='float', default=1.0, required=True),
                ParamSchema(name='phase', type='float', default=0.0, required=False),
                ParamSchema(name='offset', type='float', default=0.0, required=False),
                ParamSchema(name='dt', type='float', default=0.01, required=True),
            ],
            input_ports=[],
            output_ports=['output'],
            icon='AudioWaveform',
            port_types=PortTypeSpec(
                inputs={},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Pulse',
            category='Sources',
            description='Pulse/square wave.',
            param_schema=[
                ParamSchema(name='amplitude', type='float', default=1.0, required=True),
                ParamSchema(name='period', type='float', default=1.0, required=True),
                ParamSchema(name='duty_cycle', type='float', default=0.5, required=True),
                ParamSchema(name='offset', type='float', default=0.0, required=False),
                ParamSchema(name='dt', type='float', default=0.01, required=True),
            ],
            input_ports=[],
            output_ports=['output'],
            icon='Activity',
            port_types=PortTypeSpec(
                inputs={},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Noise',
            category='Signal Processing',
            description='Random noise source.',
            param_schema=[
                ParamSchema(name='mean', type='float', default=0.0, required=False),
                ParamSchema(name='std', type='float', default=1.0, required=True),
                ParamSchema(name='shape', type='array', default=[1], required=False),
            ],
            input_ports=[],
            output_ports=['output'],
            icon='Sparkles',
            port_types=PortTypeSpec(
                inputs={},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Saturation',
            category='Signal Processing',
            description='Clamp to min/max range.',
            param_schema=[
                ParamSchema(name='min_val', type='float', default=-1.0, required=True),
                ParamSchema(name='max_val', type='float', default=1.0, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='SlidersHorizontal',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='DelayLine',
            category='Signal Processing',
            description='Discrete delay buffer.',
            param_schema=[
                ParamSchema(name='delay', type='int', default=1, min=0, required=True),
                ParamSchema(name='init_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Clock',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='MLP',
            category='Neural Networks',
            description='Multi-layer perceptron.',
            param_schema=[
                ParamSchema(name='input_size', type='int', default=4, min=1, required=True),
                ParamSchema(name='output_size', type='int', default=2, min=1, required=True),
                ParamSchema(name='hidden_sizes', type='array', default=[64], required=False),
                ParamSchema(
                    name='activation',
                    type='enum',
                    options=['relu', 'tanh', 'identity'],
                    default='relu',
                    required=False,
                ),
                ParamSchema(
                    name='final_activation',
                    type='enum',
                    options=['identity', 'tanh', 'relu'],
                    default='identity',
                    required=False,
                ),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Brain',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Linear',
            category='Neural Networks',
            description='Linear layer.',
            param_schema=[
                ParamSchema(name='input_size', type='int', default=1, min=1, required=True),
                ParamSchema(name='output_size', type='int', default=1, min=1, required=True),
                ParamSchema(name='use_bias', type='bool', default=True, required=False),
                ParamSchema(
                    name='activation',
                    type='enum',
                    options=['identity', 'tanh', 'relu', 'sigmoid'],
                    default='identity',
                    required=False,
                ),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Minus',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='GRU',
            category='Neural Networks',
            description='GRU cell.',
            param_schema=[
                ParamSchema(name='input_size', type='int', default=4, min=1, required=True),
                ParamSchema(name='hidden_size', type='int', default=4, min=1, required=True),
            ],
            input_ports=['input', 'hidden'],
            output_ports=['output', 'hidden'],
            icon='BrainCircuit',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector'), 'hidden': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector'), 'hidden': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='LSTM',
            category='Neural Networks',
            description='LSTM cell.',
            param_schema=[
                ParamSchema(name='input_size', type='int', default=4, min=1, required=True),
                ParamSchema(name='hidden_size', type='int', default=4, min=1, required=True),
            ],
            input_ports=['input', 'hidden', 'cell'],
            output_ports=['output', 'hidden', 'cell'],
            icon='BrainCircuit',
            port_types=PortTypeSpec(
                inputs={
                    'input': PortType(dtype='vector'),
                    'hidden': PortType(dtype='vector'),
                    'cell': PortType(dtype='vector'),
                },
                outputs={
                    'output': PortType(dtype='vector'),
                    'hidden': PortType(dtype='vector'),
                    'cell': PortType(dtype='vector'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='GRUOracle',
            category='Neural Networks',
            description='GRU-based oracle/policy network that maps observations to muscle excitations.',
            param_schema=[
                ParamSchema(
                    name='hidden_size', type='int',
                    default=128, min=1, required=True,
                ),
                ParamSchema(
                    name='n_layers', type='int',
                    default=1, min=1, required=False,
                ),
                ParamSchema(
                    name='out_size', type='int',
                    default=6, min=1, required=True,
                ),
            ],
            input_ports=['input'],
            output_ports=['output', 'hidden'],
            icon='BrainCircuit',
            port_types=PortTypeSpec(
                inputs={
                    'input': PortType(dtype='vector'),
                },
                outputs={
                    'output': PortType(dtype='vector'),
                    'hidden': PortType(dtype='vector'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Spring',
            category='Mechanics',
            description='Linear spring.',
            param_schema=[
                ParamSchema(name='stiffness', type='float', default=1.0, required=True),
            ],
            input_ports=['displacement'],
            output_ports=['force'],
            icon='Move',
            port_types=PortTypeSpec(
                inputs={'displacement': PortType(dtype='vector')},
                outputs={'force': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Damper',
            category='Mechanics',
            description='Viscous damper.',
            param_schema=[
                ParamSchema(name='damping', type='float', default=1.0, required=True),
            ],
            input_ports=['velocity'],
            output_ports=['force'],
            icon='Move',
            port_types=PortTypeSpec(
                inputs={'velocity': PortType(dtype='vector')},
                outputs={'force': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='TwoLinkArm',
            category='Mechanics',
            description='Two-link arm plant with direct force input.',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.01, min=0.001, required=True),
            ],
            input_ports=['force'],
            output_ports=['effector', 'state'],
            icon='Activity',
            port_types=PortTypeSpec(
                inputs={'force': PortType(dtype='vector')},
                outputs={
                    'effector': PortType(dtype='state'),
                    'state': PortType(dtype='state'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='PointMass',
            category='Mechanics',
            description='Point-mass plant with direct force input.',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.01, min=0.001, required=True),
            ],
            input_ports=['force'],
            output_ports=['effector', 'state'],
            icon='Activity',
            port_types=PortTypeSpec(
                inputs={'force': PortType(dtype='vector')},
                outputs={
                    'effector': PortType(dtype='state'),
                    'state': PortType(dtype='state'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='LinearStateSpace',
            category='Mechanics',
            description='Discrete linear state-space mechanics.',
            param_schema=[
                ParamSchema(
                    name='A',
                    type='array',
                    default=[
                        [1.0, 0.0, 0.01, 0.0],
                        [0.0, 1.0, 0.0, 0.01],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    required=True,
                ),
                ParamSchema(
                    name='B',
                    type='array',
                    default=[
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.01, 0.0],
                        [0.0, 0.01],
                    ],
                    required=True,
                ),
                ParamSchema(name='B_w', type='array', default=None, required=False),
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=False),
                ParamSchema(
                    name='initial_state',
                    type='array',
                    default=[0.0, 0.0, 0.0, 0.0],
                    required=False,
                ),
                ParamSchema(name='pos_slice', type='array', default=[0, 2], required=False),
                ParamSchema(name='vel_slice', type='array', default=[2, 4], required=False),
            ],
            input_ports=['force', 'epsilon'],
            output_ports=['effector', 'state'],
            icon='Grid3x3',
            port_types=PortTypeSpec(
                inputs={
                    'force': PortType(dtype='vector'),
                    'epsilon': PortType(dtype='vector'),
                },
                outputs={
                    'effector': PortType(dtype='state'),
                    'state': PortType(dtype='vector'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='MomentArmProjection',
            category='Mechanics',
            description='Projects muscle forces to joint torques via moment arm matrix (R^T @ forces). Also computes musculotendon lengths and velocities from joint kinematics.',
            param_schema=[
                ParamSchema(name='n_muscles', type='int', default=6, min=1, required=True),
                ParamSchema(name='n_joints', type='int', default=2, min=1, required=True),
            ],
            input_ports=['forces', 'angles', 'angular_velocities'],
            output_ports=['torques', 'musculotendon_lengths', 'musculotendon_velocities'],
            icon='Ruler',
            port_types=PortTypeSpec(
                inputs={
                    'forces': PortType(dtype='vector'),
                    'angles': PortType(dtype='vector'),
                    'angular_velocities': PortType(dtype='vector'),
                },
                outputs={
                    'torques': PortType(dtype='vector'),
                    'musculotendon_lengths': PortType(dtype='vector'),
                    'musculotendon_velocities': PortType(dtype='vector'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='RadialForceProjection',
            category='Mechanics',
            description='Projects radially-arranged muscle forces to a 2D net force vector. Muscles are arranged in evenly-spaced antagonist pairs.',
            param_schema=[
                ParamSchema(name='n_muscles', type='int', default=8, min=2, required=True),
            ],
            input_ports=['forces'],
            output_ports=['force_2d'],
            icon='Compass',
            port_types=PortTypeSpec(
                inputs={'forces': PortType(dtype='vector')},
                outputs={'force_2d': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='AcausalSystem',
            category='Mechanics',
            description='Assembled acausal mechanical system (mass-spring-damper etc.).',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.001, min=0.0001, required=True),
                ParamSchema(
                    name='domain',
                    type='enum',
                    options=['translational', 'rotational'],
                    default='translational',
                    required=False,
                ),
            ],
            input_ports=['input'],
            output_ports=['state'],
            icon='Cog',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'state': PortType(dtype='state')},
            ),
            is_composite=True,
        )
    )
    registry.register(
        ComponentMeta(
            name='Channel',
            category='Channels',
            description='Delay and noise for a signal.',
            param_schema=[
                ParamSchema(name='delay', type='int', default=5, min=0, required=True),
                ParamSchema(name='noise_std', type='float', default=0.01, min=0, required=False),
                ParamSchema(name='add_noise', type='bool', default=True, required=False),
                ParamSchema(name='input_shape', type='array', default=[1], required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Signal',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='FirstOrderFilter',
            category='Channels',
            description='First-order low-pass filter.',
            param_schema=[
                ParamSchema(name='tau_rise', type='float', default=0.05, min=0.0, required=True),
                ParamSchema(name='tau_decay', type='float', default=0.05, min=0.0, required=True),
                ParamSchema(name='dt', type='float', default=0.001, min=0.0, required=True),
                ParamSchema(name='init_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Filter',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='CurlField',
            category='Interventions',
            description='Velocity-dependent curl field.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(name='amplitude', type='float', default=1.0, required=True),
                ParamSchema(name='active', type='bool', default=False, required=False),
            ],
            input_ports=['effector', 'force'],
            output_ports=['force'],
            icon='Wind',
            port_types=PortTypeSpec(
                inputs={
                    'effector': PortType(dtype='state'),
                    'force': PortType(dtype='vector'),
                },
                outputs={'force': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='FixedField',
            category='Interventions',
            description='Fixed force field.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(name='amplitude', type='float', default=1.0, required=True),
                ParamSchema(name='field', type='array', default=[0.0, 0.0], required=True),
                ParamSchema(name='active', type='bool', default=False, required=False),
            ],
            input_ports=['force'],
            output_ports=['force'],
            icon='Flag',
            port_types=PortTypeSpec(
                inputs={'force': PortType(dtype='vector')},
                outputs={'force': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='AddNoise',
            category='Interventions',
            description='Add noise to a signal.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(name='active', type='bool', default=False, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Sparkles',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='NetworkClamp',
            category='Interventions',
            description='Clamp network unit activity.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(name='active', type='bool', default=False, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Pin',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    # --- Muscles ---
    registry.register(
        ComponentMeta(
            name='ReluMuscle',
            category='Muscles',
            description='Simple muscle: force = activation * F_max.',
            param_schema=[
                ParamSchema(
                    name='max_isometric_force', type='float',
                    default=500.0, min=0.0, required=True,
                ),
                ParamSchema(
                    name='tau_activation', type='float',
                    default=0.015, min=0.001, required=False,
                ),
                ParamSchema(
                    name='tau_deactivation', type='float',
                    default=0.05, min=0.001, required=False,
                ),
                ParamSchema(
                    name='min_activation', type='float',
                    default=0.0, min=0.0, required=False,
                ),
                ParamSchema(
                    name='dt', type='float',
                    default=0.01, min=0.001, required=True,
                ),
                ParamSchema(
                    name='initial_activation', type='float',
                    default=0.0, min=0.0, required=False,
                ),
            ],
            input_ports=['excitation'],
            output_ports=['force', 'activation'],
            icon='Zap',
            port_types=PortTypeSpec(
                inputs={'excitation': PortType(dtype='scalar')},
                outputs={
                    'force': PortType(dtype='scalar'),
                    'activation': PortType(dtype='scalar'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='RigidTendonHillMuscleThelen',
            category='Muscles',
            description='Hill-type muscle with rigid tendon assumption. Vectorized for multiple muscles.',
            param_schema=[
                ParamSchema(name='n_muscles', type='int', default=6, min=1, required=True),
                ParamSchema(name='dt', type='float', default=0.01, min=0.001, required=True),
                ParamSchema(name='tau_activation', type='float', default=0.015, min=0.001, required=False),
                ParamSchema(name='tau_deactivation', type='float', default=0.05, min=0.001, required=False),
                ParamSchema(name='max_isometric_force', type='float', default=500.0, min=0.0, required=False),
                ParamSchema(name='optimal_muscle_length', type='float', default=0.1, min=0.001, required=False),
                ParamSchema(name='tendon_slack_length', type='float', default=0.2, min=0.001, required=False),
                ParamSchema(name='initial_activation', type='float', default=0.001, min=0.0, required=False),
            ],
            input_ports=['excitation', 'musculotendon_length', 'musculotendon_velocity'],
            output_ports=['force', 'activation', 'fiber_length', 'fiber_velocity'],
            icon='Dumbbell',
            port_types=PortTypeSpec(
                inputs={
                    'excitation': PortType(dtype='vector'),
                    'musculotendon_length': PortType(dtype='vector'),
                    'musculotendon_velocity': PortType(dtype='vector'),
                },
                outputs={
                    'force': PortType(dtype='vector'),
                    'activation': PortType(dtype='vector'),
                    'fiber_length': PortType(dtype='vector'),
                    'fiber_velocity': PortType(dtype='vector'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Arm6MuscleRigidTendon',
            category='Mechanics',
            description='6-muscle arm with Thelen rigid tendon.',
            param_schema=[
                ParamSchema(
                    name='dt', type='float',
                    default=0.01, min=0.001, required=True,
                ),
                ParamSchema(
                    name='max_isometric_force', type='float',
                    default=500.0, min=0.0, required=False,
                ),
                ParamSchema(
                    name='optimal_muscle_length', type='float',
                    default=0.1, min=0.001, required=False,
                ),
                ParamSchema(
                    name='tendon_slack_length', type='float',
                    default=0.1, min=0.0, required=False,
                ),
            ],
            input_ports=['excitation', 'angles', 'angular_velocities'],
            output_ports=['torques', 'forces', 'activations'],
            icon='Activity',
            is_composite=True,
            port_types=PortTypeSpec(
                inputs={
                    'excitation': PortType(dtype='vector'),
                    'angles': PortType(dtype='vector'),
                    'angular_velocities': PortType(dtype='vector'),
                },
                outputs={
                    'torques': PortType(dtype='vector'),
                    'forces': PortType(dtype='vector'),
                    'activations': PortType(dtype='vector'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='PointMass8MuscleRelu',
            category='Mechanics',
            description='8-muscle point mass with ReLU actuators.',
            param_schema=[
                ParamSchema(
                    name='n_pairs', type='int',
                    default=4, min=1, required=False,
                ),
                ParamSchema(
                    name='max_isometric_force', type='float',
                    default=500.0, min=0.0, required=False,
                ),
                ParamSchema(
                    name='dt', type='float',
                    default=0.01, min=0.001, required=True,
                ),
            ],
            input_ports=['excitation'],
            output_ports=['force_2d', 'forces', 'activations'],
            icon='Activity',
            is_composite=True,
            port_types=PortTypeSpec(
                inputs={'excitation': PortType(dtype='vector')},
                outputs={
                    'force_2d': PortType(dtype='vector'),
                    'forces': PortType(dtype='vector'),
                    'activations': PortType(dtype='vector'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='AnalyticalMusculoskeletalPlant',
            category='Mechanics',
            description=(
                'Two-link arm musculoskeletal plant with pure JAX Lagrangian '
                'dynamics and Hill-type rigid-tendon muscles. Fully '
                'differentiable; no MuJoCo dependency. ODE state: 2 joint '
                'angles + 2 angular velocities + 6 muscle activations.'
            ),
            param_schema=[
                ParamSchema(
                    name='dt', type='float',
                    default=0.01, min=0.0001, required=True,
                ),
                ParamSchema(
                    name='n_steps', type='int',
                    default=1, min=1, required=False,
                ),
                ParamSchema(
                    name='tau_act', type='float',
                    default=0.01, min=0.001, required=False,
                ),
                ParamSchema(
                    name='tau_deact', type='float',
                    default=0.04, min=0.001, required=False,
                ),
                ParamSchema(
                    name='clip_states', type='bool',
                    default=True, required=False,
                ),
            ],
            input_ports=['excitation'],
            output_ports=['effector', 'state'],
            icon='Activity',
            is_composite=False,
            port_types=PortTypeSpec(
                inputs={
                    'excitation': PortType(dtype='vector'),
                },
                outputs={
                    'effector': PortType(dtype='vector'),
                    'state': PortType(dtype='state'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='NetworkConstantInput',
            category='Interventions',
            description='Add constant input to network units.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(name='active', type='bool', default=False, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Asterisk',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='ConstantInput',
            category='Interventions',
            description='Add a constant input to a signal.',
            param_schema=[
                ParamSchema(name='scale', type='float', default=1.0, required=True),
                ParamSchema(name='active', type='bool', default=False, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Minus',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='SimpleReaches',
            category='Tasks',
            description='Random reach endpoints in a workspace.',
            param_schema=[
                ParamSchema(name='n_steps', type='int', default=200, min=1, required=True),
                ParamSchema(
                    name='workspace',
                    type='bounds2d',
                    default=[[-1.0, -1.0], [1.0, 1.0]],
                    required=True,
                ),
                ParamSchema(name='eval_n_directions', type='int', default=7, min=1, required=False),
                ParamSchema(name='eval_reach_length', type='float', default=0.5, required=False),
                ParamSchema(name='eval_grid_n', type='int', default=1, min=1, required=False),
            ],
            input_ports=[],
            output_ports=['inputs', 'targets', 'inits', 'intervene'],
            icon='Target',
            port_types=PortTypeSpec(
                inputs={},
                outputs={
                    'inputs': PortType(dtype='any'),
                    'targets': PortType(dtype='state'),
                    'inits': PortType(dtype='state'),
                    'intervene': PortType(dtype='any'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='DelayedReaches',
            category='Tasks',
            description='Reaches with a delay period before movement.',
            param_schema=[
                ParamSchema(name='n_steps', type='int', default=140, min=1, required=True),
                ParamSchema(
                    name='n_control_stages',
                    type='int',
                    default=None,
                    min=1,
                    required=False,
                ),
                ParamSchema(
                    name='workspace',
                    type='bounds2d',
                    default=[[-1.0, -1.0], [1.0, 1.0]],
                    required=True,
                ),
                ParamSchema(
                    name='preset',
                    type='enum',
                    options=['default', 'delayed_center_out'],
                    default='default',
                    required=False,
                ),
                ParamSchema(
                    name='train_endpoint_mode',
                    type='enum',
                    options=['workspace', 'center_out'],
                    default='workspace',
                    required=False,
                ),
                ParamSchema(
                    name='epoch_len_ranges',
                    type='array',
                    default=[[5, 15], [10, 20]],
                    required=False,
                ),
                ParamSchema(
                    name='epoch_names',
                    type='array',
                    default=['hold', 'target_on', 'movement'],
                    required=False,
                ),
                ParamSchema(name='target_on_epochs', type='array', default=[1, 2], required=False),
                ParamSchema(name='hold_epochs', type='array', default=[0, 1], required=False),
                ParamSchema(name='move_epochs', type='array', default=[2], required=False),
                ParamSchema(
                    name='target_visible_from_start',
                    type='bool',
                    default=False,
                    required=False,
                ),
                ParamSchema(name='go_cue_event_name', type='str', default=None, required=False),
                ParamSchema(
                    name='p_catch_trial',
                    type='float',
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    required=False,
                ),
                ParamSchema(
                    name='catch_metadata_policy',
                    type='enum',
                    options=['none', 'flag'],
                    default='none',
                    required=False,
                ),
                ParamSchema(name='eval_n_directions', type='int', default=7, min=1, required=False),
                ParamSchema(name='eval_reach_length', type='float', default=0.5, required=False),
                ParamSchema(name='eval_grid_n', type='int', default=1, min=1, required=False),
            ],
            input_ports=[],
            output_ports=['inputs', 'targets', 'inits', 'intervene'],
            icon='Timer',
            port_types=PortTypeSpec(
                inputs={},
                outputs={
                    'inputs': PortType(dtype='any'),
                    'targets': PortType(dtype='state'),
                    'inits': PortType(dtype='state'),
                    'intervene': PortType(dtype='any'),
                },
            ),
        )
    )
    # --- Control components ---
    registry.register(
        ComponentMeta(
            name='Integrator',
            category='Control',
            description='Continuous-time integrator (Euler).',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
                ParamSchema(name='initial_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Integral',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Derivative',
            category='Control',
            description='Finite-difference derivative.',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
                ParamSchema(name='initial_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='TrendingUp',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='StateSpace',
            category='Control',
            description='Continuous LTI state-space (Euler).',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Grid3x3',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='TransferFunction',
            category='Control',
            description='Transfer function H(s)=num/den.',
            param_schema=[
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='FunctionSquare',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='PID',
            category='Control',
            description='Continuous PID with anti-windup.',
            param_schema=[
                ParamSchema(name='Kp', type='float', default=1.0, required=True),
                ParamSchema(name='Ki', type='float', default=0.0, required=False),
                ParamSchema(name='Kd', type='float', default=0.0, required=False),
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='integral_limit', type='float', default=1000.0, required=False),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
            ],
            input_ports=['error'],
            output_ports=['output'],
            icon='Gauge',
            port_types=PortTypeSpec(
                inputs={'error': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='PIDDiscrete',
            category='Control',
            description='Discrete PID (velocity form).',
            param_schema=[
                ParamSchema(name='Kp', type='float', default=1.0, required=True),
                ParamSchema(name='Ki', type='float', default=0.0, required=False),
                ParamSchema(name='Kd', type='float', default=0.0, required=False),
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='output_limit', type='float', default=1000.0, required=False),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
            ],
            input_ports=['error'],
            output_ports=['output'],
            icon='Gauge',
            port_types=PortTypeSpec(
                inputs={'error': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    # --- Discrete components ---
    registry.register(
        ComponentMeta(
            name='IntegratorDiscrete',
            category='Discrete',
            description='Discrete-time accumulator.',
            param_schema=[
                ParamSchema(name='dt', type='float', default=1.0, min=0.0, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
                ParamSchema(name='initial_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='PlusSquare',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='UnitDelay',
            category='Discrete',
            description='Unit delay (z^-1).',
            param_schema=[
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
                ParamSchema(name='initial_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Clock',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='ZeroOrderHold',
            category='Discrete',
            description='Sample and hold every N steps.',
            param_schema=[
                ParamSchema(name='hold_steps', type='int', default=1, min=1, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
                ParamSchema(name='initial_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Pause',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    # --- Signal processing components ---
    registry.register(
        ComponentMeta(
            name='Mux',
            category='Signal Processing',
            description='Concatenate inputs into single vector.',
            param_schema=[
                ParamSchema(name='n_inputs', type='int', default=2, min=1, required=True),
            ],
            input_ports=['in_0', 'in_1'],
            output_ports=['output'],
            icon='GitMerge',
            port_types=PortTypeSpec(
                inputs={
                    'in_0': PortType(dtype='vector'),
                    'in_1': PortType(dtype='vector'),
                },
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Ravel',
            category='Signal Processing',
            description='Flatten a PyTree value into a vector.',
            param_schema=[],
            input_ports=['input'],
            output_ports=['output'],
            icon='Layers',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Demux',
            category='Signal Processing',
            description='Split vector into multiple outputs.',
            param_schema=[
                ParamSchema(name='sizes', type='array', default=[1, 1], required=True),
            ],
            input_ports=['input'],
            output_ports=['out_0', 'out_1'],
            icon='GitBranch',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={
                    'out_0': PortType(dtype='vector'),
                    'out_1': PortType(dtype='vector'),
                },
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Switch',
            category='Signal Processing',
            description='Route signal by threshold condition.',
            param_schema=[
                ParamSchema(name='threshold', type='float', default=0.0, required=True),
            ],
            input_ports=['condition', 'true_input', 'false_input'],
            output_ports=['output'],
            icon='GitCompare',
            port_types=PortTypeSpec(
                inputs={
                    'condition': PortType(dtype='scalar'),
                    'true_input': PortType(dtype='any'),
                    'false_input': PortType(dtype='any'),
                },
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='DeadZone',
            category='Signal Processing',
            description='Zero output for small inputs.',
            param_schema=[
                ParamSchema(name='threshold', type='float', default=0.1, min=0.0, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='MinusSquare',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='any')},
                outputs={'output': PortType(dtype='any')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='RateLimiter',
            category='Signal Processing',
            description='Limit rate of change of signal.',
            param_schema=[
                ParamSchema(name='max_rate', type='float', default=1.0, min=0.0, required=True),
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
                ParamSchema(name='initial_value', type='float', default=0.0, required=False),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Gauge',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='HighPassFilter',
            category='Signal Processing',
            description='High-pass filter (input - lowpass).',
            param_schema=[
                ParamSchema(name='tau', type='float', default=0.1, min=0.0, required=True),
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Filter',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='BandPassFilter',
            category='Signal Processing',
            description='Band-pass: high-pass then low-pass.',
            param_schema=[
                ParamSchema(name='tau_low', type='float', default=0.1, min=0.0, required=True),
                ParamSchema(name='tau_high', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='dt', type='float', default=0.01, min=0.0, required=True),
                ParamSchema(name='n_dims', type='int', default=1, min=1, required=True),
            ],
            input_ports=['input'],
            output_ports=['output'],
            icon='Filter',
            port_types=PortTypeSpec(
                inputs={'input': PortType(dtype='vector')},
                outputs={'output': PortType(dtype='vector')},
            ),
        )
    )
    registry.register(
        ComponentMeta(
            name='Stabilization',
            category='Tasks',
            description='Hold position against perturbations.',
            param_schema=[
                ParamSchema(name='n_steps', type='int', default=200, min=1, required=True),
                ParamSchema(
                    name='workspace',
                    type='bounds2d',
                    default=[[-1.0, -1.0], [1.0, 1.0]],
                    required=True,
                ),
            ],
            input_ports=[],
            output_ports=['inputs', 'targets', 'inits', 'intervene'],
            icon='Anchor',
            port_types=PortTypeSpec(
                inputs={},
                outputs={
                    'inputs': PortType(dtype='any'),
                    'targets': PortType(dtype='state'),
                    'inits': PortType(dtype='state'),
                    'intervene': PortType(dtype='any'),
                },
            ),
        )
    )
    # --- CDE Controllers ---
    register_cde_templates(registry)
