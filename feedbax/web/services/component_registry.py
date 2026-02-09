from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import importlib.util

from feedbax.web.models.component import ComponentDefinition, PortTypeSpec, PortType
from feedbax.web.models.graph import ParamSchema


@dataclass
class ComponentMeta:
    name: str
    category: str
    description: str
    param_schema: List[ParamSchema]
    input_ports: List[str]
    output_ports: List[str]
    icon: str = 'box'
    port_types: Optional[PortTypeSpec] = None
    is_composite: bool = False

    @property
    def default_params(self) -> Dict[str, object]:
        return {schema.name: schema.default for schema in self.param_schema}


class ComponentRegistry:
    def __init__(self) -> None:
        self._components: Dict[str, ComponentMeta] = {}
        self._register_builtins()
        self.load_user_components(Path.home() / '.feedbax' / 'components')

    def _register_builtins(self) -> None:
        self.register(
            ComponentMeta(
                name='Network',
                category='Neural Networks',
                description='Composite recurrent network template.',
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
                        options=['tanh', 'relu', 'identity'],
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
        self.register(
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
        self.register(
            ComponentMeta(
                name='PenzaiSubgraph',
                category='Structure',
                description='Penzai model wrapper for feedbax Graphs.',
                param_schema=[
                    ParamSchema(
                        name='builder_name',
                        type='enum',
                        options=[],  # Populated dynamically from registry
                        default='',
                        required=True,
                    ),
                    ParamSchema(name='input_port', type='string', default='input', required=False),
                    ParamSchema(name='output_port', type='string', default='output', required=False),
                ],
                input_ports=['input'],
                output_ports=['output'],
                icon='Hexagon',
                is_composite=True,
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='any')},
                    outputs={'output': PortType(dtype='any')},
                ),
            )
        )
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
            ComponentMeta(
                name='Linear',
                category='Neural Networks',
                description='Linear layer.',
                param_schema=[
                    ParamSchema(name='input_size', type='int', default=1, min=1, required=True),
                    ParamSchema(name='output_size', type='int', default=1, min=1, required=True),
                    ParamSchema(name='use_bias', type='bool', default=True, required=False),
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
            ComponentMeta(
                name='Mechanics',
                category='Mechanics',
                description='Biomechanical plant simulator for the limb.',
                param_schema=[
                    ParamSchema(
                        name='plant_type',
                        type='enum',
                        options=['TwoLinkArm', 'PointMass'],
                        default='TwoLinkArm',
                        required=True,
                    ),
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
        self.register(
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
        self.register(
            ComponentMeta(
                name='Channel',
                category='Channels',
                description='Delay and noise for a signal.',
                param_schema=[
                    ParamSchema(name='delay', type='int', default=5, min=0, required=True),
                    ParamSchema(name='noise_std', type='float', default=0.01, min=0, required=False),
                    ParamSchema(name='add_noise', type='bool', default=True, required=False),
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
            ComponentMeta(
                name='RigidTendonHillMuscleThelen',
                category='Muscles',
                description='Thelen 2003 rigid tendon Hill muscle.',
                param_schema=[
                    ParamSchema(
                        name='max_isometric_force', type='float',
                        default=500.0, min=0.0, required=True,
                    ),
                    ParamSchema(
                        name='optimal_muscle_length', type='float',
                        default=0.1, min=0.001, required=True,
                    ),
                    ParamSchema(
                        name='tendon_slack_length', type='float',
                        default=0.1, min=0.0, required=True,
                    ),
                    ParamSchema(
                        name='vmax_factor', type='float',
                        default=10.0, min=1.0, required=False,
                    ),
                    ParamSchema(
                        name='dt', type='float',
                        default=0.01, min=0.001, required=True,
                    ),
                ],
                input_ports=[
                    'excitation', 'musculotendon_length',
                    'musculotendon_velocity',
                ],
                output_ports=[
                    'force', 'activation', 'fiber_length', 'fiber_velocity',
                ],
                icon='Zap',
                port_types=PortTypeSpec(
                    inputs={
                        'excitation': PortType(dtype='scalar'),
                        'musculotendon_length': PortType(dtype='scalar'),
                        'musculotendon_velocity': PortType(dtype='scalar'),
                    },
                    outputs={
                        'force': PortType(dtype='scalar'),
                        'activation': PortType(dtype='scalar'),
                        'fiber_length': PortType(dtype='scalar'),
                        'fiber_velocity': PortType(dtype='scalar'),
                    },
                ),
            )
        )
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
            ComponentMeta(
                name='DelayedReaches',
                category='Tasks',
                description='Reaches with a delay period before movement.',
                param_schema=[
                    ParamSchema(name='n_steps', type='int', default=240, min=1, required=True),
                    ParamSchema(
                        name='workspace',
                        type='bounds2d',
                        default=[[-1.0, -1.0], [1.0, 1.0]],
                        required=True,
                    ),
                    ParamSchema(name='delay_steps', type='int', default=40, min=0, required=False),
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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
        self.register(
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

    def register(self, meta: ComponentMeta) -> None:
        self._components[meta.name] = meta

    def get(self, name: str) -> Optional[ComponentMeta]:
        return self._components.get(name)

    def list_all(self) -> List[ComponentDefinition]:
        return [self._to_definition(meta) for meta in self._components.values()]

    def list_by_category(self) -> Dict[str, List[ComponentDefinition]]:
        by_category: Dict[str, List[ComponentDefinition]] = {}
        for meta in self._components.values():
            by_category.setdefault(meta.category, []).append(self._to_definition(meta))
        return by_category

    def load_user_components(self, path: Path) -> None:
        if not path.exists():
            return

        for py_file in path.glob('*.py'):
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name in dir(module):
                obj = getattr(module, name)
                meta = getattr(obj, '_feedbax_component_meta', None)
                if not isinstance(meta, dict):
                    continue
                self.register(
                    ComponentMeta(
                        name=meta.get('name', name),
                        category=meta.get('category', 'Custom'),
                        description=meta.get('description', ''),
                        param_schema=[ParamSchema(**schema) for schema in meta.get('param_schema', [])],
                        input_ports=list(meta.get('input_ports', [])),
                        output_ports=list(meta.get('output_ports', [])),
                        icon=meta.get('icon', 'box'),
                        is_composite=bool(meta.get('is_composite', False)),
                        port_types=(
                            PortTypeSpec(
                                inputs={
                                    key: PortType(**value)
                                    for key, value in meta.get('port_types', {}).get('inputs', {}).items()
                                },
                                outputs={
                                    key: PortType(**value)
                                    for key, value in meta.get('port_types', {}).get('outputs', {}).items()
                                },
                            )
                            if meta.get('port_types')
                            else None
                        ),
                    )
                )

    def _to_definition(self, meta: ComponentMeta) -> ComponentDefinition:
        return ComponentDefinition(
            name=meta.name,
            category=meta.category,
            description=meta.description,
            param_schema=meta.param_schema,
            input_ports=meta.input_ports,
            output_ports=meta.output_ports,
            icon=meta.icon,
            default_params=meta.default_params,
            port_types=meta.port_types,
            is_composite=meta.is_composite,
        )
