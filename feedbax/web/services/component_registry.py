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
                description='Generic recurrent network block.',
                param_schema=[
                    ParamSchema(name='hidden_size', type='int', default=100, min=1, required=True),
                    ParamSchema(name='input_size', type='int', default=6, min=1, required=True),
                    ParamSchema(name='output_size', type='int', default=2, min=1, required=True),
                    ParamSchema(
                        name='hidden_type',
                        type='enum',
                        options=['GRUCell', 'LSTMCell', 'Linear'],
                        default='GRUCell',
                        required=False,
                    ),
                    ParamSchema(
                        name='out_nonlinearity',
                        type='enum',
                        options=['tanh', 'relu', 'identity'],
                        default='tanh',
                        required=False,
                    ),
                ],
                input_ports=['target', 'feedback'],
                output_ports=['output', 'hidden'],
                icon='CircuitBoard',
                port_types=PortTypeSpec(
                    inputs={
                        'target': PortType(dtype='vector'),
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
                name='MLP',
                category='Neural Networks',
                description='Feedforward multilayer perceptron.',
                param_schema=[
                    ParamSchema(name='input_size', type='int', default=6, min=1, required=True),
                    ParamSchema(name='output_size', type='int', default=2, min=1, required=True),
                    ParamSchema(name='hidden_sizes', type='array', default=[64, 64], required=True),
                    ParamSchema(
                        name='activation',
                        type='enum',
                        options=['relu', 'tanh', 'gelu', 'identity'],
                        default='relu',
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
                name='GRU',
                category='Neural Networks',
                description='Gated recurrent unit network.',
                param_schema=[
                    ParamSchema(name='input_size', type='int', default=6, min=1, required=True),
                    ParamSchema(name='hidden_size', type='int', default=64, min=1, required=True),
                    ParamSchema(name='output_size', type='int', default=2, min=1, required=True),
                    ParamSchema(name='num_layers', type='int', default=1, min=1, required=False),
                ],
                input_ports=['input'],
                output_ports=['output', 'hidden'],
                icon='BrainCircuit',
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='vector')},
                    outputs={
                        'output': PortType(dtype='vector'),
                        'hidden': PortType(dtype='vector'),
                    },
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='LSTM',
                category='Neural Networks',
                description='Long short-term memory network.',
                param_schema=[
                    ParamSchema(name='input_size', type='int', default=6, min=1, required=True),
                    ParamSchema(name='hidden_size', type='int', default=64, min=1, required=True),
                    ParamSchema(name='output_size', type='int', default=2, min=1, required=True),
                    ParamSchema(name='num_layers', type='int', default=1, min=1, required=False),
                ],
                input_ports=['input'],
                output_ports=['output', 'hidden'],
                icon='BrainCog',
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='vector')},
                    outputs={
                        'output': PortType(dtype='vector'),
                        'hidden': PortType(dtype='vector'),
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
                        options=['TwoLinkArm', 'PointMass', 'MuscledArm'],
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
                        'effector': PortType(dtype='vector'),
                        'state': PortType(dtype='state'),
                    },
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='TwoLinkArm',
                category='Mechanics',
                description='Two-link arm plant.',
                param_schema=[
                    ParamSchema(name='dt', type='float', default=0.01, min=0.001, required=True),
                ],
                input_ports=['force'],
                output_ports=['effector', 'state'],
                icon='Move',
                port_types=PortTypeSpec(
                    inputs={'force': PortType(dtype='vector')},
                    outputs={
                        'effector': PortType(dtype='vector'),
                        'state': PortType(dtype='state'),
                    },
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='PointMass',
                category='Mechanics',
                description='Point mass plant.',
                param_schema=[
                    ParamSchema(name='dt', type='float', default=0.01, min=0.001, required=True),
                ],
                input_ports=['force'],
                output_ports=['effector', 'state'],
                icon='Circle',
                port_types=PortTypeSpec(
                    inputs={'force': PortType(dtype='vector')},
                    outputs={
                        'effector': PortType(dtype='vector'),
                        'state': PortType(dtype='state'),
                    },
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='Spring',
                category='Mechanics',
                description='Simple spring element.',
                param_schema=[
                    ParamSchema(name='stiffness', type='float', default=1.0, required=True),
                    ParamSchema(name='rest_length', type='float', default=0.0, required=False),
                ],
                input_ports=['input'],
                output_ports=['output'],
                icon='MoveHorizontal',
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='vector')},
                    outputs={'output': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='Damper',
                category='Mechanics',
                description='Viscous damping element.',
                param_schema=[
                    ParamSchema(name='damping', type='float', default=0.5, required=True),
                ],
                input_ports=['input'],
                output_ports=['output'],
                icon='SlidersVertical',
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='vector')},
                    outputs={'output': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='Channel',
                category='Channels',
                description='Delay and noise for a signal.',
                param_schema=[
                    ParamSchema(name='delay', type='int', default=5, min=0, required=True),
                    ParamSchema(name='noise_std', type='float', default=0.01, min=0, required=True),
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
                name='Constant',
                category='Signals',
                description='Emit a constant signal value.',
                param_schema=[
                    ParamSchema(name='value', type='float', default=0.5, required=True),
                ],
                input_ports=[],
                output_ports=['output'],
                icon='Minus',
                port_types=PortTypeSpec(
                    inputs={},
                    outputs={'output': PortType(dtype='scalar')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='Gain',
                category='Math',
                description='Scale a signal by a gain factor.',
                param_schema=[
                    ParamSchema(name='gain', type='float', default=1.0, required=True),
                ],
                input_ports=['input'],
                output_ports=['output'],
                icon='Sigma',
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='vector')},
                    outputs={'output': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='Sum',
                category='Math',
                description='Sum two signals.',
                param_schema=[],
                input_ports=['a', 'b'],
                output_ports=['output'],
                icon='Plus',
                port_types=PortTypeSpec(
                    inputs={'a': PortType(dtype='vector'), 'b': PortType(dtype='vector')},
                    outputs={'output': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='Multiply',
                category='Math',
                description='Multiply two signals.',
                param_schema=[],
                input_ports=['a', 'b'],
                output_ports=['output'],
                icon='X',
                port_types=PortTypeSpec(
                    inputs={'a': PortType(dtype='vector'), 'b': PortType(dtype='vector')},
                    outputs={'output': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='Saturation',
                category='Math',
                description='Clamp a signal between min and max.',
                param_schema=[
                    ParamSchema(name='min', type='float', default=-1.0, required=True),
                    ParamSchema(name='max', type='float', default=1.0, required=True),
                ],
                input_ports=['input'],
                output_ports=['output'],
                icon='SlidersHorizontal',
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='vector')},
                    outputs={'output': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='DelayLine',
                category='Channels',
                description='Delay a signal by N steps.',
                param_schema=[
                    ParamSchema(name='delay', type='int', default=1, min=0, required=True),
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
                name='NoiseInjector',
                category='Channels',
                description='Add noise to a signal.',
                param_schema=[
                    ParamSchema(name='noise_std', type='float', default=0.01, min=0, required=True),
                ],
                input_ports=['input'],
                output_ports=['output'],
                icon='Sparkles',
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='vector')},
                    outputs={'output': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='Ramp',
                category='Signals',
                description='Linearly increasing signal.',
                param_schema=[
                    ParamSchema(name='slope', type='float', default=1.0, required=True),
                    ParamSchema(name='intercept', type='float', default=0.0, required=True),
                ],
                input_ports=[],
                output_ports=['output'],
                icon='TrendingUp',
                port_types=PortTypeSpec(inputs={}, outputs={'output': PortType(dtype='scalar')}),
            )
        )
        self.register(
            ComponentMeta(
                name='Sine',
                category='Signals',
                description='Sinusoidal oscillator.',
                param_schema=[
                    ParamSchema(name='amplitude', type='float', default=1.0, required=True),
                    ParamSchema(name='frequency', type='float', default=1.0, required=True),
                    ParamSchema(name='phase', type='float', default=0.0, required=True),
                ],
                input_ports=[],
                output_ports=['output'],
                icon='AudioWaveform',
                port_types=PortTypeSpec(inputs={}, outputs={'output': PortType(dtype='scalar')}),
            )
        )
        self.register(
            ComponentMeta(
                name='Pulse',
                category='Signals',
                description='Square pulse generator.',
                param_schema=[
                    ParamSchema(name='amplitude', type='float', default=1.0, required=True),
                    ParamSchema(name='period', type='float', default=1.0, required=True),
                    ParamSchema(
                        name='duty_cycle',
                        type='float',
                        default=0.5,
                        min=0.0,
                        max=1.0,
                        required=True,
                    ),
                ],
                input_ports=[],
                output_ports=['output'],
                icon='HeartPulse',
                port_types=PortTypeSpec(inputs={}, outputs={'output': PortType(dtype='scalar')}),
            )
        )
        self.register(
            ComponentMeta(
                name='Noise',
                category='Signals',
                description='Random noise source.',
                param_schema=[
                    ParamSchema(name='std', type='float', default=0.1, min=0.0, required=True),
                ],
                input_ports=[],
                output_ports=['output'],
                icon='Sparkles',
                port_types=PortTypeSpec(inputs={}, outputs={'output': PortType(dtype='scalar')}),
            )
        )
        self.register(
            ComponentMeta(
                name='CurlField',
                category='Interventions',
                description='Curl force field on mechanics.',
                param_schema=[
                    ParamSchema(name='scale', type='float', default=1.0, required=True),
                    ParamSchema(name='amplitude', type='float', default=1.0, required=True),
                    ParamSchema(name='active', type='bool', default=True, required=False),
                ],
                input_ports=['velocity'],
                output_ports=['force'],
                icon='Wind',
                port_types=PortTypeSpec(
                    inputs={'velocity': PortType(dtype='vector')},
                    outputs={'force': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='ForceField',
                category='Interventions',
                description='General force field on mechanics.',
                param_schema=[
                    ParamSchema(name='scale', type='float', default=1.0, required=True),
                    ParamSchema(name='field', type='array', default=[0.0, 0.0], required=True),
                    ParamSchema(name='active', type='bool', default=True, required=False),
                ],
                input_ports=['velocity'],
                output_ports=['force'],
                icon='Magnet',
                port_types=PortTypeSpec(
                    inputs={'velocity': PortType(dtype='vector')},
                    outputs={'force': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='FixedField',
                category='Interventions',
                description='Fixed force field on mechanics.',
                param_schema=[
                    ParamSchema(name='scale', type='float', default=1.0, required=True),
                    ParamSchema(name='amplitude', type='float', default=1.0, required=True),
                    ParamSchema(name='field', type='array', default=[0.0, 0.0], required=True),
                    ParamSchema(name='active', type='bool', default=True, required=False),
                ],
                input_ports=[],
                output_ports=['force'],
                icon='Flag',
                port_types=PortTypeSpec(
                    inputs={},
                    outputs={'force': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='Clamp',
                category='Interventions',
                description='Clamp a signal or state to a range.',
                param_schema=[
                    ParamSchema(name='min', type='float', default=-1.0, required=True),
                    ParamSchema(name='max', type='float', default=1.0, required=True),
                    ParamSchema(name='active', type='bool', default=True, required=False),
                ],
                input_ports=['input'],
                output_ports=['output'],
                icon='Shield',
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='vector')},
                    outputs={'output': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='Perturbation',
                category='Interventions',
                description='Add a perturbation to a signal.',
                param_schema=[
                    ParamSchema(name='scale', type='float', default=1.0, required=True),
                    ParamSchema(name='active', type='bool', default=True, required=False),
                ],
                input_ports=['input'],
                output_ports=['output'],
                icon='Zap',
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='vector')},
                    outputs={'output': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='AddNoise',
                category='Interventions',
                description='Add noise to a state.',
                param_schema=[
                    ParamSchema(name='scale', type='float', default=1.0, required=True),
                    ParamSchema(name='active', type='bool', default=True, required=False),
                ],
                input_ports=['input'],
                output_ports=['output'],
                icon='Sparkles',
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='state')},
                    outputs={'output': PortType(dtype='state')},
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
                    ParamSchema(name='active', type='bool', default=True, required=False),
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
        self.register(
            ComponentMeta(
                name='NetworkConstantInput',
                category='Interventions',
                description='Add constant input to network units.',
                param_schema=[
                    ParamSchema(name='scale', type='float', default=1.0, required=True),
                    ParamSchema(name='active', type='bool', default=True, required=False),
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
                description='Add constant input to a state.',
                param_schema=[
                    ParamSchema(name='scale', type='float', default=1.0, required=True),
                    ParamSchema(name='active', type='bool', default=True, required=False),
                ],
                input_ports=[],
                output_ports=['output'],
                icon='Minus',
                port_types=PortTypeSpec(
                    inputs={},
                    outputs={'output': PortType(dtype='vector')},
                ),
            )
        )
        self.register(
            ComponentMeta(
                name='Copy',
                category='Interventions',
                description='Copy a state sub-tree.',
                param_schema=[
                    ParamSchema(name='scale', type='float', default=1.0, required=True),
                    ParamSchema(name='active', type='bool', default=True, required=False),
                ],
                input_ports=['input'],
                output_ports=['output'],
                icon='Copy',
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='state')},
                    outputs={'output': PortType(dtype='state')},
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
                        type='array',
                        default=[[-1.0, -1.0], [1.0, 1.0]],
                        required=True,
                    ),
                    ParamSchema(name='eval_n_directions', type='int', default=7, min=1, required=False),
                    ParamSchema(name='eval_reach_length', type='float', default=0.5, required=False),
                    ParamSchema(name='eval_grid_n', type='int', default=1, min=1, required=False),
                ],
                input_ports=[],
                output_ports=[],
                icon='Target',
            )
        )
        self.register(
            ComponentMeta(
                name='ReachingTask',
                category='Tasks',
                description='Simple reaching task.',
                param_schema=[
                    ParamSchema(name='n_steps', type='int', default=200, min=1, required=True),
                    ParamSchema(
                        name='workspace',
                        type='array',
                        default=[[-1.0, -1.0], [1.0, 1.0]],
                        required=True,
                    ),
                ],
                input_ports=[],
                output_ports=[],
                icon='Target',
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
                        type='array',
                        default=[[-1.0, -1.0], [1.0, 1.0]],
                        required=True,
                    ),
                    ParamSchema(name='delay_steps', type='int', default=40, min=0, required=False),
                ],
                input_ports=[],
                output_ports=[],
                icon='Timer',
            )
        )
        self.register(
            ComponentMeta(
                name='TrackingTask',
                category='Tasks',
                description='Tracking task with a moving target.',
                param_schema=[
                    ParamSchema(name='n_steps', type='int', default=200, min=1, required=True),
                    ParamSchema(name='trajectory', type='array', default=[], required=False),
                ],
                input_ports=[],
                output_ports=[],
                icon='Radar',
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
                        type='array',
                        default=[[-1.0, -1.0], [1.0, 1.0]],
                        required=True,
                    ),
                ],
                input_ports=[],
                output_ports=[],
                icon='Anchor',
            )
        )
        self.register(
            ComponentMeta(
                name='HoldTask',
                category='Tasks',
                description='Hold a target position.',
                param_schema=[
                    ParamSchema(name='n_steps', type='int', default=200, min=1, required=True),
                    ParamSchema(
                        name='workspace',
                        type='array',
                        default=[[-1.0, -1.0], [1.0, 1.0]],
                        required=False,
                    ),
                ],
                input_ports=[],
                output_ports=[],
                icon='Pause',
            )
        )
        self.register(
            ComponentMeta(
                name='Probe',
                category='Utilities',
                description='Tap a signal/state for loss and analysis.',
                param_schema=[
                    ParamSchema(name='label', type='str', default='probe', required=False),
                ],
                input_ports=['input'],
                output_ports=['output'],
                icon='Eye',
                port_types=PortTypeSpec(
                    inputs={'input': PortType(dtype='any')},
                    outputs={'output': PortType(dtype='any')},
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
        )
