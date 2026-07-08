from __future__ import annotations

from typing import Protocol

from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.graph import ComponentSpec, GraphSpec, GraphUIState, NodeUIState, WireSpec

from .meta import ComponentMeta


class _Registry(Protocol):
    def register(self, meta: ComponentMeta) -> None: ...


def register_cde_templates(registry: _Registry) -> None:
    _cde_port_types = PortTypeSpec(
        inputs={
            'obs': PortType(dtype='vector'),
            'obs_prev': PortType(dtype='vector'),
            'h_prev': PortType(dtype='vector'),
        },
        outputs={
            'h_new': PortType(dtype='vector'),
            'action': PortType(dtype='vector'),
        },
    )
    _cde_input_ports = ['obs', 'obs_prev', 'h_prev']
    _cde_output_ports = ['h_new', 'action']
    _cde_input_bindings = {
        'obs': ('obs_in', 'input'),
        'obs_prev': ('obs_prev_in', 'input'),
        'h_prev': ('h_prev_in', 'input'),
    }
    _cde_output_bindings = {
        'h_new': ('sum_h', 'output'),
        'action': ('sigmoid', 'output'),
    }
    _obs_dim = 1
    _hidden_dim = 1
    _action_dim = 1

    # ------------------------------------------------------------------ #
    # Layout constants — mirror the cdeTemplates.ts column/row grid
    # ------------------------------------------------------------------ #
    COL0, COL1, COL2, COL3, COL4, COL5, COL6 = 20, 160, 320, 480, 640, 800, 960
    ROW_TOP, ROW_MID, ROW_BOT = 40, 160, 280
    COL_GATE = 340

    # ------------------------------------------------------------------ #
    # Helpers: make ComponentSpec / WireSpec / NodeUIState
    # ------------------------------------------------------------------ #
    def _node(type_: str, input_ports: list[str], output_ports: list[str], **params) -> ComponentSpec:
        if type_ == 'Input':
            input_ports = ['input']
            params.setdefault('output_port', output_ports[0])
        elif type_ == 'MLP':
            hidden_size = params.pop('hidden_size', 128)
            params.setdefault('input_size', _hidden_dim)
            params.setdefault('output_size', _hidden_dim * _obs_dim)
            params.setdefault('hidden_sizes', [hidden_size])
            params['activation'] = 'relu'
            params.setdefault('final_activation', 'tanh')
        elif type_ == 'Reshape':
            params.setdefault('shape', [_hidden_dim, _obs_dim])
        elif type_ == 'Linear':
            params.setdefault('input_size', _hidden_dim)
            params.setdefault('output_size', _action_dim)
        elif type_ == 'GRU':
            params['input_size'] = _obs_dim
            params['hidden_size'] = _hidden_dim
        return ComponentSpec(type=type_, params=params, input_ports=input_ports, output_ports=output_ports)

    def _wire(src_node: str, src_port: str, tgt_node: str, tgt_port: str) -> WireSpec:
        return WireSpec(source_node=src_node, source_port=src_port, target_node=tgt_node, target_port=tgt_port)

    def _pos(x: int, y: int) -> NodeUIState:
        return NodeUIState(position={'x': float(x), 'y': float(y)}, collapsed=False, selected=False)

    # ------------------------------------------------------------------ #
    # Standard CDE
    # ------------------------------------------------------------------ #
    standard_nodes = {
        'obs_in':      _node('Input',    [],              ['obs']),
        'obs_prev_in': _node('Input',    [],              ['obs_prev']),
        'h_prev_in':   _node('Input',    [],              ['h_prev']),
        'subtract':    _node('Subtract', ['a', 'b'],      ['out']),
        'vf':          _node('MLP',      ['input'],       ['output'], hidden_size=128, activation='tanh'),
        'reshape':     _node('Reshape',  ['input'],       ['output']),
        'matmul':      _node('MatMul',   ['a', 'b'],      ['out']),
        'sum_h':       _node('Sum',      ['a', 'b'],      ['output']),
        'linear':      _node('Linear',   ['input'],       ['output']),
        'sigmoid':     _node('Sigmoid',  ['input'],       ['output']),
    }
    standard_wires = [
        _wire('obs_in',      'obs',      'subtract',  'a'),
        _wire('obs_prev_in', 'obs_prev', 'subtract',  'b'),
        _wire('h_prev_in',   'h_prev',   'vf',        'input'),
        _wire('vf',          'output',   'reshape',   'input'),
        _wire('reshape',     'output',   'matmul',    'a'),
        _wire('subtract',    'out',      'matmul',    'b'),
        _wire('h_prev_in',   'h_prev',   'sum_h',     'a'),
        _wire('matmul',      'out',      'sum_h',     'b'),
        _wire('sum_h',       'output',   'linear',    'input'),
        _wire('linear',      'output',   'sigmoid',   'input'),
    ]
    standard_ui = GraphUIState(node_states={
        'obs_in':      _pos(COL0, ROW_TOP),
        'obs_prev_in': _pos(COL0, ROW_MID),
        'h_prev_in':   _pos(COL0, ROW_BOT),
        'subtract':    _pos(COL1, ROW_TOP),
        'vf':          _pos(COL2, ROW_BOT),
        'reshape':     _pos(COL3, ROW_BOT),
        'matmul':      _pos(COL3, ROW_TOP),
        'sum_h':       _pos(COL4, ROW_MID),
        'linear':      _pos(COL5, ROW_MID),
        'sigmoid':     _pos(COL6, ROW_MID),
    })
    registry.register(ComponentMeta(
        name='CDE Standard',
        category='CDE Controllers',
        description='Basic CDE step: VectorField × dX → h_new, Linear → Sigmoid → action.',
        param_schema=[],
        input_ports=_cde_input_ports,
        output_ports=_cde_output_ports,
        icon='BrainCircuit',
        is_composite=True,
        port_types=_cde_port_types,
        template_graph=GraphSpec(
            nodes=standard_nodes,
            wires=standard_wires,
            input_ports=_cde_input_ports,
            output_ports=_cde_output_ports,
            input_bindings=_cde_input_bindings,
            output_bindings=_cde_output_bindings,
        ),
        template_ui_state=standard_ui,
        template_id='feedbax.templates.cde_standard',
        template_kind='executable',
    ))

    # ------------------------------------------------------------------ #
    # CDE + Decay
    # ------------------------------------------------------------------ #
    decay_nodes = {
        'obs_in':      _node('Input',    [],                 ['obs']),
        'obs_prev_in': _node('Input',    [],                 ['obs_prev']),
        'h_prev_in':   _node('Input',    [],                 ['h_prev']),
        'subtract':    _node('Subtract', ['a', 'b'],         ['out']),
        'vf':          _node('MLP',      ['input'],          ['output'], hidden_size=128, activation='tanh'),
        'reshape':     _node('Reshape',  ['input'],          ['output']),
        'matmul':      _node('MatMul',   ['a', 'b'],         ['out']),
        'decay':       _node('Scale',    ['input'],          ['output'], scale=-0.1),
        'sum_h':       _node('Sum',      ['a', 'b', 'c'],    ['output']),
        'linear':      _node('Linear',   ['input'],          ['output']),
        'sigmoid':     _node('Sigmoid',  ['input'],          ['output']),
    }
    decay_wires = [
        _wire('obs_in',      'obs',      'subtract',  'a'),
        _wire('obs_prev_in', 'obs_prev', 'subtract',  'b'),
        _wire('h_prev_in',   'h_prev',   'vf',        'input'),
        _wire('vf',          'output',   'reshape',   'input'),
        _wire('reshape',     'output',   'matmul',    'a'),
        _wire('subtract',    'out',      'matmul',    'b'),
        _wire('h_prev_in',   'h_prev',   'decay',     'input'),
        _wire('h_prev_in',   'h_prev',   'sum_h',     'a'),
        _wire('matmul',      'out',      'sum_h',     'b'),
        _wire('decay',       'output',   'sum_h',     'c'),
        _wire('sum_h',       'output',   'linear',    'input'),
        _wire('linear',      'output',   'sigmoid',   'input'),
    ]
    decay_ui = GraphUIState(node_states={
        'obs_in':      _pos(COL0, ROW_TOP),
        'obs_prev_in': _pos(COL0, ROW_MID),
        'h_prev_in':   _pos(COL0, ROW_BOT),
        'subtract':    _pos(COL1, ROW_TOP),
        'vf':          _pos(COL2, ROW_BOT),
        'reshape':     _pos(COL3, ROW_BOT),
        'matmul':      _pos(COL3, ROW_TOP),
        'decay':       _pos(COL2, ROW_MID),
        'sum_h':       _pos(COL4, ROW_MID),
        'linear':      _pos(COL5, ROW_MID),
        'sigmoid':     _pos(COL6, ROW_MID),
    })
    registry.register(ComponentMeta(
        name='CDE + Decay',
        category='CDE Controllers',
        description='CDE with exponential decay: h_new = h_prev + M×dX − decay×h_prev.',
        param_schema=[],
        input_ports=_cde_input_ports,
        output_ports=_cde_output_ports,
        icon='TrendingUp',
        is_composite=True,
        port_types=_cde_port_types,
        template_graph=GraphSpec(
            nodes=decay_nodes,
            wires=decay_wires,
            input_ports=_cde_input_ports,
            output_ports=_cde_output_ports,
            input_bindings=_cde_input_bindings,
            output_bindings=_cde_output_bindings,
        ),
        template_ui_state=decay_ui,
        template_id='feedbax.templates.cde_decay',
        template_kind='executable',
    ))

    # ------------------------------------------------------------------ #
    # CDE + Anti-NF
    # ------------------------------------------------------------------ #
    antinf_nodes = {
        'obs_in':      _node('Input',    [],                 ['obs']),
        'obs_prev_in': _node('Input',    [],                 ['obs_prev']),
        'h_prev_in':   _node('Input',    [],                 ['h_prev']),
        'subtract':    _node('Subtract', ['a', 'b'],         ['out']),
        'negate_h':    _node('Scale',    ['input'],          ['output'], scale=-1.0),
        'vf':          _node('MLP',      ['input'],          ['output'], hidden_size=128, activation='tanh'),
        'reshape':     _node('Reshape',  ['input'],          ['output']),
        'matmul':      _node('MatMul',   ['a', 'b'],         ['out']),
        'gru_gate':    _node('GRU',      ['input', 'hidden'], ['output', 'hidden'], hidden_size=64),
        'alpha':       _node('Scale',    ['input'],          ['output'], scale=0.1),
        'sum_h':       _node('Sum',      ['a', 'b', 'c'],    ['output']),
        'linear':      _node('Linear',   ['input'],          ['output']),
        'sigmoid':     _node('Sigmoid',  ['input'],          ['output']),
    }
    antinf_wires = [
        _wire('obs_in',      'obs',      'subtract',  'a'),
        _wire('obs_prev_in', 'obs_prev', 'subtract',  'b'),
        _wire('h_prev_in',   'h_prev',   'vf',        'input'),
        _wire('vf',          'output',   'reshape',   'input'),
        _wire('reshape',     'output',   'matmul',    'a'),
        _wire('subtract',    'out',      'matmul',    'b'),
        _wire('h_prev_in',   'h_prev',   'negate_h',  'input'),
        _wire('obs_in',      'obs',      'gru_gate',  'input'),
        _wire('negate_h',    'output',   'gru_gate',  'hidden'),
        _wire('gru_gate',    'output',   'alpha',     'input'),
        _wire('h_prev_in',   'h_prev',   'sum_h',     'a'),
        _wire('matmul',      'out',      'sum_h',     'b'),
        _wire('alpha',       'output',   'sum_h',     'c'),
        _wire('sum_h',       'output',   'linear',    'input'),
        _wire('linear',      'output',   'sigmoid',   'input'),
    ]
    antinf_ui = GraphUIState(node_states={
        'obs_in':      _pos(COL0, ROW_TOP),
        'obs_prev_in': _pos(COL0, ROW_MID),
        'h_prev_in':   _pos(COL0, ROW_BOT),
        'subtract':    _pos(COL1, ROW_TOP),
        'negate_h':    _pos(COL1, ROW_BOT),
        'vf':          _pos(COL2, ROW_MID),
        'reshape':     _pos(COL3, ROW_TOP),
        'matmul':      _pos(COL3, ROW_TOP + 80),
        'gru_gate':    _pos(COL_GATE, ROW_BOT),
        'alpha':       _pos(COL4, ROW_BOT),
        'sum_h':       _pos(COL4 + 80, ROW_MID),
        'linear':      _pos(COL5, ROW_MID),
        'sigmoid':     _pos(COL6, ROW_MID),
    })
    registry.register(ComponentMeta(
        name='CDE + Anti-NF',
        category='CDE Controllers',
        description='CDE with Anti-NF gate: GRU(obs, −h) × α provides gated feedback correction.',
        param_schema=[],
        input_ports=_cde_input_ports,
        output_ports=_cde_output_ports,
        icon='BrainCog',
        is_composite=True,
        port_types=_cde_port_types,
        template_graph=GraphSpec(
            nodes=antinf_nodes,
            wires=antinf_wires,
            input_ports=_cde_input_ports,
            output_ports=_cde_output_ports,
            input_bindings=_cde_input_bindings,
            output_bindings=_cde_output_bindings,
        ),
        template_ui_state=antinf_ui,
        template_id='feedbax.templates.cde_anti_nf',
        template_kind='executable',
    ))

    # ------------------------------------------------------------------ #
    # CDE Hybrid v9b
    # ------------------------------------------------------------------ #
    hybrid_nodes = {
        'obs_in':      _node('Input',    [],                       ['obs']),
        'obs_prev_in': _node('Input',    [],                       ['obs_prev']),
        'h_prev_in':   _node('Input',    [],                       ['h_prev']),
        'subtract':    _node('Subtract', ['a', 'b'],               ['out']),
        'vf':          _node('MLP',      ['input'],                ['output'], hidden_size=128, activation='tanh'),
        'reshape':     _node('Reshape',  ['input'],                ['output']),
        'matmul':      _node('MatMul',   ['a', 'b'],               ['out']),
        'decay':       _node('Scale',    ['input'],                ['output'], scale=-0.1),
        'negate_h':    _node('Scale',    ['input'],                ['output'], scale=-1.0),
        'gru_gate':    _node('GRU',      ['input', 'hidden'],      ['output', 'hidden'], hidden_size=64),
        'alpha':       _node('Scale',    ['input'],                ['output'], scale=0.1),
        'sum_h':       _node('Sum',      ['a', 'b', 'c', 'd'],     ['output']),
        'linear':      _node('Linear',   ['input'],                ['output']),
        'sigmoid':     _node('Sigmoid',  ['input'],                ['output']),
    }
    hybrid_wires = [
        _wire('obs_in',      'obs',      'subtract',  'a'),
        _wire('obs_prev_in', 'obs_prev', 'subtract',  'b'),
        _wire('h_prev_in',   'h_prev',   'vf',        'input'),
        _wire('vf',          'output',   'reshape',   'input'),
        _wire('reshape',     'output',   'matmul',    'a'),
        _wire('subtract',    'out',      'matmul',    'b'),
        _wire('h_prev_in',   'h_prev',   'decay',     'input'),
        _wire('h_prev_in',   'h_prev',   'negate_h',  'input'),
        _wire('obs_in',      'obs',      'gru_gate',  'input'),
        _wire('negate_h',    'output',   'gru_gate',  'hidden'),
        _wire('gru_gate',    'output',   'alpha',     'input'),
        _wire('h_prev_in',   'h_prev',   'sum_h',     'a'),
        _wire('matmul',      'out',      'sum_h',     'b'),
        _wire('decay',       'output',   'sum_h',     'c'),
        _wire('alpha',       'output',   'sum_h',     'd'),
        _wire('sum_h',       'output',   'linear',    'input'),
        _wire('linear',      'output',   'sigmoid',   'input'),
    ]
    hybrid_ui = GraphUIState(node_states={
        'obs_in':      _pos(COL0, ROW_TOP),
        'obs_prev_in': _pos(COL0, ROW_MID),
        'h_prev_in':   _pos(COL0, ROW_BOT),
        'subtract':    _pos(COL1, ROW_TOP),
        'vf':          _pos(COL1, ROW_BOT),
        'reshape':     _pos(COL2, ROW_BOT),
        'matmul':      _pos(COL2, ROW_TOP),
        'decay':       _pos(COL2, ROW_MID),
        'negate_h':    _pos(COL2, ROW_BOT + 80),
        'gru_gate':    _pos(COL3, ROW_BOT + 80),
        'alpha':       _pos(COL4, ROW_BOT + 80),
        'sum_h':       _pos(COL4, ROW_MID),
        'linear':      _pos(COL5, ROW_MID),
        'sigmoid':     _pos(COL6, ROW_MID),
    })
    registry.register(ComponentMeta(
        name='CDE Hybrid v9b',
        category='CDE Controllers',
        description='Production architecture: fixed-decay floor + Anti-NF gate (v9b hybrid).',
        param_schema=[],
        input_ports=_cde_input_ports,
        output_ports=_cde_output_ports,
        icon='Sparkles',
        is_composite=True,
        port_types=_cde_port_types,
        template_graph=GraphSpec(
            nodes=hybrid_nodes,
            wires=hybrid_wires,
            input_ports=_cde_input_ports,
            output_ports=_cde_output_ports,
            input_bindings=_cde_input_bindings,
            output_bindings=_cde_output_bindings,
        ),
        template_ui_state=hybrid_ui,
        template_id='feedbax.templates.cde_hybrid_v9b',
        template_kind='executable',
    ))
