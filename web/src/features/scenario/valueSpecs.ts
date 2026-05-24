import type {
  StudioTaskTimelineSignalSpec,
  StudioValueSpec,
  StudioValueSpecMode,
  StudioValueSpecSamplingScope,
} from '@/types/workspace';

export const VALUE_SPEC_MODE_OPTIONS: Array<{
  value: StudioValueSpecMode;
  label: string;
}> = [
  { value: 'constant', label: 'Constant' },
  { value: 'reference', label: 'Reference' },
  { value: 'expression', label: 'Expression' },
  { value: 'function', label: 'Function' },
  { value: 'distribution', label: 'Distribution' },
  { value: 'schedule', label: 'Schedule' },
];

export const VALUE_SPEC_SCOPE_OPTIONS: Array<{
  value: StudioValueSpecSamplingScope;
  label: string;
}> = [
  { value: 'snapshot', label: 'Snapshot' },
  { value: 'run', label: 'Run' },
  { value: 'replicate', label: 'Replicate' },
  { value: 'trial', label: 'Trial' },
  { value: 'epoch', label: 'Epoch' },
  { value: 'timestep', label: 'Timestep' },
  { value: 'sweep', label: 'Sweep' },
];

export const VALUE_SPEC_FUNCTION_TEMPLATES: Array<{
  id: string;
  label: string;
  kind: 'constant' | 'step' | 'pulse' | 'ramp' | 'piecewise' | 'trajectory';
  defaultParameters: Record<string, unknown>;
}> = [
  { id: 'constant', label: 'Constant', kind: 'constant', defaultParameters: { value: 1 } },
  { id: 'step', label: 'Step', kind: 'step', defaultParameters: { before: 0, after: 1 } },
  {
    id: 'pulse',
    label: 'Pulse',
    kind: 'pulse',
    defaultParameters: { baseline: 0, amplitude: 1, width: 1 },
  },
  { id: 'ramp', label: 'Ramp', kind: 'ramp', defaultParameters: { start: 0, end: 1 } },
  {
    id: 'piecewise_linear',
    label: 'Piecewise',
    kind: 'piecewise',
    defaultParameters: { points: [[0, 0], [1, 1]] },
  },
  {
    id: 'delayed_reach_target_position',
    label: 'Target trajectory',
    kind: 'trajectory',
    defaultParameters: {},
  },
  {
    id: 'delayed_reach_movement_target',
    label: 'Movement target',
    kind: 'trajectory',
    defaultParameters: {},
  },
];

export const VALUE_SPEC_DISTRIBUTIONS = [
  'uniform',
  'normal',
  'log_uniform',
  'categorical',
] as const;

export function valueSpecAllowedModes(
  signal: StudioTaskTimelineSignalSpec
): StudioValueSpecMode[] {
  const modes = signal.metadata.value_spec_modes;
  if (Array.isArray(modes) && modes.every((mode) => typeof mode === 'string')) {
    return modes;
  }
  return ['constant', 'function', 'distribution', 'schedule', 'expression'];
}

export function valueSpecAllowedScopes(
  signal: StudioTaskTimelineSignalSpec
): StudioValueSpecSamplingScope[] {
  const scopes = signal.metadata.value_spec_scopes;
  if (Array.isArray(scopes) && scopes.every((scope) => typeof scope === 'string')) {
    return scopes;
  }
  return ['trial', 'epoch', 'timestep'];
}

export function valueSpecChipLabel(valueSpec: StudioValueSpec | null | undefined): string {
  if (!valueSpec) return 'Value';
  if (valueSpec.mode === 'function') {
    const template = VALUE_SPEC_FUNCTION_TEMPLATES.find(
      (candidate) => candidate.id === valueSpec.function_id
    );
    return template?.label ?? valueSpec.function_id ?? 'Function';
  }
  if (valueSpec.mode === 'distribution') {
    const family = String(valueSpec.distribution?.family ?? 'distribution');
    return `${family}${valueSpec.sampling_scope ? `/${valueSpec.sampling_scope}` : ''}`;
  }
  if (valueSpec.mode === 'schedule') {
    const domain = String(valueSpec.schedule?.domain ?? valueSpec.sampling_scope ?? 'time');
    return `Schedule/${domain}`;
  }
  if (valueSpec.mode === 'expression') {
    return valueSpec.expression?.trim() || 'Expression';
  }
  if (valueSpec.mode === 'reference') {
    return valueSpec.reference?.compact ?? valueSpec.reference?.path ?? 'Reference';
  }
  const value = valueSpec.value;
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    if ('active' in record && 'inactive' in record) {
      return `${record.active}/${record.inactive}`;
    }
    if ('min' in record && 'max' in record) {
      return `${record.min}-${record.max}`;
    }
  }
  if (value === null || value === undefined) return 'Constant';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  return 'Constant';
}

export function setValueSpecMode(
  valueSpec: StudioValueSpec,
  mode: StudioValueSpecMode
): StudioValueSpec {
  const base = {
    schema_version: valueSpec.schema_version,
    mode,
    dtype: valueSpec.dtype ?? null,
    shape: valueSpec.shape ?? null,
    units: valueSpec.units ?? null,
    frame: valueSpec.frame ?? null,
    metadata: {
      ...valueSpec.metadata,
      authored_as: 'value_spec_editor',
    },
  };
  if (mode === 'constant') {
    return {
      ...base,
      value: valueSpec.value ?? { active: 1, inactive: 0 },
      sampling_scope: valueSpec.sampling_scope ?? 'trial',
    };
  }
  if (mode === 'function') {
    const template =
      VALUE_SPEC_FUNCTION_TEMPLATES.find((item) => item.id === valueSpec.function_id) ??
      VALUE_SPEC_FUNCTION_TEMPLATES[0];
    return {
      ...base,
      function_id: template.id,
      parameters: valueSpec.parameters ?? template.defaultParameters,
      sampling_scope: valueSpec.sampling_scope ?? 'timestep',
    };
  }
  if (mode === 'distribution') {
    return {
      ...base,
      distribution: valueSpec.distribution ?? {
        family: 'uniform',
        parameters: { min: 0, max: 1 },
      },
      sampling_scope: valueSpec.sampling_scope ?? 'trial',
    };
  }
  if (mode === 'schedule') {
    return {
      ...base,
      schedule: valueSpec.schedule ?? {
        domain: 'epoch',
        function_id: valueSpec.function_id ?? 'step',
      },
      sampling_scope: valueSpec.sampling_scope ?? 'epoch',
    };
  }
  if (mode === 'expression') {
    return {
      ...base,
      expression: valueSpec.expression ?? '',
      sampling_scope: valueSpec.sampling_scope ?? 'run',
    };
  }
  return {
    ...base,
    reference: valueSpec.reference ?? null,
    sampling_scope: valueSpec.sampling_scope ?? 'run',
  };
}

export function setValueSpecScope(
  valueSpec: StudioValueSpec,
  samplingScope: StudioValueSpecSamplingScope
): StudioValueSpec {
  return {
    ...valueSpec,
    sampling_scope: samplingScope,
    metadata: {
      ...valueSpec.metadata,
      authored_as: 'value_spec_editor',
    },
  };
}

export function setValueSpecFunction(
  valueSpec: StudioValueSpec,
  functionId: string
): StudioValueSpec {
  const template =
    VALUE_SPEC_FUNCTION_TEMPLATES.find((candidate) => candidate.id === functionId) ??
    VALUE_SPEC_FUNCTION_TEMPLATES[0];
  return {
    ...setValueSpecMode(valueSpec, 'function'),
    function_id: template.id,
    parameters: template.defaultParameters,
  };
}

export function setValueSpecDistributionFamily(
  valueSpec: StudioValueSpec,
  family: string
): StudioValueSpec {
  const defaults: Record<string, Record<string, unknown>> = {
    uniform: { min: 0, max: 1 },
    normal: { mean: 0, std: 1 },
    log_uniform: { min: 0.0001, max: 0.01 },
    categorical: { values: [0, 1] },
  };
  return {
    ...setValueSpecMode(valueSpec, 'distribution'),
    distribution: {
      family,
      parameters: defaults[family] ?? {},
    },
  };
}
