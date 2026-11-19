import { useState } from 'react';
import type { JsonValue, ParamSchema, SiteProjection, StudioFieldView } from '@/generated/studioContracts';
import { ValueField } from '@/components/values/ValueField';
import { fromFieldView, type ProjectedEdit } from '@/components/values/adapters/fieldView';
import { setIn } from '@/components/values/adapters/jsonSchema';
import { SettingsForm } from '@/components/values/SettingsForm';
import { fromParamSchema } from '@/components/values/adapters/paramSchema';

const site = (kinds: SiteProjection['kinds']): SiteProjection => ({
  site: {
    expected: { base: 'real' }, finest: 0,
    names: [
      { id: 'hidden', name: 'hidden_size', type: { base: 'integer' }, level: 0 },
      { id: 'tau', name: 'tau', type: { base: 'real', dimensions: { time: 1 } }, level: 0 },
    ],
  },
  kinds,
  functions: [
    { id: 'linspace', signature: 'linspace(first, last, count)', display_class: 'sweep_rule', min_args: 3, max_args: 3 },
    { id: 'geomspace', signature: 'geomspace(first, last, count)', display_class: 'sweep_rule', min_args: 3, max_args: 3 },
    { id: 'sample', signature: 'sample(draw, count)', display_class: 'sweep_rule', min_args: 2, max_args: 2 },
  ],
});

/** A stand-in for the server's field projection: numbers are constants, top-level commas a list sweep. */
function useProjected(label: string, initial: string, kinds: SiteProjection['kinds']) {
  const [field, setField] = useState<StudioFieldView>({ label, text: initial, site: site(kinds), can_sweep: true, can_change_axis: true });
  const commit = async (edit: ProjectedEdit) => {
    if (edit.kind === 'to_sweep') { setField({ ...field, text: field.text + ',', sweep: list([field.text]) }); return; }
    if (edit.kind === 'to_value') { setField({ ...field, sweep: null, text: field.sweep?.kind === 'list' ? String(field.sweep.values[0].value) : field.text }); return; }
    if (edit.kind !== 'text') return;
    const parts = edit.text.split(',').map((part) => part.trim()).filter(Boolean);
    if (parts.some((part) => !Number.isFinite(Number(part)))) return `${label} takes Value, Sweep, Formula`;
    setField(parts.length > 1 || edit.text.includes(',')
      ? { ...field, text: parts.join(', '), sweep: list(parts) } : { ...field, text: parts[0], sweep: null });
  };
  return { field, commit };
}

function list(values: readonly string[]): StudioFieldView['sweep'] {
  return { kind: 'list', id: 'axis', editable: true, values: values.map((value, index) => ({ id: `v${index}`, label: null, value: Number(value) })) };
}

const params: ParamSchema[] = [
  { name: 'use_bias', type: 'bool', default: true },
  { name: 'nonlinearity', type: 'str', options: ['tanh', 'relu', 'gelu'], default: 'tanh' },
  { name: 'hidden_size', type: 'int', default: 64 },
  { name: 'noise_std', type: 'float', default: 0.01 },
  { name: 'label', type: 'str', default: 'policy' },
  { name: 'init_state', type: 'array', default: [0, 0, 0] },
  { name: 'bounds', type: 'bounds2d', default: [-1, 1] },
  { name: 'optimizer', type: 'object', default: { learning_rate: 0.001, beta: 0.9 }, nested_schema: [
    { name: 'learning_rate', type: 'float' }, { name: 'beta', type: 'float' },
  ] },
  { name: 'checkpoint', type: 'object', default: { path: 'run/7' } },
];

/** A method's settings: a referenced group with free parameters, an optional number, and a discriminated union. */
const settingsSchema: JsonValue = {
  type: 'object',
  properties: {
    n_reaches: { type: 'integer', title: 'Reaches', default: 8 },
    workspace: { type: 'array', minItems: 2, maxItems: 2, items: { type: 'number' }, default: [0.2, 0.3] },
    optimizer: { $ref: '#/$defs/Optimizer' },
    grad_clip: { anyOf: [{ type: 'number' }, { type: 'null' }], default: 1, title: 'Grad Clip' },
    schedule: { oneOf: [{ $ref: '#/$defs/Constant' }, { $ref: '#/$defs/Cosine' }] },
  },
  $defs: {
    Optimizer: { title: 'Optimizer', type: 'object', properties: {
      type: { type: 'string', enum: ['adam', 'sgd'], title: 'Type' },
      params: { type: 'object', additionalProperties: { anyOf: [{ type: 'number' }, { type: 'boolean' }] } },
    } },
    Constant: { type: 'object', properties: { kind: { const: 'constant' } } },
    Cosine: { type: 'object', properties: { kind: { const: 'cosine' }, warmup: { type: 'integer', default: 100 } } },
  },
};

function useRaw(initial: Record<string, JsonValue>) {
  const [values, setValues] = useState(initial);
  return { values, change: (key: string) => (value: JsonValue) => { setValues({ ...values, [key]: value }); } };
}

export function ValueFieldGallery() {
  const hidden = useProjected('Hidden size', '64', ['value', 'sweep', 'formula']);
  const rate = useProjected('Learning rate', '0.001', ['value', 'sweep', 'formula']);
  const step = useProjected('Step', '0.01', ['value']);
  const decay = useProjected('Decay', 'learning_rate / 10', ['value', 'sweep', 'formula']);
  const raw = useRaw(Object.fromEntries(params.map(({ name, default: value }) => [name, value as JsonValue])));
  const [settings, setSettings] = useState<JsonValue>({
    optimizer: { type: 'adam', params: { learning_rate: 0.001, nesterov: false } }, schedule: { kind: 'constant' },
  });
  const rangeField: StudioFieldView = {
    label: 'Clip', text: 'geomspace(0.1, 10, 3)', site: site(['value', 'sweep', 'formula']),
    sweep: { kind: 'range', id: 'clip', rule: { start: 0.1, stop: 10, count: 3, scale: 'log' }, can_list: true },
    members: [{ id: 'a', label: null, value: 0.1 }, { id: 'b', label: null, value: 1 }, { id: 'c', label: null, value: 10 }],
    can_change_axis: true,
  };
  const project = (view: ReturnType<typeof useProjected>, extra = {}) =>
    fromFieldView(view.field, { id: view.field.label, commit: view.commit, ...extra });
  return (
    <section className="grid gap-4" aria-label="ValueField">
      <h2 className="text-sm font-medium text-shell-muted">ValueField</h2>
      <div className="grid gap-6 md:grid-cols-2">
        <div className="grid w-80 content-start gap-2 rounded-control border border-shell-divider p-2" aria-label="Projected">
          <ValueField model={project(hidden)} />
          <ValueField model={project(rate)} />
          <ValueField model={project(step)} />
          <ValueField model={project(decay, { kind: 'formula' })} />
          <ValueField model={fromFieldView(rangeField, { id: 'clip', commit: async () => undefined })} />
          <ValueField model={fromFieldView({ ...hidden.field, label: 'Selected', effective: '96' },
            { id: 'selected', commit: hidden.commit, rowSelection: true })} />
          <ValueField model={fromFieldView({ ...hidden.field, label: 'Several' },
            { id: 'several', commit: hidden.commit, rowSelection: true, multipleRows: true })} />
        </div>
        <div className="grid w-80 content-start gap-2 rounded-control border border-shell-divider p-2" aria-label="Parameters">
          {params.map((schema) => <ValueField key={schema.name}
            model={fromParamSchema(schema, raw.values[schema.name], raw.change(schema.name))} />)}
        </div>
        <div className="grid w-80 content-start gap-2 rounded-control border border-shell-divider p-2" aria-label="SettingsForm">
          <SettingsForm schema={settingsSchema} value={settings}
            write={(path, value) => { setSettings((current) => setIn(current, path, value)); }} />
          <ValueField model={fromParamSchema({ name: 'locked', type: 'float', default: 3 }, undefined)} />
        </div>
      </div>
      <div className="grid gap-2" aria-label="Cells">
        <div className="flex items-start gap-2">
          <div className="w-[90px]"><ValueField form="cell" model={project(hidden)} /></div>
          <div className="w-[90px]"><ValueField form="cell" model={project(step)} /></div>
          <div className="w-[90px]"><ValueField form="cell" model={fromParamSchema(params[2], raw.values.hidden_size, raw.change('hidden_size'))} /></div>
        </div>
      </div>
    </section>
  );
}
