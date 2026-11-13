import { useState, type ReactNode } from 'react';
import type { ComponentDefinition, DelayAttachment, InterventionAttachment, JsonValue, ModelAddress, ModelDocument, ModelTime, ParamSchema, ProbeAttachment } from '@/generated/studioContracts';
import { ModelAddressSchema } from '@/generated/studioContracts';
import { z } from 'zod';
import { humanizeLabel } from '@/components/values/ValueSpecField';
import { addressKey, type Component, type Port } from './modelGraph';

export type Attachment = DelayAttachment | InterventionAttachment | ProbeAttachment;
const inputClass = 'min-w-0 w-full rounded border border-shell-divider bg-shell-chrome px-1 py-0.5';
const action = 'rounded px-1.5 py-0.5 hover:bg-shell-hover disabled:opacity-50';

export function Field({ label, value, parse, commit, type = 'text', suffix }: {
  label: string; value: string; parse(value: string): JsonValue | undefined;
  commit?: (value: JsonValue) => void; type?: 'text' | 'number'; suffix?: ReactNode;
}) {
  const [invalid, setInvalid] = useState(false);
  return <form className="grid gap-1" onSubmit={(event) => {
    event.preventDefault();
    if (!commit) return;
    const raw = new FormData(event.currentTarget).get('value');
    const parsed = typeof raw === 'string' ? parse(raw) : undefined;
    setInvalid(parsed === undefined);
    if (parsed !== undefined) commit(parsed);
  }}>
    <label className="grid gap-1">{label}
      <input key={value} className={inputClass} name="value" type={type} step="any" defaultValue={value}
        disabled={!commit} aria-invalid={invalid || undefined} onChange={() => setInvalid(false)} />
    </label>
    {suffix}
  </form>;
}

function numeric(raw: string): number | undefined {
  const value = Number(raw);
  return raw.trim() && Number.isFinite(value) ? value : undefined;
}

function period(raw: string) {
  const value = numeric(raw.trim().replace(/\s*s$/, ''));
  return value !== undefined && value > 0 ? value : undefined;
}

function duration(raw: string) {
  const match = raw.trim().match(/^(\d+(?:\.\d+)?|\.\d+)\s*(s|st)$/);
  if (!match) return;
  const value = Number(match[1]);
  const unit = match[2];
  return value > 0 && (unit === 's' || Number.isInteger(value)) ? { value, unit } : undefined;
}

function conversion(delay: ModelTime, step: number) {
  return delay.unit === 's' ? `${Math.round(delay.value / step)} st` : `${Number((delay.value * step).toPrecision(8))} s`;
}

function linkedAddress(value: string) {
  try {
    return ModelAddressSchema.parse(JSON.parse(value));
  } catch (error) {
    if (error instanceof SyntaxError || error instanceof z.ZodError) return undefined;
    throw error;
  }
}

function fieldValue(value: JsonValue | undefined, schema: ParamSchema) {
  const options = schema.options?.map((option) => typeof option === 'object' ? JSON.stringify(option) : String(option));
  return { options, value: value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value) };
}

function Parameter({ name, schema, value, change }: {
  name: string; schema: ParamSchema; value: JsonValue | undefined; change?: (value: JsonValue) => void;
}) {
  const label = humanizeLabel(name);
  const field = fieldValue(value, schema);
  if (typeof value === 'boolean') return <label className="flex items-center gap-2">
    <input type="checkbox" checked={value} disabled={!change} onChange={(event) => change?.(event.currentTarget.checked)} />{label}
  </label>;
  if (field.options?.length) return <label className="grid gap-1">{label}
    <select className={inputClass} value={field.value} disabled={!change}
      onChange={(event) => change?.(schema.options![Number(event.currentTarget.selectedOptions[0].dataset.index)] as JsonValue)}>
      {field.options.map((option, index) => <option key={option} data-index={index} value={option}>{option}</option>)}
    </select>
  </label>;
  if (Array.isArray(value) && value.every((item) => typeof item === 'number' || typeof item === 'string')) {
    return <fieldset className="grid gap-1"><legend>{label}</legend>
      {value.map((item, index) => <Field key={index} label={String(index + 1)} value={String(item)}
        type={typeof item === 'number' ? 'number' : 'text'} parse={typeof item === 'number' ? numeric : (raw) => raw}
        commit={change && ((next) => change(value.map((old, position) => position === index ? next : old)))} />)}
    </fieldset>;
  }
  if (value && typeof value === 'object') return <dl><dt>{label}</dt><dd>{field.value}</dd></dl>;
  return <Field label={label} value={field.value} type={typeof value === 'number' ? 'number' : 'text'}
    parse={typeof value === 'number' ? numeric : (raw) => raw} commit={change} />;
}

export function ModelFields({ model, change }: { model: ModelDocument; change?: (path: string[], value: JsonValue) => void }) {
  return <Field label="Step" value={`${model.step_seconds} s`} parse={period} commit={change && ((value) => change(['step_seconds'], value))} />;
}

export function ComponentFields({ component, definition, change }: {
  component: Component; definition?: ComponentDefinition; change?: (path: string[], value: JsonValue) => void;
}) {
  if (component.kind === 'composite') {
    const domain = component.interior.physical_domain;
    return domain === 'planar_multibody' || domain === 'spatial_multibody'
      ? <Field label="Substeps" value={String(component.substeps ?? 1)} type="number"
        parse={(raw) => { const value = numeric(raw); return value && value > 0 && Number.isInteger(value) ? value : undefined; }}
        commit={change && ((value) => change(['substeps'], value))} /> : null;
  }
  const values = { ...definition?.default_params, ...component.fields };
  return <>{definition?.param_schema?.filter(({ name }) => name !== 'dt' && name !== 'control_dt').map((schema) =>
    <Parameter key={schema.name} name={schema.name} schema={schema} value={values[schema.name] as JsonValue | undefined}
      change={change && ((value) => change(['fields', schema.name], value))} />)}</>;
}

export function PortFields({ port }: { port: Port }) {
  if (typeof port.type === 'string') return <span>{humanizeLabel(port.type)}</span>;
  const type = [port.type.base, port.type.shape?.length ? `[${port.type.shape.join(', ')}]` : null, port.type.unit].filter(Boolean).join(' ');
  return type ? <span>{type}</span> : null;
}

export function AttachmentFields({ item, step, change, arm, linkLabel }: {
  item: Attachment; step: number; change?: (value: Attachment) => void; arm?: () => void;
  linkLabel?: (address: ModelAddress) => string;
}) {
  if (item.kind === 'delay') {
    const initial = item.initial;
    const linked = initial && typeof initial === 'object' && !Array.isArray(initial) &&
      initial.kind === 'name' && typeof initial.reference === 'string' ? initial.reference : null;
    const parsed = linked ? linkedAddress(linked) : undefined;
    const label = parsed ? linkLabel?.(parsed) || addressKey(parsed) : linked;
    return <>
      <Field label="Delay" value={`${item.delay.value} ${item.delay.unit}`} parse={duration}
        suffix={<small>{conversion(item.delay, step)}</small>}
        commit={change && ((delay) => change({ ...item, delay: delay as unknown as ModelTime }))} />
      {linked ? <div>Initial <output>{label}</output>
        {change && <button className={action} type="button" onClick={() => change({ ...item, initial: 0 })}>Clear Link</button>}
      </div> : <Field label="Initial" value={String(initial ?? 0)} type="number" parse={numeric}
        commit={change && ((value) => change({ ...item, initial: value }))} />}
      {arm && <button className={action} type="button" onClick={arm}>Pick Initial</button>}
    </>;
  }
  if (item.kind === 'probe') return <Field label="Name" value={item.name} parse={(raw) => raw.trim() || undefined}
    commit={change && ((value) => change({ ...item, name: String(value) }))} />;
  const clamp = item.operation.toLowerCase() === 'clamp';
  const names = clamp ? ['low', 'high'] : ['factor'];
  return <>
    <label className="grid gap-1">Operation
      <select className={inputClass} value={item.operation.toLowerCase()} disabled={!change} onChange={(event) => {
        const operation = event.currentTarget.value;
        change?.({ ...item, operation, fields: operation === 'clamp' ? { low: -1, high: 1 } : { factor: 1 } });
      }}>
        <option value="clamp">Clamp</option><option value="scale">Scale</option>
      </select>
    </label>
    {names.map((name) => <Field key={name} label={humanizeLabel(name)} type="number"
      value={String(item.fields?.[name] ?? '')} parse={numeric}
      commit={change && ((value) => change({ ...item, fields: { ...item.fields, [name]: value } }))} />)}
    <div>Window <output>Whole trial</output></div>
  </>;
}
