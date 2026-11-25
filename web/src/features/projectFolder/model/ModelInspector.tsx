import {
  ModelAddressSchema, type ComponentDefinition, type DelayAttachment, type DelayInitialTask, type DelayInitialValue, type DelayInitialWire,
  type InterventionAttachment, type JsonValue, type ModelAddress, type ModelDocument, type ModelTime, type ModelWire,
} from '@/generated/studioContracts';
import { studioName, studioReason, studioUnit } from '@/components/activity/studioVocabulary';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type { FieldCommit, FieldModel } from '@/components/values/fieldModel';
import { fromParamSchema } from '@/components/values/adapters/paramSchema';
import { arrayParts, rawField, type RawChange } from '@/components/values/adapters/raw';
import { addressKey, type Component } from './modelGraph';

export type Attachment = NonNullable<ModelWire['attachments']>[number];
/** Writes a raw value at a path under the element; answers with a reason when the edit is refused. */
export type FieldChange = (path: string[], value: JsonValue) => Promise<string | undefined>;
export type AttachmentChange = (item: Attachment) => Promise<string | undefined> | void;

function period(raw: string) {
  const value = Number(raw.trim().replace(/\s*s$/, ''));
  return raw.trim() && Number.isFinite(value) && value > 0 ? value : undefined;
}

function duration(raw: string): ModelTime | undefined {
  const match = raw.trim().match(/^(\d+(?:\.\d+)?|\.\d+)\s*(s|st)$/);
  if (!match) return;
  const value = Number(match[1]);
  const unit = match[2] as ModelTime['unit'];
  return value > 0 && (unit === 's' || Number.isInteger(value)) ? { value, unit } : undefined;
}

export function delayText(delay: DelayAttachment['delay']): string {
  return 'text' in delay ? delay.text : `${delay.value} ${delay.unit}`;
}

/** The model's own fields, shown when nothing is selected. */
export function modelFields(model: ModelDocument, change?: FieldChange): FieldModel[] {
  const label = studioName('step');
  return [rawField({
    id: 'step_seconds', label, type: { kind: 'string' }, value: typeof model.step_seconds === 'number' ? `${model.step_seconds} ${studioUnit('second')}` : model.step_seconds.text,
    change: change && ((raw) => {
      const value = period(String(raw));
      return value === undefined ? studioReason('takesPositive', label) : change(['step_seconds'], value);
    }),
  })];
}

/** A component's raw fields: a composite's substeps, a leaf's declared parameters that the server does not project. */
export function componentFields(component: Component, definition: ComponentDefinition | undefined, change?: FieldChange,
  projected?: ReadonlySet<string>): FieldModel[] {
  if (component.kind === 'composite') {
    const domain = component.interior.physical_domain;
    if (projected?.has('substeps') || (domain !== 'planar_multibody' && domain !== 'spatial_multibody')) return [];
    const label = studioName('substeps');
    return [rawField({
      id: 'substeps', label, type: { kind: typeof component.substeps === 'object' ? 'string' : 'integer' },
      value: typeof component.substeps === 'object' ? component.substeps.text : component.substeps ?? 1,
      change: change && ((value) => typeof value === 'number' && value > 0 ? change(['substeps'], value) : studioReason('takesPositive', label)),
    })];
  }
  const values = { ...definition?.default_params, ...component.fields };
  return (definition?.param_schema ?? [])
    .filter(({ name }) => !projected?.has(name))
    .map((schema) => fromParamSchema(schema, values[schema.name] as JsonValue | undefined,
      change && ((value) => change(['fields', schema.name], value))));
}

/**
 * An attachment's raw fields. Its sources (`attachmentSources`) and an intervention's Window are the caller's rows,
 * since each can read a Task variable or a port.
 */
export function attachmentFields(item: Attachment, change?: AttachmentChange,
  projected?: ReadonlySet<string>): FieldModel[] {
  const write = (next: Attachment): ReturnType<RawChange> => change?.(next);
  if (item.kind === 'hold') return [];
  if (item.kind === 'delay') {
    if (projected?.has('delay')) return [];
    const label = studioName('delay');
    return [rawField({
      id: 'delay', label, type: { kind: 'string' }, value: delayText(item.delay),
      change: change && ((raw) => {
        const delay = duration(String(raw));
        return delay ? write({ ...item, delay }) : studioReason('takesDuration', label);
      }),
    })];
  }
  if (item.kind === 'probe') {
    const label = studioName('name');
    return [rawField({
      id: 'name', label, type: { kind: 'string' }, value: item.name,
      change: change && ((raw) => String(raw).trim() ? write({ ...item, name: String(raw).trim() }) : studioReason('takesText', label)),
    })];
  }
  return [rawField({
    id: 'operation', label: studioName('operation'), value: operationOf(item) ?? item.operation,
    type: { kind: 'choice', options: operations.map((operation) => ({ value: operation, label: studioName(operation) })) },
    change: change && ((operation) => write({ ...item, operation: String(operation), fields: operationDefaults[operation as Operation] })),
  })];
}

const operations = ['add', 'scale', 'clamp'] as const;
type Operation = typeof operations[number];
const operationDefaults: Record<Operation, Record<string, JsonValue>> = {
  add: { value: [0] }, scale: { factor: 1 }, clamp: { low: -1, high: 1 },
};

/** An intervention's operation, by its stored name; none for a name no operation has, which the Service refuses. */
export function operationOf(item: InterventionAttachment): Operation | undefined {
  const operation = item.operation.toLowerCase();
  return (operations as readonly string[]).includes(operation) ? operation as Operation : undefined;
}

/** What a source reads: its own value, a Task variable, or the signal of a port of the same graph (none yet). */
export type SourceValue =
  | { readonly kind: 'value'; readonly value: JsonValue | undefined }
  | { readonly kind: 'task'; readonly variable: string }
  | { readonly kind: 'wire'; readonly port: ModelAddress | undefined };

/**
 * A value an attachment reads from one of three sources: a delay's Initial, an intervention's operation fields.
 * `field` is the path under the attachment of the field the Service projects for its own value.
 */
export interface AttachmentSource {
  readonly id: string;
  readonly label: string;
  readonly field: string;
  readonly read: SourceValue;
  /** The value it holds when it reads nothing: the number or numbers of a new attachment. */
  readonly unlinked: JsonValue;
  readonly set?: (value: SourceValue) => ReturnType<RawChange>;
}

/** A port's address in `graph`, as a model address names it. */
function portAddress(graph: ModelAddress, end: DelayInitialWire['source']): ModelAddress | undefined {
  if (!end) return undefined;
  return { ...graph, element: { kind: 'port', end: end.owner === 'component' ? { ...end, direction: 'output' } : end } };
}

/** The end a delay's Initial reads: an output of a component, or one of the graph's Inputs. */
function signalEnd(port: ModelAddress | undefined): DelayInitialWire['source'] {
  if (port?.element.kind !== 'port') return null;
  const { end } = port.element;
  if (end.owner === 'component') return { owner: 'component', component: end.component, port: end.port };
  return end.owner === 'inputs' ? { owner: 'inputs', port: end.port } : null;
}

function parsedAddress(reference: string) {
  try {
    const parsed = ModelAddressSchema.safeParse(JSON.parse(reference));
    return parsed.success ? parsed.data : undefined;
  } catch (error) {
    if (error instanceof SyntaxError) return undefined;
    throw error;
  }
}

/** An intervention field's stored value as a source: a number or numbers, or a name, a port's address or a Task variable's id. */
function fieldSource(value: JsonValue | undefined): SourceValue {
  if (!value || typeof value !== 'object' || Array.isArray(value) || value.kind !== 'name' || typeof value.reference !== 'string') {
    return { kind: 'value', value };
  }
  const port = parsedAddress(value.reference);
  return port ? { kind: 'wire', port } : { kind: 'task', variable: value.reference };
}

function storedField(value: SourceValue, unlinked: JsonValue): JsonValue {
  switch (value.kind) {
    case 'value': return value.value ?? unlinked;
    case 'task': return { kind: 'name', reference: value.variable };
    case 'wire': return value.port ? { kind: 'name', reference: addressKey(value.port) } : unlinked;
  }
}

/** The sources of an attachment in `graph`, the graph that holds it. */
export function attachmentSources(item: Attachment, graph: ModelAddress, change?: AttachmentChange): AttachmentSource[] {
  if (item.kind === 'delay') {
    const initial = item.initial_source ?? { kind: 'value', value: 0 };
    const read: SourceValue = initial.kind === 'wire' ? { kind: 'wire', port: portAddress(graph, initial.source) }
      : initial.kind === 'task' ? initial : { kind: 'value', value: initial.value as JsonValue };
    return [{
      id: 'initial', label: studioName('initial'), field: 'initial_source.value', read, unlinked: 0,
      set: change && ((value) => {
        if (value.kind === 'value' && typeof value.value !== 'number') return studioReason('takesNumber', studioName('initial'));
        const initial_source: DelayInitialValue | DelayInitialTask | DelayInitialWire = value.kind === 'value'
          ? { kind: 'value', value: value.value as number } : value.kind === 'task' ? value : { kind: 'wire', source: signalEnd(value.port) };
        return change({ ...item, initial_source });
      }),
    }];
  }
  if (item.kind !== 'intervention') return [];
  const operation = operationOf(item);
  return Object.entries(operation ? operationDefaults[operation] : {}).map(([name, unlinked]) => ({
    id: name, label: humanizeLabel(name), field: `fields.${name}`, read: fieldSource(item.fields?.[name]), unlinked,
    set: change && ((value) => change({ ...item, fields: { ...item.fields, [name]: storedField(value, unlinked) } })),
  }));
}

/** A source's own value as a raw field, a number or numbers side by side; `linked` names what it reads instead. */
export function sourceValueField(source: AttachmentSource, linked: string | undefined): FieldModel {
  const { id, label, read, set } = source;
  if (linked !== undefined || read.kind !== 'value') return rawField({ id, label, type: { kind: 'readOnly' }, value: linked ?? '' });
  const { value } = read;
  const write: RawChange | undefined = set && ((next) => set({ kind: 'value', value: next }));
  if (Array.isArray(value)) {
    return rawField({ id, label, type: { kind: 'vector' }, value, change: write, parts: arrayParts(id, label, value, write,
      (index, part, change) => rawField({ id: `${id}.${index}`, label: String(index + 1), type: { kind: 'number' }, value: part, change })) });
  }
  if (value && typeof value === 'object') return rawField({ id, label, type: { kind: 'readOnly' }, value: 'text' in value ? String(value.text) : '' });
  return rawField({ id, label, type: { kind: 'number' }, value: typeof value === 'number' ? value : 0, change: write });
}

function sameValue(models: readonly FieldModel[]) {
  const [first, ...rest] = models;
  return rest.every((model) => model.text === first.text && JSON.stringify(model.value) === JSON.stringify(first.value));
}

/**
 * The fields several selected elements share, matched by id: one row each, its value `Mixed` (text null) where they
 * differ. Committing sets it on every element in turn and answers with the first refusal.
 */
export function sharedFields(lists: readonly (readonly FieldModel[])[]): FieldModel[] {
  if (lists.length < 2) return [...lists[0] ?? []];
  const [first, ...rest] = lists;
  return first.flatMap((model) => {
    const matches = [model, ...rest.map((list) => list.find(({ id, type }) => id === model.id && type.kind === model.type.kind))];
    if (matches.some((match) => !match)) return [];
    const all = matches as FieldModel[];
    const commits = all.map(({ commit }) => commit);
    const commit: FieldCommit | undefined = commits.every(Boolean) ? async (edit) => {
      for (const each of commits) {
        const reason = await each!(edit);
        if (reason) return reason;
      }
      return undefined;
    } : undefined;
    const same = sameValue(all);
    return [{
      ...model, commit, members: undefined, parts: same ? model.parts : undefined,
      reason: all.find(({ reason }) => reason)?.reason, span: same ? model.span : undefined,
      ...(same ? {} : { text: null, value: undefined }),
    }];
  });
}
