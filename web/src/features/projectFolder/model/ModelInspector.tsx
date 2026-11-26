import {
  type ComponentDefinition, type DelayInitialWire, type JsonValue, type ModelAddress,
  type StoredInterventionAttachment, type StoredModelDocument, type StoredModelSignalSource,
} from '@/generated/studioContracts';
import { studioName, studioReason } from '@/components/activity/studioVocabulary';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type { FieldCommit, FieldModel } from '@/components/values/fieldModel';
import { fromParamSchema } from '@/components/values/adapters/paramSchema';
import { rawField, type RawChange } from '@/components/values/adapters/raw';
import type { Attachment, Component } from './modelGraph';

/** Writes a raw value at a path under the element; answers with a reason when the edit is refused. */
export type FieldChange = (path: string[], value: JsonValue) => Promise<string | undefined>;
export type AttachmentChange = (item: Attachment) => Promise<string | undefined> | void;

/** Text the author typed for a stored time or expression, which the Service reads and writes in its own form. */
const typed = (raw: unknown, label: string, write: (text: string) => ReturnType<RawChange>) =>
  String(raw).trim() ? write(String(raw).trim()) : studioReason('takesText', label);

/** The model's own fields, shown when nothing is selected. */
export function modelFields(model: StoredModelDocument, change?: FieldChange): FieldModel[] {
  const label = studioName('step');
  return [rawField({
    id: 'step_seconds', label, type: { kind: 'string' }, value: model.step_seconds,
    change: change && ((raw) => typed(raw, label, (text) => change(['step_seconds'], text))),
  })];
}

/** A component's raw fields: a composite's substeps, a leaf's declared parameters that the server does not project. */
export function componentFields(component: Component, definition: ComponentDefinition | undefined, change?: FieldChange,
  projected?: ReadonlySet<string>): FieldModel[] {
  if (component.kind === 'composite') {
    const domain = component.interior.physical_domain;
    if (projected?.has('substeps') || (domain !== 'planar_multibody' && domain !== 'spatial_multibody')) return [];
    const label = studioName('substeps');
    const text = typeof component.substeps === 'string';
    return [rawField({
      id: 'substeps', label, type: { kind: text ? 'string' : 'integer' }, value: component.substeps ?? 1,
      change: change && ((value) => text ? typed(value, label, (written) => change(['substeps'], written))
        : typeof value === 'number' && value > 0 ? change(['substeps'], value) : studioReason('takesPositive', label)),
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
export function attachmentFields(item: Attachment, change?: AttachmentChange, projected?: ReadonlySet<string>): FieldModel[] {
  const write = (next: Attachment): ReturnType<RawChange> => change?.(next);
  if (item.kind === 'hold') return [];
  if (item.kind === 'delay') {
    if (projected?.has('delay')) return [];
    const label = studioName('delay');
    return [rawField({
      id: 'delay', label, type: { kind: 'string' }, value: item.delay,
      change: change && ((raw) => typed(raw, label, (delay) => write({ ...item, delay }))),
    })];
  }
  if (item.kind === 'probe') {
    const label = studioName('name');
    return [rawField({
      id: 'name', label, type: { kind: 'string' }, value: item.label,
      change: change && ((raw) => String(raw).trim() ? write({ ...item, label: String(raw).trim() }) : studioReason('takesText', label)),
    })];
  }
  return [rawField({
    id: 'operation', label: studioName('operation'), value: operationOf(item) ?? item.operation,
    type: { kind: 'choice', options: operations.map((operation) => ({ value: operation, label: studioName(operation) })) },
    // Operands left out of another operation are the Service's defaults for it.
    change: change && ((operation) => {
      const { operands: _, ...rest } = item;
      return write({ ...rest, operation: String(operation) });
    }),
  })];
}

const operations = ['add', 'scale', 'clamp'] as const;
type Operation = typeof operations[number];
/** An intervention's operation, by its stored name; none for a name no operation has, which the Service refuses. */
export function operationOf(item: StoredInterventionAttachment): Operation | undefined {
  const operation = item.operation.toLowerCase();
  return (operations as readonly string[]).includes(operation) ? operation as Operation : undefined;
}

/** What a source reads: its own value, a Task variable, or the signal of a port of the same graph (none yet). */
export type SourceValue =
  | { readonly kind: 'value'; readonly value: JsonValue | undefined }
  | { readonly kind: 'task'; readonly variable: string }
  | { readonly kind: 'wire'; readonly port: ModelAddress | undefined };

/**
 * A value an attachment reads from one of three sources: a delay's Initial, or an operand of an intervention's operation.
 * `field` is the path under the attachment of the field the Service projects for its own value.
 */
export interface AttachmentSource {
  readonly id: string;
  readonly label: string;
  readonly field: string;
  readonly read: SourceValue;
  readonly set?: (value: SourceValue) => ReturnType<RawChange>;
  /** Stores the source as reading nothing, which the Service reads as its default; none for an operand, which has no such store. */
  readonly clear?: () => ReturnType<RawChange>;
}

/** A port's address in `graph`, as a model address names it. */
function portAddress(graph: ModelAddress, end: DelayInitialWire['source']): ModelAddress | undefined {
  if (!end) return undefined;
  return { ...graph, element: { kind: 'port', end: end.owner === 'component' ? { ...end, direction: 'output' } : end } };
}

/** The end a source reads: an output of a component, or one of the graph's Inputs. */
function signalEnd(port: ModelAddress | undefined): DelayInitialWire['source'] {
  if (port?.element.kind !== 'port') return null;
  const { end } = port.element;
  if (end.owner === 'component') return { owner: 'component', component: end.component, port: end.port };
  return end.owner === 'inputs' ? { owner: 'inputs', port: end.port } : null;
}

/** A stored source as it reads in `graph`; none stored is the Service's default value, which its projected field shows. */
export function readSource(graph: ModelAddress, source: StoredModelSignalSource | undefined): SourceValue {
  if (!source) return { kind: 'value', value: undefined };
  if (source.kind === 'wire') return { kind: 'wire', port: portAddress(graph, source.source) };
  return source.kind === 'task' ? source : { kind: 'value', value: source.value as JsonValue };
}

/** One source of `item`: its label, the field projected for its own value, and its store as `write` keeps it. */
function signalSource(id: string, label: string, field: string, graph: ModelAddress, stored: StoredModelSignalSource | undefined,
  write?: (source: StoredModelSignalSource) => ReturnType<RawChange>): AttachmentSource {
  return {
    id, label, field, read: readSource(graph, stored),
    set: write && ((value) => {
      if (value.kind === 'task') return write(value);
      if (value.kind === 'wire') return write({ kind: 'wire', source: signalEnd(value.port) });
      return typeof value.value === 'number' || typeof value.value === 'string' ? write({ kind: 'value', value: value.value })
        : studioReason('takesNumber', label);
    }),
  };
}

/** The sources of an attachment in `graph`, the graph that holds it: an intervention's are the operands it stores. */
export function attachmentSources(item: Attachment, graph: ModelAddress, change?: AttachmentChange): AttachmentSource[] {
  if (item.kind === 'delay') {
    const { initial_source: _, ...unread } = item;
    return [{
      ...signalSource('initial', studioName('initial'), 'initial_source.value', graph, item.initial_source,
        change && ((initial_source) => change({ ...item, initial_source }))),
      clear: change && (() => change(unread)),
    }];
  }
  if (item.kind !== 'intervention') return [];
  return Object.entries(item.operands ?? {}).map(([name, stored]) =>
    signalSource(name, humanizeLabel(name), `operands.${name}.value`, graph, stored,
      change && ((source) => change({ ...item, operands: { ...item.operands, [name]: source } }))));
}

/** A source's own value as a raw field, a number or numbers side by side. */
export function sourceValueField(source: AttachmentSource): FieldModel {
  const { id, label, read, set } = source;
  if (read.kind !== 'value') return rawField({ id, label, type: { kind: 'readOnly' }, value: '' });
  const { value } = read;
  const write: RawChange | undefined = set && ((next) => set({ kind: 'value', value: next }));
  if (typeof value === 'string') return rawField({ id, label, type: { kind: 'readOnly' }, value });
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
