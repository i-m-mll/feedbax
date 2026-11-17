import type { ComponentDefinition, DelayAttachment, InterventionAttachment, JsonValue, ModelDocument, ModelTime, ProbeAttachment } from '@/generated/studioContracts';
import { studioName, studioReason, studioUnit } from '@/components/activity/studioVocabulary';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type { FieldCommit, FieldModel } from '@/components/values/fieldModel';
import { fromParamSchema } from '@/components/values/adapters/paramSchema';
import { rawField, type RawChange } from '@/components/values/adapters/raw';
import type { Component } from './modelGraph';

export type Attachment = DelayAttachment | InterventionAttachment | ProbeAttachment;
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

function conversion(delay: ModelTime, step: number) {
  return delay.unit === 's' ? `${Math.round(delay.value / step)} ${studioUnit('steps')}`
    : `${Number((delay.value * step).toPrecision(8))} ${studioUnit('second')}`;
}

/** The model's own fields, shown when nothing is selected. */
export function modelFields(model: ModelDocument, change?: FieldChange): FieldModel[] {
  const label = studioName('step');
  return [rawField({
    id: 'step_seconds', label, type: { kind: 'string' }, value: `${model.step_seconds} ${studioUnit('second')}`,
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
      id: 'substeps', label, type: { kind: 'integer' }, value: component.substeps ?? 1,
      change: change && ((value) => typeof value === 'number' && value > 0 ? change(['substeps'], value) : studioReason('takesPositive', label)),
    })];
  }
  const values = { ...definition?.default_params, ...component.fields };
  return (definition?.param_schema ?? [])
    .filter(({ name }) => name !== 'dt' && name !== 'control_dt' && !projected?.has(name))
    .map((schema) => fromParamSchema(schema, values[schema.name] as JsonValue | undefined,
      change && ((value) => change(['fields', schema.name], value))));
}

/** An attachment's raw fields. A delay's Initial is the caller's row, since it can hold a link to a port. */
export function attachmentFields(item: Attachment, step: number, change?: AttachmentChange,
  projected?: ReadonlySet<string>): FieldModel[] {
  const write = (next: Attachment): ReturnType<RawChange> => change?.(next);
  if (item.kind === 'delay') {
    if (projected?.has('delay')) return [];
    const label = studioName('delay');
    return [rawField({
      id: 'delay', label, type: { kind: 'string' }, value: `${item.delay.value} ${item.delay.unit}`,
      origin: conversion(item.delay, step),
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
  const clamp = item.operation.toLowerCase() === 'clamp';
  const names = clamp ? ['low', 'high'] : ['factor'];
  return [
    rawField({
      id: 'operation', label: studioName('operation'), value: clamp ? 'clamp' : 'scale',
      type: { kind: 'choice', options: [{ value: 'clamp', label: studioName('clamp') }, { value: 'scale', label: studioName('scale') }] },
      change: change && ((operation) => write({
        ...item, operation: String(operation), fields: operation === 'clamp' ? { low: -1, high: 1 } : { factor: 1 },
      })),
    }),
    ...names.filter((name) => !projected?.has(name)).map((name) => rawField({
      id: name, label: humanizeLabel(name), type: { kind: 'number' }, value: item.fields?.[name] ?? null,
      change: change && ((value) => write({ ...item, fields: { ...item.fields, [name]: value } })),
    })),
    rawField({ id: 'window', label: studioName('window'), type: { kind: 'readOnly' }, value: studioName('wholeTrial') }),
  ];
}

export function initialField(item: DelayAttachment, linked: string | undefined, change?: AttachmentChange): FieldModel {
  const label = studioName('initial');
  if (linked !== undefined) return rawField({ id: 'initial', label, type: { kind: 'readOnly' }, value: linked });
  return rawField({
    id: 'initial', label, type: { kind: 'number' }, value: typeof item.initial === 'number' ? item.initial : 0,
    change: change && ((value) => change({ ...item, initial: value })),
  });
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
      ...(same ? {} : { text: null, value: undefined }),
    }];
  });
}
