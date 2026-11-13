import { BarChart3, FileText, FlaskConical, PlayCircle } from 'lucide-react';
import type { FieldKind, StudioActivityJob, StudioSavedEvaluationResult } from '@/generated/studioContracts';
import vocabulary from './vocabulary.json';

export type StudioNameId = keyof typeof vocabulary.name;
export type StudioReasonId = keyof typeof vocabulary.reason;
export type StudioRunStatusId = StudioActivityJob['status'];

const runStatus = vocabulary.status.run satisfies Record<StudioRunStatusId, string>;
const stepGlyphs = {
  flask: FlaskConical,
  play: PlayCircle,
  chart: BarChart3,
  page: FileText,
};

export function studioName(id: StudioNameId) {
  return vocabulary.name[id];
}

export function studioAccessibleName(id: StudioNameId, value?: string | number) {
  return value === undefined || value === '' ? studioName(id) : `${studioName(id)} ${value}`;
}

export function studioUnit(id: keyof typeof vocabulary.unit) {
  return vocabulary.unit[id];
}

export function studioFieldLabel(id: StudioNameId, label: string) {
  return studioFormat('fieldLabel', { label, name: studioName(id).toLowerCase() });
}

export function studioFieldValueLabel(label: string, index: number, naming = false) {
  const values = { label, name: studioName('value').toLowerCase(), index };
  return naming ? studioFormat('fieldValueName', { ...values, nameField: studioName('name').toLowerCase() })
    : studioFormat('fieldValue', values);
}

export function studioFieldKindLabel(label: string, kind: FieldKind) {
  return studioFormat('fieldKind', { label, name: studioName('kind').toLowerCase(), value: studioName(kind) });
}

export function studioReplayLabel(update: number) {
  return studioFormat('replayUpdate', { name: studioName('replay'), update });
}

const countFormat = new Intl.NumberFormat();

export function studioFormat(
  id: keyof typeof vocabulary.format,
  values: Readonly<Record<string, string | number>> = {},
) {
  return vocabulary.format[id].replace(/\{([A-Za-z]+)\}/g, (_, key: string) => {
    const value = values[key];
    if (value === undefined) throw new Error(`Missing Studio format value ${key} for ${id}`);
    return String(value);
  });
}

export function studioCountOf(count: number, total: number) {
  return studioFormat('ofCount', { count: countFormat.format(count), total: countFormat.format(total) });
}

export function studioRunStatus(status: StudioRunStatusId) {
  return runStatus[status];
}

export function studioMachineStatus(status: keyof typeof vocabulary.status.machine) {
  return vocabulary.status.machine[status];
}

export function studioDefinitionStatus(status: keyof typeof vocabulary.status.definition) {
  return vocabulary.status.definition[status];
}

export function studioEvaluationPlacement(placement: NonNullable<StudioSavedEvaluationResult['placement']>) {
  return vocabulary.status.evaluationPlacement[placement];
}

export function studioCheckpointStatus(status: keyof typeof vocabulary.status.checkpoint) {
  return vocabulary.status.checkpoint[status];
}

export function studioRunPhase(phase: string | null | undefined) {
  return phase && Object.prototype.hasOwnProperty.call(vocabulary.status.phase, phase)
    ? vocabulary.status.phase[phase as keyof typeof vocabulary.status.phase]
    : null;
}

export function studioReason(id: StudioReasonId, value: string) {
  return vocabulary.reason[id].replace('{value}', () => value);
}

export function studioStep(kind: keyof typeof vocabulary.step) {
  const { glyph } = vocabulary.step[kind];
  return { Icon: stepGlyphs[glyph as keyof typeof stepGlyphs], tint: `var(--studio-kind-${kind})`, name: studioName(kind) };
}
