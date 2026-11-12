import { BarChart3, FileText, FlaskConical, PlayCircle } from 'lucide-react';
import type { StudioActivityJob } from '@/generated/studioContracts';
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

export function studioAccessibleName(id: StudioNameId, authoredName?: string) {
  return authoredName ? `${studioName(id)} ${authoredName}` : studioName(id);
}

export function studioUnit(id: keyof typeof vocabulary.unit) {
  return vocabulary.unit[id];
}

const countFormat = new Intl.NumberFormat();

export function studioCountOf(count: number, total: number) {
  return vocabulary.format.ofCount
    .replace('{count}', countFormat.format(count))
    .replace('{total}', countFormat.format(total));
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

export function studioRunPhase(phase: string | null | undefined) {
  return phase && Object.prototype.hasOwnProperty.call(vocabulary.status.phase, phase)
    ? vocabulary.status.phase[phase as keyof typeof vocabulary.status.phase]
    : null;
}

export function studioReason(id: StudioReasonId, value: string) {
  return vocabulary.reason[id].replace('{value}', () => value);
}

export function studioStep(kind: keyof typeof vocabulary.step) {
  const { glyph, tint } = vocabulary.step[kind];
  return { Icon: stepGlyphs[glyph as keyof typeof stepGlyphs], tint, name: studioName(kind) };
}
