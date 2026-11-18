import { BarChart3, FileText, FlaskConical, PlayCircle } from 'lucide-react';
import type { StatusMarkStatus } from '@/components/ui/StatusMark';
import type {
  ComputationChange, DriverChanged, EnvironmentDifference, ProgramsChanged, StepCountChanged, EnvironmentNotice, FieldKind, RegistryBinding, StudioActivityJob, StudioEvaluateValueSource,
  StudioPlanReviewSource, StudioSavedEvaluationResult, ValidityReason,
} from '@/generated/studioContracts';
import vocabulary from './vocabulary.json';

export type StudioNameId = keyof typeof vocabulary.name;
export type StudioReasonId = keyof typeof vocabulary.reason;
export type StudioRunStatusId = StudioActivityJob['status'];
/** A run's status with its result's validity folded in: what one mark draws. */
export type StudioResultState = StatusMarkStatus;

const runStatus = vocabulary.status.run satisfies Record<StudioResultState, string>;
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

export function studioFieldKindLabel(label: string, kind: FieldKind) {
  return studioFormat('fieldKind', { label, name: studioName('kind').toLowerCase(), value: studioName(kind) });
}

export function studioReplayLabel(update: number) {
  return studioFormat('replayUpdate', { name: studioName('replay'), update });
}

export function studioEvaluateValueSource(source: StudioEvaluateValueSource) {
  return studioFormat(source.kind === 'trained' ? 'trainedValue' : 'evaluateChangedValue', { step: source.step });
}

export function studioPlanValueSource(source: StudioPlanReviewSource) {
  switch (source.kind) {
    case 'sweep_value':
      return studioFormat('fieldSource', { kind: source.axis ?? studioName('sweep'), name: source.value ?? source.name });
    case 'upstream':
      return studioFormat('trainedValue', { step: source.step ?? source.name });
    default:
      return studioFormat('fieldSource', { kind: studioName(source.kind), name: source.name });
  }
}

const countFormat = new Intl.NumberFormat();
const usdFormat = new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 6 });
const dateTimeFormat = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'long' });

export function studioMoney(microusd: number) {
  return usdFormat.format(microusd / 1_000_000);
}

export function studioDateTime(epochMs: number) {
  return dateTimeFormat.format(epochMs);
}

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

export function studioRunStatus(status: StudioResultState) {
  return runStatus[status];
}

export function studioValidity(id: keyof typeof vocabulary.status.validity) {
  return vocabulary.status.validity[id];
}

function binding({ family, key, version }: RegistryBinding) {
  return `${family} ${key} ${version}`;
}

function environmentChange({ distribution, before, after }: EnvironmentDifference) {
  return `${distribution} ${before} → ${after ?? studioValidity('removed')}`;
}

const drift = new Intl.NumberFormat(undefined, { style: 'percent', maximumSignificantDigits: 2 });

function computationChange(change: ComputationChange) {
  switch (change.kind) {
    case 'driver': {
      const { previous, present } = change as DriverChanged;
      return studioFormat('validityDriver', { driver: present.name, previous: previous.revision, present: present.revision });
    }
    case 'steps': {
      const { previous, present } = change as StepCountChanged;
      return studioFormat('validitySteps', { previous, present });
    }
    case 'program':
    case 'arguments':
      return studioFormat(change.kind === 'program' ? 'validityProgram' : 'validityArguments', {
        slots: (change as ProgramsChanged).slots.join(', '),
      });
    default:
      throw new Error(`Computation change without a kind: ${JSON.stringify(change)}`);
  }
}

function prefixFinding({ change, drift: measured }: Pick<EnvironmentNotice, 'change' | 'drift'>) {
  return [
    change && vocabulary.status.environmentChange[change],
    measured?.max_relative_loss != null && studioFormat('validityLossDrift', { value: drift.format(measured.max_relative_loss) }),
    measured?.relative_parameter_l2 != null &&
      studioFormat('validityParameterDrift', { value: drift.format(measured.relative_parameter_l2) }),
  ].filter((line): line is string => !!line);
}

/** A reason as the inspector draws it: its title, then one row per change. */
export interface StudioValidityRows {
  readonly title: string;
  readonly changes: readonly string[];
}

/**
 * One reason a result is unchecked or out of date. The wire omits defaulted `kind` fields, so reasons
 * are told apart by their required fields.
 */
export function studioValidityReason(reason: ValidityReason): StudioValidityRows {
  if ('basis' in reason) {
    return {
      title: studioValidity(reason.basis),
      changes: [
        ...reason.differences.map((difference) => 'path' in difference
          ? [difference.path.path, studioValidity(difference.status), ...(difference.functions ?? [])].join(' ')
          : `${binding(difference.before)} → ${difference.after ? binding(difference.after) : studioValidity('removed')}`),
        ...(reason.changes ?? []).map(computationChange),
        ...(reason.detail ? [reason.detail] : []),
      ],
    };
  }
  if ('role' in reason) {
    return {
      title: studioValidity('input'),
      changes: [`${reason.role} ${reason.member} ${studioValidity(reason.after ? 'changed' : 'removed')}`,
        ...(reason.detail ? [reason.detail] : [])],
    };
  }
  if ('path' in reason) {
    return { title: studioValidity('file'), changes: [`${reason.path.path} ${studioValidity(reason.status ?? 'changed')}`] };
  }
  return {
    title: studioValidity('environment'),
    changes: [...reason.differences.map(environmentChange), ...prefixFinding(reason), ...(reason.detail ? [reason.detail] : [])],
  };
}

export function studioEnvironmentNotice(notice: EnvironmentNotice): StudioValidityRows {
  return {
    title: `${studioValidity('environment')} · ${studioValidity(notice.check)}`,
    changes: [...notice.differences.map(environmentChange), ...prefixFinding(notice), ...(notice.detail ? [notice.detail] : [])],
  };
}

export function studioMachineStatus(status: keyof typeof vocabulary.status.machine) {
  return vocabulary.status.machine[status];
}

export function studioMachinePower(state: keyof typeof vocabulary.status.power) {
  return vocabulary.status.power[state];
}

export function studioOpeningStep(step: keyof typeof vocabulary.status.openingStep) {
  return vocabulary.status.openingStep[step];
}

export function studioOpeningStage(step: keyof typeof vocabulary.status.openingStage) {
  return vocabulary.status.openingStage[step];
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

export function studioStep(kind: keyof typeof vocabulary.step | StudioActivityJob['kind']) {
  const step = kind === 'render' ? 'analyze' : kind;
  const { glyph } = vocabulary.step[step];
  return {
    Icon: stepGlyphs[glyph as keyof typeof stepGlyphs],
    tint: `var(--studio-kind-${step})`,
    name: studioName(kind === 'render' ? 'figures' : kind),
  };
}
