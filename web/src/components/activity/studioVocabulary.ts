import { BarChart3, FileText, FlaskConical, PlayCircle } from 'lucide-react';
import type { ResultStatus } from '@/components/ui/StatusMark';
import type {
  ValueType,
  ComputationChange, DriverChanged, EnvironmentDifference, ProgramsChanged, StepCountChanged, EnvironmentNotice, RegistryBinding, StudioActivityJob, StudioEvaluateValueSource,
  StudioDefinitionChange, StudioPlanReviewSource, StudioProjectWindowMapProblem, StudioSavedEvaluationResult,
  StudioSavedTrainChange, StudioSweptParameterMissing, ValidityReason,
} from '@/generated/studioContracts';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type { ValueKind } from '@/components/values/fieldModel';
import vocabulary from './vocabulary.json';

export type StudioNameId = keyof typeof vocabulary.name;
export type StudioReasonId = keyof typeof vocabulary.reason;
export type StudioRunStatusId = StudioActivityJob['status'];
/** A run's status with its result's validity folded in: what one mark draws. */
export type StudioResultState = ResultStatus;

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

/** A name with why what it names does not apply. */
export function studioReasonedName(id: StudioNameId, reason: string) {
  return studioFormat('reasonedName', { name: studioName(id), reason });
}

export function studioUnit(id: keyof typeof vocabulary.unit) {
  return vocabulary.unit[id];
}

export function studioFieldLabel(id: StudioNameId, label: string) {
  return studioFormat('fieldLabel', { label, name: studioName(id).toLowerCase() });
}

/** The accessible name of a chosen class's only parameter, drawn in its field's box under the field's label. */
export function studioParameterLabel(field: string, parameter: string) {
  return studioFormat('fieldLabel', { label: field, name: parameter });
}

/** Why a class cannot be chosen: the step sweeps a parameter it lacks. */
export function studioSweptParameterMissing({ class_label, parameter_label }: StudioSweptParameterMissing) {
  return studioFormat('sweptParameterMissing', { class: class_label, parameter: parameter_label });
}

export function studioFieldKindLabel(label: string, kind: ValueKind) {
  return studioFormat('fieldKind', { label, name: studioName('kind').toLowerCase(), value: studioName(kind) });
}

export function studioReplayLabel(update: number) {
  return studioFormat('replayUpdate', { name: studioName('replay'), update });
}

export function studioEvaluateValueSource(source: StudioEvaluateValueSource) {
  return studioFormat(source.kind === 'trained' ? 'trainedValue' : 'evaluateChangedValue', { step: source.step });
}

/** Where a value comes from. An exception reads by the runs it picks (`runs`, as the Exceptions list reads them). */
export function studioPlanValueSource(source: StudioPlanReviewSource, runs?: string) {
  switch (source.kind) {
    case 'exception':
      return runs ? studioFormat('fieldSource', { kind: studioName('exception'), name: runs }) : studioName('exception');
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

/**
 * A format whose `{name}` slots are filled with nodes, such as a link to the object a value names,
 * rather than with text.
 */
export function studioFormatParts<T>(id: keyof typeof vocabulary.format, values: Readonly<Record<string, T>>): (string | T)[] {
  return vocabulary.format[id].split(/(\{[A-Za-z]+\})/).flatMap((part): (string | T)[] => {
    const key = /^\{([A-Za-z]+)\}$/.exec(part)?.[1];
    if (key === undefined) return part ? [part] : [];
    if (!(key in values)) throw new Error(`Missing Studio format value ${key} for ${id}`);
    return [values[key]];
  });
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

/** The place in a step's definition that a reason or problem names. */
export type StudioPlace = NonNullable<StudioProjectWindowMapProblem['place']>;

/** The name of a place, as its link draws it. */
export function studioPlaceName(place: StudioPlace | null) {
  switch (place?.part) {
    case 'model': return studioName('model');
    case 'task': return place.parameter ? humanizeLabel(place.parameter) : studioName('task');
    case 'objective': return studioName('objective');
    case 'training': return place.setting ? humanizeLabel(place.setting) : studioName('training');
    case 'method': return place.setting ? humanizeLabel(place.setting.split('.').at(-1)!) : studioName('trainingMethod');
    case 'analysis': return humanizeLabel(place.element);
    case 'input': return humanizeLabel(place.role);
    case 'step': return place.field ? humanizeLabel(place.field) : studioName('step');
    case undefined: return studioName('step');
  }
}

/** A changed value, as a stale run's reason draws it. */
export function studioChange(before: unknown, after: unknown) {
  return `${JSON.stringify(before)} → ${JSON.stringify(after)}`;
}

/** A typed reason as one line of text: the name of what it names, then what it says. */
export function studioPlaceReason(place: StudioPlace | null, detail: string) {
  return `${studioPlaceName(place)} · ${detail}`;
}

/** What a problem says, after the member of a document it names. */
export function studioProblemDetail({ detail, document_path }: StudioProjectWindowMapProblem) {
  return document_path ? `${document_path} · ${detail}` : detail;
}

/** What a definition change says: the values it changed, or the field it changed when it carries none. */
export function studioChangeDetail(change: StudioDefinitionChange | StudioSavedTrainChange) {
  return 'before' in change ? studioChange(change.before, change.after) : `${change.path} ${studioValidity('changed')}`;
}

/** A run's reason as one line of text, for its status tooltip. */
export function studioRunReason({ openingReason, problems = [], change, reason }: {
  readonly openingReason?: string;
  readonly problems?: readonly StudioProjectWindowMapProblem[];
  readonly change?: StudioDefinitionChange | StudioSavedTrainChange;
  readonly reason?: string;
}) {
  if (openingReason) return openingReason;
  if (problems.length) return problems.map((problem) => studioPlaceReason(problem.place, studioProblemDetail(problem))).join('\n');
  if (change) return studioPlaceReason(change.place, studioChangeDetail(change));
  return reason;
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

/** A type as its parts: its base, its shape, and its unit's dimensions, as in `Real · 2 · length`, after its name where it has one. */
export function studioValueType({ base, shape = [], dimensions = {} }: ValueType, name?: string) {
  const unit = Object.entries(dimensions).filter(([, exponent]) => exponent).map(([name, exponent]) =>
    exponent === 1 ? name : `${name}^${exponent}`).join('·');
  return [name, studioName(base), shape.length ? shape.join('×') : undefined, unit || undefined].filter(Boolean).join(' · ');
}
