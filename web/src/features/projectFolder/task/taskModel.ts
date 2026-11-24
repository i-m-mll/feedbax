import type { ChangeMark, FieldModel } from '@/components/values/fieldModel';
import type { ValueType } from '@/generated/studioContracts';
import type { TimelineEpoch, TimelineRepeat } from '@/components/timeline/EpochTimeline';

/**
 * How often a variable's value is read again: once per trial, or every step when it reads `time` or the model. A
 * variable read once per trial carries no mark.
 */
export type VariableLevel = 'trial' | 'step';

/** A value of the paired Model that a variable may initialize: read once, when a trial starts. */
export interface ModelStart {
  /** The Model field's address, as a link names it. */
  readonly id: string;
  /** Its components' names and its path. */
  readonly name: string;
  /** The Model has no start of its own for it, so a variable must initialize it. */
  readonly required: boolean;
}

/** The Model values a variable initializes, and those its type fits. */
export interface VariableInitializes {
  readonly chosen: readonly ModelStart[];
  readonly options: readonly ModelStart[];
  /** The problem of one of its links. */
  readonly reason?: string;
  readonly changed?: ChangeMark;
  /** Initializes the value, or stops initializing it; a refusal's reason, or nothing once made. */
  toggle?(id: string): Promise<string | undefined> | void;
}

/** A Model value no variable initializes, at the foot of Variables with its type fixed and its name empty. */
export interface UninitializedStart extends ModelStart {
  /** The type's name. */
  readonly type: string;
  readonly valueType: ValueType;
}

export interface TaskVariable {
  readonly id: string;
  readonly name: string;
  /** The type's name, from the catalogue's types. */
  readonly type: string;
  /** The type as its parts: its base, shape and unit's dimensions. */
  readonly valueType: ValueType;
  /** The type as a choice of the catalogue's types. */
  readonly typeChoice: FieldModel;
  /** A list's length. */
  readonly count?: number;
  readonly level: VariableLevel;
  readonly value: FieldModel;
  readonly initializes?: VariableInitializes;
}

export interface TaskEpoch {
  readonly id: string;
  readonly name: string;
  /** The epoch's End: a length's kinds, Until, or Rest. */
  readonly end: FieldModel;
  /** The epoch's least, nominal and greatest length, as the compiled task reports them. */
  readonly bounds: TimelineEpoch;
}

/** Adjacent epochs from `first` to `last` repeated `count` times; the count is written as the total. */
export interface TaskRepeat {
  readonly id: string;
  readonly first: string;
  readonly last: string;
  readonly count: FieldModel;
}

/** A signal's value over adjacent epochs from `first` to `last`: a field, or absent (no value at all, not zero). */
export type TaskCell =
  | { readonly id: string; readonly first: string; readonly last: string; readonly kind: 'value'; readonly value: FieldModel }
  | { readonly id: string; readonly first: string; readonly last: string; readonly kind: 'absent' };

export interface TaskSignal {
  readonly id: string;
  readonly name: string;
  readonly type: string;
  /** Covers every epoch once, in order. */
  readonly cells: readonly TaskCell[];
}

/** A task document as the Task view draws it. */
export interface TaskDocumentView {
  readonly kind: 'document';
  readonly variables: readonly TaskVariable[];
  /** The paired Model's values no variable initializes. */
  readonly starts?: readonly UninitializedStart[];
  readonly epochs: readonly TaskEpoch[];
  readonly repeats: readonly TaskRepeat[];
  readonly signals: readonly TaskSignal[];
  /** The trial's whole length, in seconds; Rest ends it. */
  readonly duration: number;
  /** The paired model's step, in seconds, when a model is paired. */
  readonly stepSeconds?: number;
  readonly changed?: ChangeMark;
}

/** A task an installed package defines in Python: a form over its parameters, and the signals it declares. */
export interface TaskPythonView {
  readonly kind: 'python';
  readonly parameters: readonly FieldModel[];
  readonly signals: readonly { readonly id: string; readonly name: string; readonly type: string }[];
}

export type TaskView = TaskDocumentView | TaskPythonView;

/** What is selected in the Task view, which the Inspector shows. */
export type TaskSelection =
  | { readonly kind: 'variable' | 'epoch' | 'end' | 'signal' | 'repeat'; readonly id: string }
  | { readonly kind: 'cell'; readonly signal: string; readonly id: string };

export const sameSelection = (a: TaskSelection | undefined, b: TaskSelection | undefined) =>
  !!a && !!b && a.kind === b.kind && a.id === b.id && (a.kind !== 'cell' || b.kind === 'cell' && a.signal === b.signal);
