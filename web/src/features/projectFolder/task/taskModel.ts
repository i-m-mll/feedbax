import type { ChangeMark, FieldModel } from '@/components/values/fieldModel';
import type { StoredVariable, ValueType } from '@/generated/studioContracts';

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
  /** Its composites' and component's names, then the segments of its path: the levels Initializes descends. */
  readonly path: readonly string[];
}

/** The Model values a variable initializes, and those its type fits. */
export interface VariableInitializes {
  /** The `data-field` a revealed problem of its links focuses. */
  readonly anchor: string;
  readonly chosen: readonly ModelStart[];
  readonly options: readonly ModelStart[];
  /** The problem of one of its links. */
  readonly reason?: string;
  readonly changed?: ChangeMark;
  /** Initializes the value, or stops initializing it; a refusal's reason, or nothing once made. */
  toggle?(id: string): Promise<string | undefined> | void;
}

/**
 * A Model value no variable initializes that the Task has a problem with, at the foot of Variables: the Model needs it,
 * or a link names no variable.
 */
export interface UnstartedValue extends ModelStart {
  /** The `data-field` a revealed problem of the value focuses. */
  readonly anchor: string;
  readonly reason: string;
  /** The type's name. */
  readonly type: string;
  readonly valueType: ValueType;
  /** The Task's variables whose type fits the value, by id. */
  readonly variables: readonly { readonly id: string; readonly name: string }[];
  /** Makes the variable initialize the value; a refusal's reason, or nothing once made. */
  initialize?(variable: string): Promise<string | undefined>;
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
  /** A list's count, as a field; none where the variable is not a list. */
  readonly count?: FieldModel;
  readonly level: VariableLevel;
  readonly value: FieldModel;
  readonly initializes?: VariableInitializes;
  /** The Set entry that adds the variable, where one does. */
  readonly entry?: EntryName;
}

/** A Set entry as the Task view names it: its stored label, and for a family the count the service gives. */
export interface EntryName {
  readonly label: string;
  readonly members?: number;
}

/** A Set entry the step's trials read: its own Task variables and links, and the store of new ones. */
export interface EntryTask {
  /** Which entry of which axis of which step, one key per entry. */
  readonly key: string;
  /** The entry's variables, by key. */
  readonly variables: Readonly<Record<string, StoredVariable>>;
  /** The entry's links: each Model address text with the key of the variable that initializes it. */
  readonly links: Readonly<Record<string, string>>;
  /** Stores the entry's variables or links, each kept as authored where omitted; resolves to a refusal's reason. */
  readonly write?: (name: string, change: { task_variables?: Record<string, StoredVariable>; task_links?: Record<string, string> }) => Promise<string | undefined>;
}

/** Which part of a co-occurring entry's weight: of trials that receive it, or of those that do not. */
export type WeightSide = 'on' | 'off';

/** An entry's weight in parts: one number, or its on and off parts where entries co-occur. */
export type SetEntryWeight = number | Readonly<Record<WeightSide, number>>;

/** A weight's edit as it ended: stored, not written because Apply To was dismissed, or refused with its reason. */
export type WeightEdit = { readonly kind: 'stored' } | { readonly kind: 'kept' } | { readonly kind: 'refused'; readonly reason: string };

/** A Set the step's trials apply: a variable whose values are the Set's entries, each with its weight in parts. */
export interface TaskSetVariable {
  /** The sweep axis the Set was moved out of, which names the variable. */
  readonly axis: string;
  readonly name: string;
  /** Each entry, then None where a trial may receive no entry, with its weight in parts when the trials are drawn. */
  readonly entries: readonly {
    readonly id: string | null; readonly label: string; readonly weight?: SetEntryWeight;
    /** How many entries a family entry stands for, as the service counts them; absent for an entry that is not a family. */
    readonly members?: number;
  }[];
  /** Sets one part of an entry's weight (`side` where entries co-occur). */
  readonly weigh?: (entry: string | null, side: WeightSide | undefined, weight: number) => Promise<WeightEdit>;
  /** The condition on Task variables a trial meets to receive an entry; a trial that fails it receives None. */
  readonly condition?: FieldModel;
}

/**
 * An epoch as a definition fixes it, in seconds: its least, nominal and greatest length. A drawn length spans least
 * to greatest around its nominal; a condition ends it somewhere between its least and its limit; Rest is what remains
 * of the trial.
 */
export interface EpochBounds {
  readonly id: string;
  readonly name: string;
  readonly ends: 'length' | 'drawn' | 'condition' | 'rest';
  readonly least: number;
  readonly nominal: number;
  readonly greatest: number;
}

export interface TaskEpoch {
  readonly id: string;
  readonly name: string;
  /** The epoch's End: a length's kinds, Until, or Rest. */
  readonly end: FieldModel;
  /** The epoch's least, nominal and greatest length, as the compiled task reports them. */
  readonly bounds: EpochBounds;
}

/** Adjacent epochs from `first` to `last` repeated `count` times; the count is written as the total. */
export interface TaskRepeat {
  readonly id: string;
  readonly first: string;
  readonly last: string;
  readonly count: FieldModel;
  /** The repeat's own problem, about its span rather than its count. */
  readonly reason?: string;
}

/**
 * A signal's value over adjacent epochs from `first` to `last`: a field, or absent, where the model reads the signal's
 * absent value and the trial records the signal as not present.
 */
export type TaskCell =
  | { readonly id: string; readonly first: string; readonly last: string; readonly kind: 'value'; readonly value: FieldModel }
  | { readonly id: string; readonly first: string; readonly last: string; readonly kind: 'absent' };

export interface TaskSignal {
  readonly id: string;
  readonly name: string;
  readonly type: string;
  /** The type as its parts: its base, shape and unit's dimensions. */
  readonly valueType: ValueType;
  /** What the model reads where the signal has no cell: zero of its type unless the document sets another. */
  readonly absent: FieldModel;
  /** Covers every epoch once, in order. */
  readonly cells: readonly TaskCell[];
}

/** A task document as the Task view draws it. */
export interface TaskDocumentView {
  readonly kind: 'document';
  readonly variables: readonly TaskVariable[];
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

/** The selection as the window's selection of a task element names it: `<kind>:<id>`, a cell's id with its signal's. */
export const selectionKey = (selection: TaskSelection) =>
  `${selection.kind}:${selection.kind === 'cell' ? JSON.stringify([selection.signal, selection.id]) : selection.id}`;

/** The selection a window's selection key names, or none where it names no element of the Task view. */
export function selectionOf(key: string | undefined): TaskSelection | undefined {
  const at = key?.indexOf(':') ?? -1;
  if (!key || at < 0) return undefined;
  const kind = key.slice(0, at);
  const id = key.slice(at + 1);
  if (kind === 'cell') {
    const [signal, cell] = JSON.parse(id) as [string, string];
    return { kind, signal, id: cell };
  }
  return kind === 'variable' || kind === 'epoch' || kind === 'end' || kind === 'signal' || kind === 'repeat'
    ? { kind, id } : undefined;
}
