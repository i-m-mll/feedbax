import type {
  FieldKind, FieldKindEdit, FieldNameEdit, FieldTextEdit, FunctionInfo, JsonValue, Level, ScopedValue, StudioEvaluateValueSource,
  StudioAddClassMember, StudioRemoveClassMember, StudioValueStructure, ValueType,
} from '@/generated/studioContracts';

/**
 * What a field holds, which decides its widget. A vector, list or group is drawn through its parts. A class is a group
 * whose value is one of several classes: `value` is the chosen class's option value and the parts are its parameters.
 */
export type FieldType =
  | { readonly kind: 'number' | 'integer' | 'string' | 'readOnly' | 'vector' | 'list' | 'group' }
  | { readonly kind: 'choice' | 'class'; readonly options: readonly FieldOption[] }
  | { readonly kind: 'boolean' };

export interface FieldOption {
  readonly value: JsonValue;
  readonly label: string;
}

/**
 * An edit the field asks its source to make. A source answers with a reason when it refuses. A swept class field adds
 * a member by a class's option value and removes one by its id.
 */
export type FieldEdit = FieldTextEdit | FieldKindEdit | FieldNameEdit | StudioAddClassMember | StudioRemoveClassMember |
  { readonly kind: 'set'; readonly value: JsonValue } | { readonly kind: 'reset' };
export type FieldCommit = (edit: FieldEdit) => Promise<string | undefined>;

/**
 * What a changed value differs from: the base of the document whose difference sets it, the document's value where the
 * field shows the selected runs' other one, or the Evaluate step whose document changes it. A field that shows the
 * document's value marks the selected runs' other one (`selected`).
 */
export type ChangeSource =
  | { readonly kind: 'base'; readonly document: string }
  | { readonly kind: 'value'; readonly text: string }
  | { readonly kind: 'selected'; readonly text: string }
  | { readonly kind: 'evaluate'; readonly step: string };

/** The changed mark of a value: what it differs from, and the Reset that gives the value the inherited one. */
export interface ChangeMark {
  readonly from: ChangeSource;
  readonly reset?: () => Promise<string | undefined>;
}

/** The mark of a value an Evaluate step changes; a value it trains is not a change, and is its origin. */
export function evaluateChange(source: StudioEvaluateValueSource | undefined): ChangeMark | undefined {
  return source?.kind === 'evaluate' ? { from: { kind: 'evaluate', step: source.step } } : undefined;
}

/** `commit`, with Reset going to `change`'s reset. A field that takes no edit takes no Reset. */
export function withReset(commit: FieldCommit | undefined, change: ChangeMark | undefined): FieldCommit | undefined {
  const reset = change?.reset;
  if (!reset || !commit) return commit;
  return (edit) => edit.kind === 'reset' ? reset() : commit(edit);
}

export interface FieldKindOffer {
  readonly kind: FieldKind;
  readonly enabled: boolean;
}

/** One value of a swept field. Its text is the value; a member is never drawn by its id. */
export interface FieldMember {
  readonly id: string;
  readonly text: string;
  readonly selected: boolean;
  /** A swept class's member: its class's option value. */
  readonly option?: JsonValue;
}

export interface FieldMembers {
  readonly values: readonly FieldMember[];
  /** Selects the member's group of runs; `extend` adds it to the selection. */
  readonly choose?: (id: string, extend: boolean) => void;
}

/**
 * Everything `ValueField` draws. Adapters at the data boundary build it; the field holds no knowledge of where it came from.
 * `text` is null when the selection holds several values (drawn as Mixed).
 */
export interface FieldModel {
  readonly id: string;
  /** The value's path in the schema of the form that draws it, when a form built it from a schema. */
  readonly path?: readonly string[];
  /** The `data-field` of the field's row, by which a command that goes to the field finds it. */
  readonly anchor?: string;
  readonly label: string;
  readonly type: FieldType;
  readonly unit?: string;
  readonly text: string | null;
  readonly value?: JsonValue;
  readonly kind: FieldKind;
  /** The kinds the site offers. One offer of Value means the field has no mark and no menu. */
  readonly kinds: readonly FieldKindOffer[];
  readonly members?: FieldMembers;
  /** The member a sweep keeps when Value is chosen from its kind menu. */
  readonly keeps?: string;
  /** A chosen class's only parameter's own name; its `label` is the field holding the class, since the parameter is that field's value. */
  readonly parameter?: string;
  /** The value's parts: a vector's components side by side; otherwise indented rows under the field (a group's members, a rule's first, last and count). */
  readonly parts?: readonly FieldModel[];
  /** The sweep may be turned into its listed values. */
  readonly canList?: boolean;
  /** Where the value comes from, when that is not a change. */
  readonly origin?: string;
  /** The value differs from what it inherits, or the selected runs hold another than the document's. */
  readonly change?: ChangeMark;
  /** Shows the object the origin names, where it names one. */
  readonly revealOrigin?: () => void;
  readonly reason?: string;
  /** What the text reads as, where the source sends it: the value's tree, its root's arguments' texts, a draw's density. */
  readonly structure?: StudioValueStructure;
  /** How often the site's consumer reads the value again. */
  readonly finest?: Level;
  readonly names: readonly ScopedValue[];
  readonly functions: readonly FunctionInfo[];
  readonly commit?: FieldCommit;
}

export function hasMark(model: FieldModel) {
  return model.kinds.length > 1 || model.kinds.some(({ kind }) => kind !== 'value');
}

/**
 * The edit that turns the field into `kind`, where one does. Formula, Draw, Shape and Link have none: the kind is
 * read from the text, so the mark shows them and no command makes them.
 */
export function kindEdit(model: FieldModel, kind: FieldKind): FieldKindEdit | undefined {
  if (kind === 'sweep' && model.kind !== 'sweep') return { kind: 'to_sweep' };
  if (kind === 'value' && model.kind === 'sweep' && model.keeps) return { kind: 'to_value', value_id: model.keeps };
  return undefined;
}

/** The kinds the field's kind menu offers: those an edit converts it to, and Draw, whose parts make the edit. */
export function kindOffers(model: FieldModel) {
  return model.kinds.filter(({ kind }) => kindEdit(model, kind) || kind === 'draw' && model.kind !== 'draw');
}

/** A draw as its parts: a family, the text of each argument (undefined while unfilled), and when it is drawn. */
export interface DrawDraft {
  readonly family: string;
  readonly args: readonly (string | undefined)[];
  readonly per: 'run' | 'trial';
}

export const drawFamilies = (model: FieldModel) => model.functions.filter(({ display_class }) => display_class === 'draw');

/** A draw may be drawn per trial only where the site reads the value again each trial. */
export const perIsFree = (model: FieldModel) => (model.finest ?? 0) >= 1;

/** The draw the field's value is, as its parts. */
export function drawOf(model: FieldModel): DrawDraft | undefined {
  const value = model.structure?.value;
  if (value?.kind !== 'draw') return undefined;
  return { family: value.family, args: model.structure!.arguments ?? [], per: value.per ?? 'trial' };
}

/** The arguments a family takes: each named parameter, or for a family of any count, at least its least. */
function slots(family: FunctionInfo) {
  return family.parameters?.length ? family.parameters.length : Math.max(family.min_args, 1);
}

/** `draft` as a draw of `family`: a parameter keeps its text where the family has one of the same name. */
export function withFamily(model: FieldModel, draft: DrawDraft | undefined, family: FunctionInfo): DrawDraft {
  const previous = draft && drawFamilies(model).find(({ id }) => id === draft.family);
  const named = new Map((previous?.parameters ?? []).map(({ name }, index) => [name, draft!.args[index]]));
  const args = family.parameters?.length
    ? family.parameters.map(({ name }) => named.get(name))
    : !previous?.parameters?.length && draft ? draft.args : Array.from({ length: slots(family) }, () => undefined);
  return { family: family.id, args, per: draft?.per ?? (perIsFree(model) ? 'trial' : 'run') };
}

/** The text a filled draft writes; undefined while a part is empty. */
export function drawText(draft: DrawDraft) {
  if (draft.args.some((arg) => !arg?.trim())) return undefined;
  return `${draft.family}(${[...draft.args.map((arg) => arg!.trim()), ...draft.per === 'run' ? ['per=run'] : []].join(', ')})`;
}

export function typeText({ base, shape }: ValueType) {
  return shape?.length ? `${base}[${shape.join('×')}]` : base;
}

export interface Completion {
  readonly id: string;
  readonly insert: string;
  readonly label: string;
  readonly detail: string;
}

/**
 * Names and functions whose names start with the identifier ending at the caret. An identifier written after a
 * number is that number's unit, as in `0.02 s` or `5 ms`, so it is offered nothing.
 */
export function completions(model: FieldModel, before: string): { token: string; items: readonly Completion[] } {
  const token = before.match(/[A-Za-z_][A-Za-z0-9_]*$/)?.[0] ?? '';
  if (!token || /(?:^|[^A-Za-z0-9_.])(?:\d+\.?\d*|\.\d+)\s*[A-Za-z_][A-Za-z0-9_]*$/.test(before)) return { token, items: [] };
  const lower = token.toLowerCase();
  const names = model.names.filter(({ name }) => name.toLowerCase().startsWith(lower) && name !== token)
    .map(({ id, name, type }) => ({ id: 'name:' + id, insert: name, label: name, detail: typeText(type) }));
  const functions = model.functions.filter(({ id }) => id.toLowerCase().startsWith(lower))
    .map(({ id, signature }) => ({ id: 'function:' + id, insert: id + '(', label: id, detail: signature }));
  return { token, items: [...names, ...functions] };
}
