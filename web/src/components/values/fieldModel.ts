import type {
  ExpressionPartSite, FieldKind, FieldKindEdit, FieldNameEdit, FieldTextEdit, FieldValue, StudioFieldFunction, JsonValue, Level, NodePath, StudioFieldName,
  StudioAddClassMember, StudioRemoveClassMember, StudioValueStructure, TextSpan, ValueType,
} from '@/generated/studioContracts';
import type { EvaluateOrigin } from '@/components/activity/studioVocabulary';

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
export type FieldCommit = (edit: FieldEdit) => Promise<FieldRefusal | undefined>;

/** Why a source refused an edit: a reason, or a reason about the term of the edit's text at `span`. */
export type FieldRefusal = string | { readonly reason: string; readonly span?: TextSpan };

/** The reason of a refusal, for a control that marks the whole value. */
export const reasonOf = (refusal: FieldRefusal | undefined) => typeof refusal === 'string' ? refusal : refusal?.reason;

/**
 * What a changed value differs from: the base of the document whose difference sets it, the document's value where the
 * field shows the selected runs' other one, or the Evaluate step whose document changes it. A field that shows the
 * document's value marks the selected runs' other one (`selected`).
 */
export type ChangeSource =
  | { readonly kind: 'base'; readonly name: string }
  | { readonly kind: 'value'; readonly text: string }
  | { readonly kind: 'selected'; readonly text: string }
  | { readonly kind: 'evaluate'; readonly name: string };

/** The changed mark of a value: what it differs from, and the Reset that gives the value the inherited one. */
export interface ChangeMark {
  readonly from: ChangeSource;
  readonly reset?: () => Promise<FieldRefusal | undefined>;
}

/** The mark of a value an Evaluate step changes; a value it trains is not a change, and is its origin. */
export function evaluateChange(origin: EvaluateOrigin | undefined): ChangeMark | undefined {
  return origin?.kind === 'evaluate' ? { from: { kind: 'evaluate', name: origin.step } } : undefined;
}

/** `commit`, with Reset going to `change`'s reset. A field that takes no edit takes no Reset. */
export function withReset(commit: FieldCommit | undefined, change: ChangeMark | undefined): FieldCommit | undefined {
  const reset = change?.reset;
  if (!reset || !commit) return commit;
  return (edit) => edit.kind === 'reset' ? reset() : commit(edit);
}

/** A field's kind: one its site offers, or an epoch End's Until or Rest, which only an End's site offers. */
export type ValueKind = FieldKind | 'until' | 'rest';

/** The text of an epoch End that is Until: its condition, then its limit. */
export const untilText = (condition: string, limit: string) => `${condition} ≤ ${limit}`;

export interface FieldKindOffer {
  readonly kind: ValueKind;
  readonly enabled: boolean;
  /** The names a value of this kind reads, where they are not the field's: an Until's condition reads the Model's outputs. */
  readonly names?: readonly StudioFieldName[];
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
  readonly kind: ValueKind;
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
  /** The value in the other time unit under the Model's Step, as `30 st` beside `0.3 s`. */
  readonly paired?: string;
  /** Where the value comes from, when that is not a change. */
  readonly origin?: string;
  /** The value differs from what it inherits, or the selected runs hold another than the document's. */
  readonly change?: ChangeMark;
  /** Shows the object the origin names, where it names one. */
  readonly revealOrigin?: () => void;
  readonly reason?: string;
  /** The term of `text` that `reason` is about, when it is about one term rather than the whole value. */
  readonly span?: TextSpan;
  /** What the text reads as, where the source sends it: the value's tree, its root's arguments' texts, a draw's density. */
  readonly structure?: StudioValueStructure;
  /**
   * The value at each step of the trial the clock shows, null at a step where it is not read (outside a term's
   * window); a Shape's parts draw it as its curve.
   */
  readonly curve?: readonly (number | null)[];
  /** How often the site's consumer reads the value again. */
  readonly finest?: Level;
  readonly names: readonly StudioFieldName[];
  readonly functions: readonly StudioFieldFunction[];
  /** The functions and names that fit at each node of the value: its root, `['args', 0]` and so on. */
  readonly menus?: PartMenus;
  /** The levels of a name that belongs to a tree (a Model's states under its components), by the name's id. */
  readonly namePaths?: Readonly<Record<string, readonly string[]>>;
  readonly commit?: FieldCommit;
  /** Told as a choice's list opens and closes. */
  readonly listOpen?: (open: boolean) => void;
}

/**
 * Where the functions and names that fit at a node of the value come from: the list the view sent, or the service when
 * the node's menu opens. `key` names what a served menu reads, so a menu of another view is never shown as this one's.
 */
export type PartMenus =
  | { readonly kind: 'listed'; readonly parts: readonly ExpressionPartSite[] }
  | { readonly kind: 'served'; readonly key: readonly unknown[]; read(part: NodePath, signal: AbortSignal): Promise<ExpressionPartSite> };

/** A node's menu: its site once read, none while it is read, or why it could not be. */
export type PartMenu =
  | { readonly kind: 'read'; readonly site: ExpressionPartSite | undefined }
  | { readonly kind: 'reading' }
  | { readonly kind: 'failed'; readonly reason: string };

export function hasMark(model: FieldModel) {
  return model.kinds.length > 1 || model.kinds.some(({ kind }) => kind !== 'value');
}

/** The edit that turns the field into `kind`, where an edit of the kind's own does: a sweep, and a sweep's Value. */
export function kindEdit(model: FieldModel, kind: ValueKind): FieldKindEdit | undefined {
  if (kind === 'sweep' && model.kind !== 'sweep') return { kind: 'to_sweep' };
  if (kind === 'value' && model.kind === 'sweep' && model.keeps) return { kind: 'to_value', value_id: model.keeps };
  return undefined;
}

/** The kinds the field's kind menu lists: every kind its site takes. A kind the value's type does not fit gives a Problem. */
export const kindOffers = (model: FieldModel) => model.kinds;

const literalVector = (value: FieldValue): boolean => value.kind === 'literal' ||
  value.kind === 'call' && value.function === 'vector' && !Object.keys(value.kwargs ?? {}).length && !!value.args?.length &&
  value.args.every(literalVector);

/** The kind a value reads as: a call is a Shape when its function is one, and a vector of numbers is a Value. */
export function valueKind(value: FieldValue, functions: readonly StudioFieldFunction[]): FieldKind {
  switch (value.kind) {
    case 'list': case 'range': case 'sample': return 'sweep';
    case 'draw': return 'draw';
    case 'name': return 'link';
    case 'unresolved': return 'formula';
    case 'call': {
      if (literalVector(value)) return 'value';
      return functions.find(({ id }) => id === value.function)?.display_class === 'shape' ? 'shape' : 'formula';
    }
    default: return 'value';
  }
}

/** A draw as its parts: a family, the text of each argument (undefined while unfilled), and when it is drawn. */
export interface DrawDraft {
  readonly family: string;
  readonly args: readonly (string | undefined)[];
  readonly per: 'run' | 'trial';
}

/** A family whose first parameter is the draw's centre (a mean, a region's centre). */
export const centred = (family: StudioFieldFunction) => ['mean', 'centre'].includes(family.parameters?.[0]?.name ?? '');

/** The centre of a draw whose family has one. */
export function drawCentre(model: FieldModel, draw: DrawDraft) {
  const family = drawFamilies(model).find(({ id }) => id === draw.family);
  return family && centred(family) ? draw.args[0] : undefined;
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
function slots(family: StudioFieldFunction) {
  return family.parameters?.length ? family.parameters.length : Math.max(family.min_args, 1);
}

/** `draft` as a draw of `family`: a parameter keeps its text where the family has one of the same name. */
export function withFamily(model: FieldModel, draft: DrawDraft | undefined, family: StudioFieldFunction): DrawDraft {
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
 * number is that number's unit, as in `0.02 s` or `5 ms`, so it is offered nothing. Inside a bracket, where `i` is
 * the repetition, a list is inserted reading the repetition's item.
 */
export function completions(model: FieldModel, before: string): { token: string; items: readonly Completion[] } {
  const token = before.match(/[A-Za-z_][A-Za-z0-9_]*$/)?.[0] ?? '';
  if (!token || /(?:^|[^A-Za-z0-9_.])(?:\d+\.?\d*|\.\d+)\s*[A-Za-z_][A-Za-z0-9_]*$/.test(before)) return { token, items: [] };
  const lower = token.toLowerCase();
  const repeated = model.names.some(({ id }) => id === 'i');
  const names = model.names.filter(({ name }) => name.toLowerCase().startsWith(lower) && name !== token)
    .map(({ id, name, type, list_count }) => ({
      id: 'name:' + id, insert: repeated && list_count ? `${name}[i]` : name, label: name, detail: typeText(type),
    }));
  const functions = model.functions.filter(({ id }) => id.toLowerCase().startsWith(lower))
    .map(({ id, signature }) => ({ id: 'function:' + id, insert: id + '(', label: id, detail: signature }));
  return { token, items: [...names, ...functions] };
}

/** A call as its parts: its function, and the text of each positional argument (undefined while unfilled). */
export interface CallDraft {
  readonly function: string;
  readonly args: readonly (string | undefined)[];
}

/** The call the field's value is, as its parts. */
export function callOf(model: FieldModel): CallDraft | undefined {
  const value = model.structure?.value;
  if (value?.kind !== 'call') return undefined;
  return { function: value.function, args: model.structure!.arguments ?? [] };
}

/** The names of a function's positional parameters, from its signature (`x[, delta]`, `start, stop, over=T`). */
export function parameterNames(info: StudioFieldFunction) {
  if (info.parameters?.length) return info.parameters.map(({ label }) => label);
  const inner = info.signature.includes('(') ? info.signature.slice(info.signature.indexOf('(') + 1, info.signature.lastIndexOf(')'))
    : info.signature;
  return inner.split(',').map((part) => part.replace(/[[\]]/g, '').trim()).filter((part) => part && !part.includes('='));
}

/**
 * `draft` as a call of `info`: its arguments kept, those past the function's last dropped, and those it needs beyond
 * them unfilled.
 */
export function withFunction(draft: CallDraft, info: StudioFieldFunction): CallDraft {
  const kept = info.max_args === null ? draft.args : draft.args.slice(0, info.max_args);
  const needed = Math.max(info.min_args, 1);
  return { function: info.id, args: kept.length >= needed ? kept : [...kept, ...Array.from({ length: needed - kept.length }, () => undefined)] };
}

/**
 * The text a filled call writes; undefined while an argument is empty. The stored call's keyword arguments are kept
 * where the new function takes them.
 */
export function callText(model: FieldModel, draft: CallDraft) {
  if (draft.args.some((arg) => !arg?.trim())) return undefined;
  const stored = callOf(model);
  const value = model.structure?.value;
  const keywords = value?.kind === 'call' ? Object.keys(value.kwargs ?? {}) : [];
  const info = model.functions.find(({ id }) => id === draft.function);
  const prefix = stored && `${stored.function}(${stored.args.join(', ')}`;
  const text = model.text ?? '';
  const tail = prefix && text.startsWith(prefix) && keywords.every((key) => info?.keywords?.includes(key)) ? text.slice(prefix.length) : ')';
  return `${draft.function}(${draft.args.map((arg) => arg!.trim()).join(', ')}${tail}`;
}

const siteOf = (menu: PartMenu) => menu.kind === 'read' ? menu.site : undefined;

/**
 * The functions a Formula's root may be: `shown`, and those that fit where the stored value's root is, by its `menu`;
 * only penalties where the stored function is one.
 */
export function rootFunctions(model: FieldModel, shown: string, menu: PartMenu) {
  const root = siteOf(menu);
  const stored = callOf(model)?.function ?? shown;
  const fitting = model.functions.filter(({ id, display_class }) => id === shown || id === stored ||
    root?.functions.includes(id) && display_class !== 'draw' && display_class !== 'shape' && display_class !== 'sweep_rule');
  const penalty = model.functions.find(({ id }) => id === stored)?.display_class === 'penalty';
  return penalty ? fitting.filter(({ id, display_class }) => display_class === 'penalty' || id === shown) : fitting;
}

export const shapeFunctions = (model: FieldModel) => model.functions.filter(({ display_class }) => display_class === 'shape');

/** The Shape a value becomes: the site's first shape of one argument, or else its first, starting from `text`. */
export function shapeFrom(model: FieldModel, text: string): CallDraft | undefined {
  const shapes = shapeFunctions(model);
  const shape = shapes.find(({ min_args }) => min_args <= 1) ?? shapes[0];
  return shape && withFunction({ function: shape.id, args: text.trim() ? [text] : [] }, shape);
}

/** The names a Link may read: `reference`, and those that fit at the value's root, by its `menu`. */
export function linkNames(model: FieldModel, reference: string | undefined, menu: PartMenu) {
  const root = siteOf(menu);
  return model.names.filter(({ id }) => id === reference || root?.names.includes(id));
}

/** A name as text: an identifier as itself, another name quoted. */
export const nameText = (name: string) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) ? name : JSON.stringify(name);
