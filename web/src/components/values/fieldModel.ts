import type {
  FieldKind, FieldKindEdit, FieldNameEdit, FieldTextEdit, FunctionInfo, JsonValue, ScopedValue,
  StudioAddClassMember, StudioRemoveClassMember, ValueType,
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

/** The changed mark of a value a difference of the shown document sets: its text, and the Reset that gives the value the inherited one. */
export interface ChangeMark {
  readonly text: string;
  readonly reset?: () => Promise<string | undefined>;
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
  /** The value set on the selected runs, where it differs from the document's. */
  readonly differs?: string;
  readonly origin?: string;
  /** The value is a difference of the shown document: its origin is the changed mark, and offers Reset. */
  readonly changed?: boolean;
  /** Shows the object the origin names, where it names one. */
  readonly revealOrigin?: () => void;
  readonly reason?: string;
  readonly names: readonly ScopedValue[];
  readonly functions: readonly FunctionInfo[];
  readonly commit?: FieldCommit;
}

export function hasMark(model: FieldModel) {
  return model.kinds.length > 1 || model.kinds.some(({ kind }) => kind !== 'value');
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
