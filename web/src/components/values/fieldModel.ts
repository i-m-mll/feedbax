import type {
  FieldKind, FieldKindEdit, FieldNameEdit, FieldTextEdit, FunctionInfo, JsonValue, ScopedValue,
  ValueType,
} from '@/generated/studioContracts';

/** What a field holds, which decides its widget. A vector, list or group is drawn through its parts. */
export type FieldType =
  | { readonly kind: 'number' | 'integer' | 'string' | 'readOnly' | 'vector' | 'list' | 'group' }
  | { readonly kind: 'choice'; readonly options: readonly FieldOption[] }
  | { readonly kind: 'boolean' };

export interface FieldOption {
  readonly value: JsonValue;
  readonly label: string;
}

/** An edit the field asks its source to make. A source answers with a reason when it refuses. */
export type FieldEdit = FieldTextEdit | FieldKindEdit | FieldNameEdit | { readonly kind: 'set'; readonly value: JsonValue };
export type FieldCommit = (edit: FieldEdit) => Promise<string | undefined>;

export interface FieldKindOffer {
  readonly kind: FieldKind;
  readonly enabled: boolean;
}

export interface FieldMember {
  readonly id: string;
  readonly label: string | null;
  readonly text: string;
  readonly selected: boolean;
}

export interface FieldMembers {
  readonly values: readonly FieldMember[];
  /** Members may be edited and named. */
  readonly editable: boolean;
  /** Members may be added. */
  readonly extendable: boolean;
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
  /** The values a sweep can be reduced to, for choosing Value. */
  readonly keep?: readonly { readonly id: string; readonly text: string; readonly selected: boolean }[];
  /** The value's parts: a vector's components side by side; otherwise indented rows under the field (a group's members, a rule's first, last and count). */
  readonly parts?: readonly FieldModel[];
  /** The sweep may be turned into its listed values. */
  readonly canList?: boolean;
  /** The value set on the selection differs from the document's. */
  readonly differs: boolean;
  readonly origin?: string;
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

/** Names and functions whose names start with the identifier ending at the caret. */
export function completions(model: FieldModel, before: string): { token: string; items: readonly Completion[] } {
  const token = before.match(/[A-Za-z_][A-Za-z0-9_]*$/)?.[0] ?? '';
  if (!token) return { token, items: [] };
  const lower = token.toLowerCase();
  const names = model.names.filter(({ name }) => name.toLowerCase().startsWith(lower) && name !== token)
    .map(({ id, name, type }) => ({ id: 'name:' + id, insert: name, label: name, detail: typeText(type) }));
  const functions = model.functions.filter(({ id }) => id.toLowerCase().startsWith(lower))
    .map(({ id, signature }) => ({ id: 'function:' + id, insert: id + '(', label: id, detail: signature }));
  return { token, items: [...names, ...functions] };
}
