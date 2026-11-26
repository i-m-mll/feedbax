import type { Document, StudioProjectMemberNames, StudioProjectSnapshot } from '@/generated/studioContracts';

/** What a snapshot names its members by: as authored, and in each variant. */
export type ProjectNames = Pick<StudioProjectSnapshot, 'names' | 'variant_names'>;

/** A snapshot's names of a project's documents, by key. */
export type DocumentNames = Readonly<Record<string, string>>;

/** The names of the members of `variant`, or of the authored project when no variant is chosen. */
export function memberNames(names: ProjectNames, variant: string | null): StudioProjectMemberNames {
  if (variant === null) return names.names;
  const named = names.variant_names[variant];
  if (!named) throw new Error(`The project snapshot names no variant ${variant}.`);
  return named;
}

/**
 * The name of member `id` in `table`. A member an edit is still saving is not yet named by the snapshot; it is named by
 * the label it was made with, `pending`.
 */
export function nameOf(table: Readonly<Record<string, string>> | undefined, id: string, pending?: { readonly label?: string | null } | null) {
  const name = table?.[id] ?? pending?.label ?? undefined;
  if (name === undefined) throw new Error(`The project snapshot names no member ${id}.`);
  return name;
}

/**
 * The names of the documents the workspace shows: the saved snapshot's; for a document an edit is still saving, the
 * label it was made with; and for a document only the shown plan's revision holds, that revision's name, `shown`.
 */
export function documentNames({ saved, draft }: {
  readonly saved: Pick<StudioProjectSnapshot, 'names'>;
  readonly draft: { readonly documents?: Readonly<Record<string, Document>> | null };
}, shown?: DocumentNames): DocumentNames {
  const pending = Object.entries(draft.documents ?? {}).flatMap(([id, { label }]) => label ? [[id, label] as const] : []);
  return { ...shown, ...Object.fromEntries(pending), ...saved.names.documents };
}
