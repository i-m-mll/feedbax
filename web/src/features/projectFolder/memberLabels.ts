import type { Document, StudioProjectMemberNames, StudioProjectSnapshot } from '@/generated/studioContracts';

/** The labels each expanded variant of a project gives its members, which a variant can change. */
export type VariantNames = StudioProjectSnapshot['variant_names'];

/** The documents a view can name: every document of the project it holds, by key. */
export type ShownDocuments = Readonly<Record<string, Document>>;

/** The label of member `key` of `members`, a family of the project the caller holds. */
export function labelOf(members: Readonly<Record<string, { readonly label: string }>> | null | undefined, key: string) {
  const member = members?.[key];
  if (!member) throw new Error(`The project holds no member ${key}.`);
  return member.label;
}

/** The labels of the members of `variant`, as its expansion gives them. */
export function variantNames(variants: VariantNames, variant: string): StudioProjectMemberNames {
  const named = variants[variant];
  if (!named) throw new Error(`The project snapshot names no variant ${variant}.`);
  return named;
}

/** The label `table`, one family of a variant's labels, gives member `key`. */
export function nameIn(table: Readonly<Record<string, string>> | undefined, key: string) {
  const name = table?.[key];
  if (name === undefined) throw new Error(`The variant names no member ${key}.`);
  return name;
}

/**
 * The documents the workspace shows: those of the draft it edits, and of `shown`, the project the shown plan was compiled
 * from, which can still hold a document the draft has removed.
 */
export function shownDocuments(draft: { readonly documents?: ShownDocuments | null }, shown?: ShownDocuments | null): ShownDocuments {
  return { ...shown, ...draft.documents };
}
