/** A referent's label, or the key a reference holds when the project lacks its referent. */
export type ReferentName = { readonly name: string; readonly missing?: never } | { readonly missing: string; readonly name?: never };

/** A referent's label; a missing referent shows the key it is missing by, in code type and struck through. */
export function Referent({ name }: { name: ReferentName }) {
  return name.missing === undefined ? <>{name.name}</>
    : <span data-missing-reference="" className="font-mono text-shell-muted line-through">{name.missing}</span>;
}
