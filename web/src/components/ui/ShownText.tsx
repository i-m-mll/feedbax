/** A part of text a view shows: plain text, or the key of a referent the project lacks. */
export type TextPart = string | { readonly missing: string };

/** Text a view shows, which can hold the keys of missing referents. */
export type ShownText = TextPart | readonly TextPart[];

/** The name of `referent`, the thing reference `id` refers to; when it is missing, its key, marked missing. */
export function referentName(referent: { readonly name: string } | undefined, id: string): TextPart {
  return referent ? referent.name : { missing: id };
}

function parts(text: ShownText): readonly TextPart[] {
  return Array.isArray(text) ? text : [text as TextPart];
}

/** `texts`, with `separator` between each two. */
export function joinText(texts: readonly ShownText[], separator: string): readonly TextPart[] {
  return texts.flatMap((text, index) => [...index ? [separator] : [], ...parts(text)]);
}

/** `text` where only plain text can go: a missing referent reads as its key. */
export function plainText(text: ShownText): string {
  return parts(text).map((part) => typeof part === 'string' ? part : part.missing).join('');
}

/** Whether `text` holds a missing referent's key. */
export function holdsMissing(text: ShownText): boolean {
  return parts(text).some((part) => typeof part !== 'string');
}

/** `text`, each missing referent's key drawn as code with a wavy underline. */
export function Shown({ text }: { text: ShownText }) {
  return <>{parts(text).map((part, index) => typeof part === 'string' ? part
    : <span key={index} data-missing-reference="" className="font-mono text-shell-muted underline decoration-wavy">{part.missing}</span>)}</>;
}
