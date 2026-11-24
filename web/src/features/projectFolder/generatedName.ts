/**
 * The name Studio gives a step it creates, `base` being the step's kind, or a member, `base` being its family:
 * `base` while it is free, then the first free of `base-2`, `base-3`, …, with `separator` in place of `-`.
 */
export function generatedName(base: string, taken: (name: string) => boolean, separator = '-') {
  let name = base;
  for (let number = 2; taken(name); number += 1) name = `${base}${separator}${number}`;
  return name;
}

/** Whether `members` holds a member called `name`. */
export function holds(members: Readonly<Record<string, unknown>> | null | undefined) {
  return (name: string) => !!members && Object.prototype.hasOwnProperty.call(members, name);
}

/** The key a label gives a file: its letters and digits in lower case, each run of anything else one `-`. */
export function keyOfLabel(label: string) {
  return label.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase()
    .replace(/[^a-z0-9_]+/g, '-').replace(/^-+|-+$/g, '');
}
