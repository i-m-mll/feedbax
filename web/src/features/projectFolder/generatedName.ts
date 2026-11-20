/**
 * The name Studio gives a step it creates, `base` being the step's kind, or a document, `base` being its family:
 * `base` while it is free, then the first free of `base-2`, `base-3`, ….
 */
export function generatedName(base: string, taken: (name: string) => boolean) {
  let name = base;
  for (let number = 2; taken(name); number += 1) name = `${base}-${number}`;
  return name;
}

/** Whether `members` holds a member called `name`. */
export function holds(members: Readonly<Record<string, unknown>> | null | undefined) {
  return (name: string) => !!members && Object.prototype.hasOwnProperty.call(members, name);
}
