/**
 * A key for a new member named `name` that none of `taken` is: the name with each run of characters a member key cannot
 * hold made `_`, led by `_` where it would not start with a letter, then `-2`, `-3` and on while it is taken.
 */
export function freshKey(name: string, taken: Iterable<string>): string {
  const used = new Set(taken);
  const spelled = name.trim().replace(/[^A-Za-z0-9_-]+/g, '_');
  const base = /^[A-Za-z_]/.test(spelled) ? spelled : `_${spelled}`;
  let key = base;
  for (let count = 2; used.has(key); count += 1) key = `${base}-${count}`;
  return key;
}

/** Whether the keyed collection `members` has a member `key`. */
export const hasMember = (members: object, key: string) => Object.prototype.hasOwnProperty.call(members, key);
