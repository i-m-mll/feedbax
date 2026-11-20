import { ChangeKeySchema } from '@/generated/studioContracts';

/** Whether the project grammar accepts `name` as a document or step key. */
export function isMemberName(name: string) {
  return ChangeKeySchema.safeParse({ kind: 'change_key', family: 'steps', member: name, key: name }).success;
}
