import { RenameMemberSchema } from '@/generated/studioContracts';

/** Whether the project grammar accepts `name` as a document or step name. */
export function isMemberName(name: string) {
  return RenameMemberSchema.safeParse({ kind: 'rename', family: 'steps', member: name, name }).success;
}
