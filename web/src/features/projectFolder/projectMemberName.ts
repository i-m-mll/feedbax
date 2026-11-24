import { studioReason } from '@/components/activity/studioVocabulary';
import { ChangeKeySchema } from '@/generated/studioContracts';

/** Whether the project grammar accepts `name` as a document or step key. */
export function isMemberName(name: string) {
  return ChangeKeySchema.safeParse({ kind: 'change_key', family: 'steps', member: name, key: name }).success;
}

/** Why Change Key cannot give a member `key`: the grammar refuses it, or another member of its family holds it. */
export function keyUnavailable(key: string, held: (key: string) => boolean) {
  if (!isMemberName(key)) return studioReason('invalidPath', key);
  if (held(key)) return studioReason('taken', key);
  return undefined;
}
