import { studioAccessibleName } from '@/components/activity/studioVocabulary';
import { InlineEdit } from './InlineEdit';

/**
 * The key of a step, operation or figure, under its shown name: drawn when it differs from that name, edited in place
 * by Change Key, with the refusal of the last change beneath it.
 */
export function KeyField({ value, name, editing, onEditingChange, onCommit, refusal }: {
  value: string;
  /** The shown name the key sits under. */
  name: string;
  editing: boolean;
  onEditingChange(editing: boolean): void;
  onCommit(key: string): boolean | void;
  refusal?: string;
}) {
  if (!editing && !refusal && value === name) return null;
  return <div className="grid min-w-0 gap-0.5">
    <span className="flex min-w-0 items-center gap-1.5 text-xs text-shell-muted">
      <InlineEdit value={value} editing={editing} onEditingChange={onEditingChange} onCommit={onCommit}
        aria-label={studioAccessibleName('changeKey', name)} className="font-mono" />
    </span>
    {refusal && <p role="alert" className="text-xs">{refusal}</p>}
  </div>;
}
