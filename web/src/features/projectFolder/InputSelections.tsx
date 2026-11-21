import { X } from 'lucide-react';
import type { ReactNode } from 'react';
import { studioAccessibleName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { FieldRow } from '@/components/ui/FieldRow';
import { ReferenceField, type ReferenceChoice, type ReferenceGroup } from '@/components/values/ReferenceField';
import type { ElementSelectionStore } from '@/features/projectWindow/elementSelection';

export interface SelectionRow {
  readonly value: ReferenceChoice | null;
  /** What the row's own field may read instead; the input's groups when absent. */
  readonly groups?: readonly ReferenceGroup[];
  /** The row's restriction, under its field. */
  readonly detail?: ReactNode;
  /** Why the plan refuses this selection. */
  readonly problem?: string;
}

export type SelectionGesture =
  | { kind: 'add'; choice: ReferenceChoice }
  | { kind: 'replace'; index: number; choice: ReferenceChoice }
  | { kind: 'remove'; index: number };

/**
 * An input that takes a list of selections: a row per selection, each with its field, its restriction and its
 * remove, and a last row that adds one through the same picker.
 */
export function InputSelections({ label, labelId, groups, rows, picking, refusal, below, onChange }: {
  label: string;
  labelId: string;
  /** What a selection may read. */
  groups: readonly ReferenceGroup[];
  rows: readonly SelectionRow[];
  picking?: ElementSelectionStore;
  /** Why the last edit of the input was refused. */
  refusal?: string;
  /** Lines under the input that belong to no one selection. */
  below?: ReactNode;
  onChange?: (gesture: SelectionGesture) => void;
}) {
  return <FieldRow label={label} labelId={labelId}
    below={(refusal || below) && <>{refusal && <p role="alert">{refusal}</p>}{below}</>}>
    <ul aria-labelledby={labelId} className="grid min-w-0 gap-1">
      {rows.map((row, index) => <li key={index} className="grid min-w-0 gap-1">
        <div className="flex min-w-0 items-center gap-1">
          <ReferenceField className="flex-1" label={label} aria-labelledby={labelId} picking={picking}
            groups={row.groups ?? groups} value={row.value}
            onChoose={onChange && ((choice) => onChange({ kind: 'replace', index, choice }))} />
          {onChange && rows.length > 1 && <IconButton size="mark"
            label={studioAccessibleName('remove', row.value?.label)} icon={<X aria-hidden="true" />}
            onClick={() => onChange({ kind: 'remove', index })} />}
        </div>
        {row.problem && <p role="alert">{row.problem}</p>}
        {row.detail}
      </li>)}
      {onChange && <li className="flex min-w-0">
        <ReferenceField adds className="flex-1" label={label} aria-labelledby={labelId} picking={picking} groups={groups}
          value={null} onChoose={(choice) => onChange({ kind: 'add', choice })} />
      </li>}
    </ul>
  </FieldRow>;
}
