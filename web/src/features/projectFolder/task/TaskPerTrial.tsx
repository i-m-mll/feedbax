import { studioName } from '@/components/activity/studioVocabulary';
import { StatusMark } from '@/components/ui/StatusMark';
import { Table, TableCell, TableHeaderCell, TableRow } from '@/components/ui/Table';
import { ValueField } from '@/components/values/ValueField';
import { sameSelection, type TaskPerTrial as PerTrialRow, type TaskSelection } from './taskModel';

/** Every model-side value the task sets once per trial, each with the variable that sets it, linked or not. */
export function TaskPerTrial({ rows, selected, select }: {
  rows: readonly PerTrialRow[]; selected?: TaskSelection; select(selection: TaskSelection): void;
}) {
  return <Table aria-label={studioName('perTrial')} className="table-fixed">
    <colgroup><col /><col className="w-48" /></colgroup>
    <thead><tr>
      <TableHeaderCell scope="col">{studioName('modelInput')}</TableHeaderCell>
      <TableHeaderCell scope="col">{studioName('variable')}</TableHeaderCell>
    </tr></thead>
    <tbody>{rows.map((row) => {
      const selection = { kind: 'perTrial', id: row.id } as const;
      const chosen = sameSelection(selection, selected);
      return <TableRow key={row.id} selected={chosen} onPointerDown={() => { if (!chosen) select(selection); }}>
        <TableCell>
          <span className="flex min-w-0 items-center gap-1.5">
            <span className="truncate font-mono">{row.name}</span>
            {row.problem && <StatusMark status="cannot_run" label={row.problem} />}
          </span>
        </TableCell>
        <TableCell className="overflow-visible py-0.5"><ValueField form="cell" model={row.link} /></TableCell>
      </TableRow>;
    })}</tbody>
  </Table>;
}
