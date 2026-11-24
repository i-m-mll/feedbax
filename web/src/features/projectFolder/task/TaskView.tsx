import type { ReactNode } from 'react';
import { studioName, type StudioNameId } from '@/components/activity/studioVocabulary';
import { Table, TableCell, TableHeaderCell, TableRow } from '@/components/ui/Table';
import { ValueField } from '@/components/values/ValueField';
import type { TaskPythonView } from './taskModel';

function TaskSection({ name, children }: { name: StudioNameId; children: ReactNode }) {
  return <section aria-label={studioName(name)} className="grid min-w-0 gap-1">
    <h3 className="flex h-row items-center text-sm font-medium text-shell-muted">{studioName(name)}</h3>
    {children}
  </section>;
}

/** A task document's view: Variables, then the grid of epochs and signals. */
export function TaskDocumentBody({ variables, grid }: { variables: ReactNode; grid?: ReactNode }) {
  return <div className="grid min-w-0 gap-5 pb-4">
    <TaskSection name="variables">{variables}</TaskSection>
    {grid && <TaskSection name="epochs">{grid}</TaskSection>}
  </div>;
}

/** A Python task: a form over its parameters, and the signals it declares. */
export function TaskPythonBody({ view }: { view: TaskPythonView }) {
  return <div className="grid max-w-2xl gap-5 pb-4">
    <TaskSection name="parameters">
      <div className="grid max-w-md gap-1">{view.parameters.map((field) => <ValueField key={field.id} model={field} />)}</div>
    </TaskSection>
    <TaskSection name="signals">
      <Table aria-label={studioName('signals')} columns={[{ least: '9rem', fixed: true }, { least: '9rem' }]} held={1}>
        <thead><tr>
          <TableHeaderCell scope="col">{studioName('name')}</TableHeaderCell>
          <TableHeaderCell scope="col">{studioName('type')}</TableHeaderCell>
        </tr></thead>
        <tbody>{view.signals.map((signal) => <TableRow key={signal.id}>
          <TableCell className="font-mono">{signal.name}</TableCell>
          <TableCell className="text-shell-muted">{signal.type}</TableCell>
        </TableRow>)}</tbody>
      </Table>
    </TaskSection>
  </div>;
}
