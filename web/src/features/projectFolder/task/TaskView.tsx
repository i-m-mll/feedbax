import type { ReactNode } from 'react';
import { studioName, type StudioNameId } from '@/components/activity/studioVocabulary';
import { Table, TableCell, TableHeaderCell, TableRow } from '@/components/ui/Table';
import { EpochTimeline, type TimelineContent } from '@/components/timeline/EpochTimeline';
import { ValueField } from '@/components/values/ValueField';
import { TaskGrid, type GridEdits } from './TaskGrid';
import { TaskPerTrial } from './TaskPerTrial';
import { TaskVariables, type VariableEdits } from './TaskVariables';
import { TrialStepper, type TrialChoice } from './TrialStepper';
import type { TaskDocumentView, TaskPythonView, TaskSelection } from './taskModel';
import type { TrialShown } from './trialSample';

function TaskSection({ name, children }: { name: StudioNameId; children: ReactNode }) {
  return <section aria-label={studioName(name)} className="grid min-w-0 gap-1">
    <h3 className="flex h-row items-center text-sm font-medium text-shell-muted">{studioName(name)}</h3>
    {children}
  </section>;
}

/** What the strip draws: the definition, the shown trial's boundaries, or the batch's shares. */
export function timelineContent(view: TaskDocumentView, shown: TrialShown): TimelineContent {
  if (shown.kind === 'trial') return { kind: 'trial', intervals: shown.trial.intervals };
  if (shown.kind === 'batch') return { kind: 'batch', epochs: shown.batch.epochs };
  return {
    kind: 'definition',
    epochs: view.epochs.map(({ bounds }) => bounds),
    repeats: view.repeats.map(({ first, last, count }) => ({ first, last, count: Number(count.text) || 1 })),
  };
}

/** The selected epoch, for the strip to draw as selected. */
const selectedEpoch = (selected?: TaskSelection) => selected && (selected.kind === 'epoch' || selected.kind === 'end') ? selected.id : undefined;

/** A task document: Variables, the strip over the grid, and Per Trial. */
export function TaskDocumentBody({ view, shown, choice, sampleCount, onChoose, onResample, selected, select, variableEdits, gridEdits }: {
  view: TaskDocumentView;
  shown: TrialShown;
  choice: TrialChoice;
  /** How many trials the stepper steps through. */
  sampleCount: number;
  onChoose(choice: TrialChoice): void;
  onResample(): void;
  selected?: TaskSelection;
  select(selection: TaskSelection): void;
  variableEdits: VariableEdits;
  gridEdits: GridEdits;
}) {
  const drawn = shown.kind === 'trial' ? shown.trial.variables : shown.kind === 'batch' ? shown.batch.variables : undefined;
  return <div className="grid min-w-0 gap-5 pb-4">
    <TaskSection name="variables">
      <TaskVariables variables={view.variables} drawn={drawn} selected={selected} select={select} edits={variableEdits}
        drawnLabel={shown.kind === 'batch' ? studioName('allTrials') : studioName('trial')} />
    </TaskSection>
    <TaskSection name="epochs">
      <EpochTimeline content={timelineContent(view, shown)} duration={view.duration} stepSeconds={view.stepSeconds}
        selected={selectedEpoch(selected)} onSelect={(id) => select({ kind: 'epoch', id })} className="mb-1 pl-40"
        trailing={<TrialStepper choice={choice} count={sampleCount} onChoose={onChoose} onResample={onResample} />} />
      <TaskGrid epochs={view.epochs} repeats={view.repeats} signals={view.signals} selected={selected} select={select} edits={gridEdits} />
    </TaskSection>
    <TaskSection name="perTrial">
      <TaskPerTrial rows={view.perTrial} selected={selected} select={select} />
    </TaskSection>
  </div>;
}

/** A Python task: a form over its parameters, the signals it declares, and Per Trial. */
export function TaskPythonBody({ view, selected, select }: {
  view: TaskPythonView; selected?: TaskSelection; select(selection: TaskSelection): void;
}) {
  return <div className="grid max-w-2xl gap-5 pb-4">
    <TaskSection name="parameters">
      <div className="grid max-w-md gap-1">{view.parameters.map((field) => <ValueField key={field.id} model={field} />)}</div>
    </TaskSection>
    <TaskSection name="signals">
      <Table aria-label={studioName('signals')} className="table-fixed">
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
    <TaskSection name="perTrial">
      <TaskPerTrial rows={view.perTrial} selected={selected} select={select} />
    </TaskSection>
  </div>;
}
