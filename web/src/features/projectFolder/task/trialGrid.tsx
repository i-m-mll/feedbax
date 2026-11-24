import { useAuthoringCatalog } from '@/api/studioAuthoring';
import { studioAccessibleName, studioName, studioValueType } from '@/components/activity/studioVocabulary';
import { reasonOf, type FieldModel } from '@/components/values/fieldModel';
import { fromFieldView } from '@/components/values/adapters/fieldView';
import { rawField } from '@/components/values/adapters/raw';
import type { StudioNamedValueType, TaskDocument, ValueType } from '@/generated/studioContracts';
import { authoredPointer } from '../authoredEdits';
import { changedAt, expressionKind, type Commit, type TrialRuns, type TrialView } from '../StudioTrialAuthoring';
import { TaskGrid, type GridEdits } from './TaskGrid';
import { TaskInspector } from './TaskInspector';
import type { SampledTrial } from './trialSample';
import { selectionKey, selectionOf, type TaskCell, type TaskDocumentView, type TaskEpoch, type TaskSignal } from './taskModel';

type Structure = Extract<Parameters<Commit>[1], { kind: 'structure' }>['changes'][number];

/**
 * The field the service projects at `parts` of the task document, or an empty read-only one where it projects none
 * there (a task whose problems stop it before that part).
 */
function projected(view: TrialView, parts: readonly string[], stored: string, options: {
  id: string; label: string; value?: unknown; commit?: Commit; runs?: TrialRuns; name: string;
}): FieldModel {
  const path = authoredPointer(['task', ...parts]);
  const projection = view.fields.find((field) => field.path === path);
  if (!projection) return rawField({ id: options.id, label: options.label, value: '', type: { kind: 'readOnly' } });
  const { commit, runs } = options;
  const [problem] = projection.problems ?? [];
  return {
    ...fromFieldView({ ...projection, label: options.label, effective: runs?.selected ? projection.effective : null }, {
      id: options.id, kind: expressionKind(options.value as never), changed: changedAt(view, 'task', stored, commit),
      commit: commit && ((edit) => commit(studioAccessibleName('set', options.name), { kind: 'expression', path, edit })),
    }),
    reason: problem?.reason, span: problem?.span ?? undefined,
  };
}

/**
 * An epoch's End: its length's field, with Rest among its kinds. A Rest End's field shows Rest; committing `rest`
 * makes an End Rest, and a length's text makes a Rest End that length, each in one edit the service makes.
 */
function endField(view: TrialView, task: TaskDocument, index: number, commit?: Commit): FieldModel {
  const epoch = task.epochs[index];
  const label = studioName('end');
  const end = epoch.end;
  const rest = !('length' in end);
  const length = projected(view, ['epochs', epoch.id, 'end', 'length'], `/epochs/${index}/end/length`,
    { id: `end:${epoch.id}`, label, value: rest ? undefined : end.length, commit, name: `${epoch.name} ${label}` });
  return {
    ...length,
    ...rest ? { kind: 'rest', text: 'rest', structure: undefined } : {},
    kinds: [...length.kinds.filter(({ kind }) => !rest || kind !== 'sweep'), { kind: 'rest', enabled: !!length.commit }],
  };
}

const sameType = (a: ValueType, b: ValueType) => studioValueType(a) === studioValueType(b);

/** The grid of a task document: its epochs with their Ends, and its signals with their cells and absent values. */
function grid(view: TrialView, task: TaskDocument, types: readonly StudioNamedValueType[], commit?: Commit, runs?: TrialRuns) {
  const period = view.period_seconds;
  const epochs = task.epochs.map((epoch, index): TaskEpoch => {
    const steps = view.epochs.find(({ id }) => id === epoch.id)?.steps;
    return {
      id: epoch.id, name: epoch.name, end: endField(view, task, index, commit),
      bounds: {
        id: epoch.id, name: epoch.name, ends: !('length' in epoch.end) ? 'rest' : steps && steps.least !== steps.greatest ? 'drawn' : 'length',
        least: (steps?.least ?? 0) * period, nominal: (steps?.nominal ?? 0) * period, greatest: (steps?.greatest ?? 0) * period,
      },
    };
  });
  const signals = task.signals.map((signal, row): TaskSignal => {
    const type = types.find((named) => sameType(named.type, signal.type))?.name ?? studioValueType(signal.type);
    const order = task.epochs.map(({ id }) => id);
    const cells: TaskCell[] = [];
    for (let at = 0; at < order.length;) {
      const index = signal.cells.findIndex((cell) => cell.first_epoch === order[at]);
      const cell = signal.cells[index];
      if (!cell) {
        cells.push({ id: `absent:${order[at]}`, first: order[at], last: order[at], kind: 'absent' });
        at += 1;
        continue;
      }
      cells.push({
        id: cell.id, first: cell.first_epoch, last: cell.last_epoch, kind: 'value',
        value: projected(view, ['signals', signal.id, 'cells', cell.id, 'value'], `/signals/${row}/cells/${index}/value`,
          { id: `cell:${signal.id}:${cell.id}`, label: signal.name, value: cell.value, commit, runs, name: signal.name }),
      });
      at = Math.max(at + 1, order.indexOf(cell.last_epoch) + 1);
    }
    return {
      id: signal.id, name: signal.name, type, valueType: signal.type, cells,
      absent: projected(view, ['signals', signal.id, 'absent'], `/signals/${row}/absent`,
        { id: `absent:${signal.id}`, label: studioName('absent'), value: signal.absent ?? undefined, commit, name: signal.name }),
    };
  });
  return { epochs, signals };
}

function gridEdits(task: TaskDocument, commit?: Commit): GridEdits {
  if (!commit) return {};
  /** One edit making `changes` in order: all of them, or none when one is refused. */
  const make = async (name: string, changes: readonly Structure[], removes?: boolean) =>
    reasonOf(await commit(name, { kind: 'structure', changes: [...changes] }, removes));
  const epoch = (id: string) => task.epochs.find((item) => item.id === id)?.name;
  const signal = (id: string) => task.signals.find((item) => item.id === id)?.name;
  return {
    addEpoch: (name, index) => make(studioName('addEpoch'), [{ op: 'add', member: 'epoch', name, index }]),
    renameEpoch: (id, name) => make(studioAccessibleName('rename', epoch(id)), [{ op: 'rename', member: 'epoch', id, name }]),
    moveEpoch: (id, index) => make(studioAccessibleName('move', epoch(id)), [{ op: 'move', member: 'epoch', id, index }]),
    removeEpochs: (ids) => make(studioAccessibleName('remove', ids.map(epoch).join(', ')),
      ids.map((id) => ({ op: 'remove', member: 'epoch', id })), true),
    addSignal: (name) => make(studioName('addSignal'), [{ op: 'add', member: 'signal', name }]),
    renameSignal: (id, name) => make(studioAccessibleName('rename', signal(id)), [{ op: 'rename', member: 'signal', id, name }]),
    moveSignal: (id, index) => make(studioAccessibleName('move', signal(id)), [{ op: 'move', member: 'signal', id, index }]),
    removeSignal: (id) => make(studioAccessibleName('remove', signal(id)), [{ op: 'remove', member: 'signal', id }], true),
    merge: (id, cell, first, last) => make(studioAccessibleName('merge', signal(id)),
      [{ op: 'set_span', signal: id, cell, first_epoch: first, last_epoch: last }]),
    split: (id, cell) => make(studioAccessibleName('split', signal(id)),
      [{ op: 'set_span', signal: id, cell: cell.id, first_epoch: cell.first, last_epoch: cell.first }]),
    absent: (id, cells) => make(studioAccessibleName('absent', signal(id)),
      cells.map((cell) => ({ op: 'absent', signal: id, cell }))),
    present: (id, epochs) => make(studioAccessibleName('set', signal(id)),
      epochs.map((at) => ({ op: 'present', signal: id, epoch: at }))),
  };
}

function useGrid(view: TrialView, commit?: Commit, runs?: TrialRuns) {
  const catalog = useAuthoringCatalog();
  const task = 'variables' in view.task ? view.task : undefined;
  return task && { task, ...grid(view, task, catalog.data?.tasks.value_types ?? [], commit, runs) };
}

/** The grid of the task document the step reads; `selected` is the window's selection of a task element. */
export function TrialTaskGrid({ view, commit, runs, selected, select }: {
  view: TrialView; commit?: Commit; runs?: TrialRuns; selected?: string; select(id: string): void;
}) {
  const shown = useGrid(view, commit, runs);
  if (!shown) return null;
  return <TaskGrid epochs={shown.epochs} repeats={[]} signals={shown.signals} selected={selectionOf(selected)}
    select={(selection) => select(selectionKey(selection))} edits={gridEdits(shown.task, commit)} />;
}

/** Whether the window's selection of a task element names an epoch, an End, a signal or a cell. */
export const inGrid = (selected: string | undefined) => {
  const kind = selectionOf(selected)?.kind;
  return kind === 'epoch' || kind === 'end' || kind === 'signal' || kind === 'cell';
};

/** The Inspector of what is selected in the grid: an epoch with its End, or a signal with its absent value and cell. */
export function TrialGridInspector({ view, commit, runs, selected, trial }: {
  view: TrialView; commit?: Commit; runs?: TrialRuns; selected: string; trial?: SampledTrial;
}) {
  const shown = useGrid(view, commit, runs);
  if (!shown) return null;
  const document: TaskDocumentView = {
    kind: 'document', variables: [], epochs: shown.epochs, repeats: [], signals: shown.signals,
    duration: (view.n_steps ?? 0) * view.period_seconds, stepSeconds: view.period_seconds,
  };
  return <TaskInspector view={document} selected={selectionOf(selected)}
    shown={trial ? { kind: 'trial', index: 0, count: 1, trial } : { kind: 'definition' }} />;
}
