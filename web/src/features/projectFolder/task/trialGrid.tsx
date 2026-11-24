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

type Structure = Extract<Parameters<Commit>[1], { kind: 'structure' }>['change'];

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
 * An epoch's End: its length's field, with Rest among its kinds. Choosing Rest, or committing `rest`, makes the epoch
 * Rest; a Rest epoch's text commits a length, which is written as a length the field then holds.
 */
function endField(view: TrialView, task: TaskDocument, index: number, commit?: Commit): FieldModel {
  const epoch = task.epochs[index];
  const id = `end:${epoch.id}`;
  const label = studioName('end');
  const name = studioAccessibleName('set', `${epoch.name} ${label}`);
  const write = (end: TaskDocument['epochs'][number]['end']) => commit!(name, {
    kind: 'task', edits: [{ op: 'replace', path: `/epochs/${index}/end`, value: end as never }],
  });
  const end = epoch.end;
  if (!('length' in end)) {
    return {
      ...rawField({ id, label, value: 'rest', type: { kind: 'string' } }),
      kind: 'rest', kinds: [{ kind: 'value', enabled: !!commit }, { kind: 'rest', enabled: !!commit }],
      commit: commit && (async (edit) => {
        if (edit.kind !== 'text' || edit.text.trim() === 'rest') return undefined;
        const made = await write({ kind: 'length', length: { kind: 'literal', value: 0, unit: 's' } });
        if (made) return made;
        const refused = await commit(name, { kind: 'expression', path: authoredPointer(['task', 'epochs', epoch.id, 'end', 'length']),
          edit: { kind: 'text', text: edit.text } });
        if (refused) await write({ kind: 'rest' });
        return refused;
      }),
    };
  }
  const length = projected(view, ['epochs', epoch.id, 'end', 'length'], `/epochs/${index}/end/length`,
    { id, label, value: end.length, commit, name });
  return {
    ...length,
    kinds: [...length.kinds, { kind: 'rest', enabled: !!length.commit }],
    commit: length.commit && ((edit) => edit.kind === 'text' && edit.text.trim() === 'rest'
      ? write({ kind: 'rest' }) : length.commit!(edit)),
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

/** Each change in turn, stopping at the first that is refused. */
async function inTurn(changes: readonly (() => ReturnType<Commit>)[]) {
  for (const change of changes) {
    const refusal = await change();
    if (refusal) return reasonOf(refusal);
  }
  return undefined;
}

function gridEdits(task: TaskDocument, commit?: Commit): GridEdits {
  if (!commit) return {};
  const change = (name: string, structure: Structure, removes?: boolean) => () =>
    commit(name, { kind: 'structure', change: structure }, removes);
  const make = async (name: string, structure: Structure, removes?: boolean) => reasonOf(await change(name, structure, removes)());
  const epoch = (id: string) => task.epochs.find((item) => item.id === id)?.name;
  const signal = (id: string) => task.signals.find((item) => item.id === id)?.name;
  return {
    addEpoch: (name, index) => make(studioName('addEpoch'), { op: 'add', member: 'epoch', name, index }),
    renameEpoch: (id, name) => make(studioAccessibleName('rename', epoch(id)), { op: 'rename', member: 'epoch', id, name }),
    moveEpoch: (id, index) => make(studioAccessibleName('move', epoch(id)), { op: 'move', member: 'epoch', id, index }),
    removeEpochs: (ids) => inTurn(ids.map((id) =>
      change(studioAccessibleName('remove', epoch(id)), { op: 'remove', member: 'epoch', id }, true))),
    addSignal: (name) => make(studioName('addSignal'), { op: 'add', member: 'signal', name }),
    renameSignal: (id, name) => make(studioAccessibleName('rename', signal(id)), { op: 'rename', member: 'signal', id, name }),
    moveSignal: (id, index) => make(studioAccessibleName('move', signal(id)), { op: 'move', member: 'signal', id, index }),
    removeSignal: (id) => make(studioAccessibleName('remove', signal(id)), { op: 'remove', member: 'signal', id }, true),
    merge: (id, cell, first, last) => make(studioAccessibleName('merge', signal(id)),
      { op: 'set_span', signal: id, cell, first_epoch: first, last_epoch: last }),
    split: (id, cell) => make(studioAccessibleName('split', signal(id)),
      { op: 'set_span', signal: id, cell: cell.id, first_epoch: cell.first, last_epoch: cell.first }),
    absent: (id, cells) => inTurn(cells.map((cell) =>
      change(studioAccessibleName('absent', signal(id)), { op: 'absent', signal: id, cell }))),
    present: (id, epochs) => inTurn(epochs.map((at) =>
      change(studioAccessibleName('set', signal(id)), { op: 'present', signal: id, epoch: at }))),
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
