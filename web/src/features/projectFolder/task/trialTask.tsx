import { useAuthoringCatalog } from '@/api/studioAuthoring';
import { studioAccessibleName, studioFormat, studioName, studioValueType } from '@/components/activity/studioVocabulary';
import { reasonOf, type FieldModel } from '@/components/values/fieldModel';
import { fromFieldView } from '@/components/values/adapters/fieldView';
import { rawField } from '@/components/values/adapters/raw';
import type { StudioNamedValueType, TaskDocument, ValueType } from '@/generated/studioContracts';
import { authoredPointer } from '../authoredEdits';
import {
  changedAt, expressionKind, variableAnchor, type Commit, type TrialRuns, type TrialView,
} from '../StudioTrialAuthoring';
import { VariableInspector } from './TaskInspector';
import { TrialTaskGrid } from './trialGrid';
import { TaskVariables, type VariableEdits } from './TaskVariables';
import { TaskDocumentBody } from './TaskView';
import type { TaskSelection, TaskVariable } from './taskModel';

const sameType = (a: ValueType, b: ValueType) => studioValueType(a) === studioValueType(b);

/** The variables of the task document the step reads, each with its Type and Value fields. */
function variables(view: TrialView, task: TaskDocument, types: readonly StudioNamedValueType[], commit?: Commit,
  runs?: TrialRuns): TaskVariable[] {
  const levels = new Map(view.variables.map(({ id, level }) => [id, level]));
  return task.variables.map((variable, index): TaskVariable => {
    const named = types.find(({ type }) => sameType(type, variable.type));
    const name = named?.name ?? studioValueType(variable.type);
    const options = [...named ? [] : [{ value: name, label: name }], ...types.map(({ name }) => ({ value: name, label: name }))];
    const typeChoice = rawField({
      id: `type:${variable.id}`, label: studioName('type'), value: name, type: { kind: 'choice', options },
      change: commit && (async (chosen) => {
        const type = types.find((item) => item.name === chosen)?.type;
        if (!type) return undefined;
        return reasonOf(await commit(studioAccessibleName('set', studioName('type')),
          { kind: 'structure', changes: [{ op: 'set_type', member: 'variable', id: variable.id, type }] }));
      }),
    });
    const path = authoredPointer(['task', 'variables', variable.id, 'value']);
    const projection = view.fields.find((field) => field.path === path);
    const sweep = projection?.sweep?.id;
    const [problem] = projection?.problems ?? [];
    const value: FieldModel = projection ? {
      ...fromFieldView({ ...projection, label: studioName('value'), effective: runs?.selected ? projection.effective : null }, {
        id: `value:${variable.id}`, kind: expressionKind(variable.value),
        changed: changedAt(view, 'task', `/variables/${index}/value`, commit),
        commit: commit && ((edit) => commit(studioAccessibleName('set', variable.name), { kind: 'expression', path, edit })),
        selectedMembers: sweep ? runs?.values(sweep) : undefined,
        chooseMember: sweep && runs?.choose ? (member, extend) => runs.choose!(sweep, member, extend) : undefined,
      }),
      anchor: variableAnchor(variable.id), reason: problem?.reason, span: problem?.span ?? undefined,
    } : rawField({ id: `value:${variable.id}`, label: studioName('value'), value: '', type: { kind: 'readOnly' } });
    return {
      id: variable.id, name: variable.name, type: name, valueType: variable.type, typeChoice, value,
      level: levels.get(variable.id) === 2 ? 'step' : 'trial',
    };
  });
}

function variableEdits(task: TaskDocument, commit?: Commit): VariableEdits {
  if (!commit) return {};
  const structure = async (name: string, change: Parameters<Commit>[1] & { kind: 'structure' }, removes?: boolean) =>
    reasonOf(await commit(name, change, removes));
  const named = (id: string) => task.variables.find((variable) => variable.id === id)?.name;
  return {
    add: (name) => structure(studioName('addVariable'), { kind: 'structure', changes: [{ op: 'add', member: 'variable', name }] }),
    rename: (id, name) => structure(studioAccessibleName('rename', named(id)),
      { kind: 'structure', changes: [{ op: 'rename', member: 'variable', id, name }] }),
    remove: (id) => structure(studioAccessibleName('remove', named(id)),
      { kind: 'structure', changes: [{ op: 'remove', member: 'variable', id }] }, true),
    move: (id, index) => structure(studioAccessibleName('move', named(id)),
      { kind: 'structure', changes: [{ op: 'move', member: 'variable', id, index }] }),
  };
}

function useVariables(view: TrialView, commit?: Commit, runs?: TrialRuns) {
  const catalog = useAuthoringCatalog();
  const task = 'variables' in view.task ? view.task : undefined;
  return task && { task, variables: variables(view, task, catalog.data?.tasks.value_types ?? [], commit, runs) };
}

/** The selected variable, from the window's selection of a task element (`variable:<id>`). */
export const selectedVariable = (selected: string | undefined): TaskSelection | undefined =>
  selected?.startsWith('variable:') ? { kind: 'variable', id: selected.slice('variable:'.length) } : undefined;

/** The Task view of a task document: Variables, and the grid of epochs and signals. */
export function TrialTaskDocument({ view, commit, runs, selected, select, drawn }: {
  view: TrialView; commit?: Commit; runs?: TrialRuns; selected?: string; select(id: string): void;
  /** Each variable's value in the trial the clock shows, by variable id, when that trial was drawn. */
  drawn?: Readonly<Record<string, string>>;
}) {
  const shown = useVariables(view, commit, runs);
  if (!shown) return null;
  return <TaskDocumentBody variables={<TaskVariables variables={shown.variables} selected={selectedVariable(selected)}
    select={(selection) => select(`${selection.kind}:${selection.id}`)} edits={variableEdits(shown.task, commit)}
    drawn={drawn} />}
    grid={<TrialTaskGrid view={view} commit={commit} runs={runs} selected={selected} select={select} />} />;
}

/** The Inspector of a task document's variable: its value with the value's parts, and its type. */
export function TrialVariableInspector({ view, commit, runs, id }: { view: TrialView; commit?: Commit; runs?: TrialRuns; id: string }) {
  const variable = useVariables(view, commit, runs)?.variables.find((item) => item.id === id);
  return variable ? <VariableInspector variable={variable} /> : null;
}

/**
 * Why the step cannot read each task in the picker as it stands: the required model input it would leave unfed, by
 * the input's name.
 */
export function unfedTasks(view: TrialView): ReadonlyMap<string, string> {
  const inputs = new Map(view.inputs.map(({ address, name }) => [JSON.stringify(address), name]));
  return new Map(view.tasks.flatMap(({ document, problems }) => {
    const unfed = problems.find(({ path = [], model_address }) => path[0] === 'model' && path[1] === 'inputs' && model_address);
    const name = unfed && inputs.get(JSON.stringify(unfed.model_address));
    return name ? [[document, studioFormat('leavesUnfed', { name })] as const] : [];
  }));
}
