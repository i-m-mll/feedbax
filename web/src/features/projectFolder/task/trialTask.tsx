import { useAuthoringCatalog } from '@/api/studioAuthoring';
import { studioAccessibleName, studioFormat, studioName, studioValueType } from '@/components/activity/studioVocabulary';
import { reasonOf, type FieldModel } from '@/components/values/fieldModel';
import { rawField } from '@/components/values/adapters/raw';
import { ModelAddressSchema, type StudioNamedValueType, type StudioTrialReset, type TaskDocument, type ValueType } from '@/generated/studioContracts';
import { canonicalJsonV2 } from '@/utils/canonicalJsonV2';
import { authoredPointer } from '../authoredEdits';
import { addressKey } from '../model/modelGraph';
import {
  changedAt, expressionKind, initializesAnchor, startAnchor, trialField, variableAnchor, type Commit, type TrialRuns, type TrialView,
} from '../StudioTrialAuthoring';
import { VariableInspector } from './TaskInspector';
import { TrialTaskGrid } from './trialGrid';
import { TaskVariables, type VariableEdits } from './TaskVariables';
import { TaskDocumentBody } from './TaskView';
import type { ModelStart, TaskSelection, TaskVariable, UninitializedStart, VariableInitializes } from './taskModel';

const sameType = (a: ValueType, b: ValueType) => studioValueType(a) === studioValueType(b);

/** Whether a Model field of type `field` takes a variable of type `variable`, as the service checks a link. */
const fits = (field: ValueType, variable: ValueType) => {
  const target = field.shape ?? [];
  const source = variable.shape ?? [];
  const offset = target.length - source.length;
  return offset >= 0 && source.every((size, index) => size === 1 || size === target[offset + index])
    && canonicalJsonV2(field.dimensions ?? {}) === canonicalJsonV2(variable.dimensions ?? {})
    && (field.base === variable.base || field.base === 'real' && variable.base === 'integer');
};

const typeName = (type: ValueType, types: readonly StudioNamedValueType[]) =>
  types.find((named) => sameType(named.type, type))?.name ?? studioValueType(type);

function startOf(reset: StudioTrialReset): ModelStart {
  const components = (reset.address.composites?.length ?? 0) + 1;
  const name = [...reset.path.slice(0, components), reset.path.slice(components).join(' ')].join(' · ');
  return { id: addressKey(reset.address), name };
}

/** A linked Model field the paired Model does not have: its component's id and its path. */
function missingStart(id: string): ModelStart {
  const { element } = ModelAddressSchema.parse(JSON.parse(id));
  if (element.kind !== 'field' || element.parent.kind !== 'component') return { id, name: id };
  const [kind, ...members] = element.path;
  const label = kind === 'state' ? String(members[0]).split('/').join(' ') : members.join(' ');
  return { id, name: `${element.parent.id} · ${label}` };
}

/** The Model values `variable` initializes, the choice of those its type fits, and its links' problem and change. */
function initializes(view: TrialView, task: TaskDocument, variable: TaskDocument['variables'][number],
  commit?: Commit): VariableInitializes {
  const links = task.links ?? [];
  const starts = new Map(view.resets.map((reset) => [addressKey(reset.address), reset]));
  const own = links.flatMap((link, index) => link.variable === variable.id ? [index] : []);
  const fields = new Set(own.map((index) => links[index].model_field));
  const reason = view.problems.find(({ path = [] }) => path[0] === 'links' && fields.has(String(path[1])))?.reason;
  const from = view.changedFrom.task;
  const differs = from && view.task_changed?.some((path) =>
    own.some((index) => path === `/links/${index}` || path.startsWith(`/links/${index}/`)));
  return {
    anchor: initializesAnchor(variable.id),
    chosen: [...fields].map((field) => { const reset = starts.get(field); return reset ? startOf(reset) : missingStart(field); }),
    options: view.resets.filter((reset) => fits(reset.type, variable.type)).map(startOf),
    reason,
    changed: differs ? {
      from, reset: commit && (() => commit(studioName('reset'), { kind: 'reset', document: 'task', paths: ['/links'] })),
    } : undefined,
    toggle: commit && (async (field) => reasonOf(await commit(studioAccessibleName('set', studioName('initializes')), {
      kind: 'structure', changes: [{ op: 'link', model_field: field, variable: fields.has(field) ? null : variable.id }],
    }))),
  };
}

/** The paired Model's values no variable of `task` initializes. */
function uninitialized(view: TrialView, task: TaskDocument, types: readonly StudioNamedValueType[]): UninitializedStart[] {
  const variables = new Set(task.variables.map(({ id }) => id));
  const linked = new Set((task.links ?? []).flatMap(({ model_field, variable }) => variables.has(variable) ? [model_field] : []));
  return view.resets.filter((reset) => !linked.has(addressKey(reset.address)))
    .map((reset) => {
      const id = addressKey(reset.address);
      const reason = view.problems.find(({ path = [] }) => path[0] === 'links' && path[1] === id)?.reason;
      return { ...startOf(reset), anchor: startAnchor(id), reason, type: typeName(reset.type, types), valueType: reset.type };
    });
}

/** The variables of the task document the step reads, each with its Type and Value fields. */
function variables(view: TrialView, task: TaskDocument, types: readonly StudioNamedValueType[], commit?: Commit,
  runs?: TrialRuns): TaskVariable[] {
  const levels = new Map(view.variables.map(({ id, level }) => [id, level]));
  return task.variables.map((variable, index): TaskVariable => {
    const named = types.find(({ type }) => sameType(type, variable.type));
    const name = typeName(variable.type, types);
    const options = [...named ? [] : [{ value: name, label: name }], ...types.map(({ name }) => ({ value: name, label: name }))];
    const typeChoice = rawField({
      id: `type:${variable.id}`, label: studioName('type'), value: name, type: { kind: 'choice', options },
      changed: changedAt(view, 'task', `/variables/${index}/type`, commit),
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
    const value: FieldModel = projection ? {
      ...trialField(projection, studioName('value'), runs, {
        id: `value:${variable.id}`, kind: expressionKind(variable.value),
        changed: changedAt(view, 'task', `/variables/${index}/value`, commit), source: view.task_source ?? undefined,
        commit: commit && ((edit) => commit(studioAccessibleName('set', variable.name), { kind: 'expression', path, edit })),
        selectedMembers: sweep ? runs?.values(sweep) : undefined,
        chooseMember: sweep && runs?.choose ? (member, extend) => runs.choose!(sweep, member, extend) : undefined,
      }),
      anchor: variableAnchor(variable.id),
    } : rawField({ id: `value:${variable.id}`, label: studioName('value'), value: '', type: { kind: 'readOnly' } });
    const countPath = authoredPointer(['task', 'variables', variable.id, 'count']);
    const countProjection = variable.count ? view.fields.find((field) => field.path === countPath) : undefined;
    const count = variable.count ? countProjection ? trialField(countProjection, studioName('count'), runs, {
      id: `count:${variable.id}`, kind: expressionKind(variable.count),
      changed: changedAt(view, 'task', `/variables/${index}/count`, commit),
      commit: commit && ((edit) => commit(studioAccessibleName('set', studioName('count')), { kind: 'expression', path: countPath, edit })),
    }) : rawField({ id: `count:${variable.id}`, label: studioName('count'), value: '', type: { kind: 'readOnly' } }) : undefined;
    return {
      id: variable.id, name: variable.name, type: name, valueType: variable.type, typeChoice, value, count,
      level: levels.get(variable.id) === 2 ? 'step' : 'trial', initializes: initializes(view, task, variable, commit),
    };
  });
}

function variableEdits(task: TaskDocument, commit?: Commit): VariableEdits {
  if (!commit) return {};
  const structure = async (name: string, change: Parameters<Commit>[1] & { kind: 'structure' }, removes?: boolean) =>
    reasonOf(await commit(name, change, removes));
  const named = (id: string) => task.variables.find((variable) => variable.id === id)?.name;
  return {
    add: (name, start) => structure(studioName('addVariable'), { kind: 'structure', changes: [start
      ? { op: 'add', member: 'variable', name, type: start.valueType, initializes: [start.id] }
      : { op: 'add', member: 'variable', name }] }),
    rename: (id, name) => structure(studioAccessibleName('rename', named(id)),
      { kind: 'structure', changes: [{ op: 'rename', member: 'variable', id, name }] }),
    remove: (id) => structure(studioAccessibleName('remove', named(id)),
      { kind: 'structure', changes: [{ op: 'remove', member: 'variable', id }] }, true),
    move: (id, index) => structure(studioAccessibleName('move', named(id)),
      { kind: 'structure', changes: [{ op: 'move', member: 'variable', id, index }] }),
    list: (id, list) => structure(studioAccessibleName('set', studioName('list')), { kind: 'structure', changes: [
      { op: 'set_variable_count', id, count: list ? { kind: 'literal', value: 2 } : null }] }),
  };
}

function useVariables(view: TrialView, commit?: Commit, runs?: TrialRuns) {
  const catalog = useAuthoringCatalog();
  const task = 'variables' in view.task ? view.task : undefined;
  const types = catalog.data?.tasks.value_types ?? [];
  return task && { task, variables: variables(view, task, types, commit, runs), starts: uninitialized(view, task, types) };
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
  return <TaskDocumentBody variables={<TaskVariables variables={shown.variables} starts={shown.starts} selected={selectedVariable(selected)}
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
