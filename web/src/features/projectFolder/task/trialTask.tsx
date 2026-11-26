import { useAuthoringCatalog } from '@/api/studioAuthoring';
import { studioAccessibleName, studioFormat, studioName, studioValueType } from '@/components/activity/studioVocabulary';
import { reasonOf, type FieldModel } from '@/components/values/fieldModel';
import { rawField } from '@/components/values/adapters/raw';
import {
  ModelAddressSchema, type Link, type StudioNamedValueType, type SetVariableOrigin, type StudioTrialReset, type TaskDocument, type ValueType, type Variable,
} from '@/generated/studioContracts';
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
import type { StudioCommand } from '@/features/projectWindow/studioCommands';
import type { WeightPart } from '@/generated/studioContracts';
import type { EntryTask, ModelStart, SetEntryWeight, TaskSelection, TaskSetVariable, TaskVariable, UnstartedValue, VariableInitializes } from './taskModel';

const sameType = (a: ValueType, b: ValueType) => studioValueType(a) === studioValueType(b);

/** A type's nonzero dimension exponents: an authored type omits the dimensions it does not carry. */
const exponents = ({ dimensions = {} }: ValueType) => Object.fromEntries(Object.entries(dimensions).filter(([, exponent]) => exponent));

/** Whether a Model field of type `field` takes a variable of type `variable`, as the service checks a link. */
const fits = (field: ValueType, variable: ValueType) => {
  const target = field.shape ?? [];
  const source = variable.shape ?? [];
  const offset = target.length - source.length;
  return offset >= 0 && source.every((size, index) => size === 1 || size === target[offset + index])
    && canonicalJsonV2(exponents(field)) === canonicalJsonV2(exponents(variable))
    && (field.space == null || variable.space == null || field.space === variable.space)
    && (field.base === variable.base || field.base === 'real' && variable.base === 'integer');
};

const typeName = (type: ValueType, types: readonly StudioNamedValueType[]) =>
  types.find((named) => sameType(named.type, type))?.name ?? studioValueType(type);

/** The Set entry that adds a variable, with the variable's id there. */
interface Owner {
  readonly entry: EntryTask;
  readonly variable: string;
}

/** Each variable a Set entry adds, by its id in the trials, with its entry as the read names it. */
function owners(view: TrialView, entryOf?: (origin: SetVariableOrigin) => EntryTask | undefined): ReadonlyMap<string, Owner> {
  return new Map(view.variables.flatMap(({ id, set_entry }) => {
    const entry = set_entry && entryOf?.(set_entry);
    return entry ? [[id, { entry, variable: set_entry.variable }] as const] : [];
  }));
}

/** The entry's variables with `id` replaced by `update`'s result, or left out where that is none. */
const updated = (entry: EntryTask, id: string, update: (variable: Variable) => Variable | undefined): Variable[] =>
  entry.variables.flatMap((variable) => {
    if (variable.id !== id) return [variable];
    const next = update(variable);
    return next ? [next] : [];
  });

function startOf(reset: StudioTrialReset): ModelStart {
  const components = (reset.address.composites?.length ?? 0) + 1;
  const name = [...reset.path.slice(0, components), reset.path.slice(components).join(' ')].join(' · ');
  return { id: addressKey(reset.address), name, path: reset.path };
}

/** A linked Model field the paired Model does not have: its component's id and its path. */
function missingStart(id: string): ModelStart {
  const { element } = ModelAddressSchema.parse(JSON.parse(id));
  if (element.kind !== 'field' || element.parent.kind !== 'component') return { id, name: id, path: [id] };
  const [kind, ...members] = element.path;
  const segments = kind === 'state' ? String(members[0]).split('/') : members.map(String);
  return { id, name: `${element.parent.id} · ${segments.join(' ')}`, path: [element.parent.id, ...segments] };
}

/** The Model values `variable` initializes, the choice of those its type fits, and its links' problem and change. */
function initializes(view: TrialView, task: TaskDocument, variable: TaskDocument['variables'][number],
  commit?: Commit, owner?: Owner): VariableInitializes {
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
    toggle: owner ? owner.entry.write && ((field) => owner.entry.write!(studioAccessibleName('set', studioName('initializes')), {
      task_links: [
        ...owner.entry.links.filter(({ model_field }) => model_field !== field),
        ...fields.has(field) ? [] : [{ model_field: field, variable: owner.variable } satisfies Link],
      ],
    })) : commit && (async (field) => reasonOf(await commit(studioAccessibleName('set', studioName('initializes')), {
      kind: 'structure', changes: [{ op: 'link', model_field: field, variable: fields.has(field) ? null : variable.id }],
    }))),
  };
}

/** The paired Model's values no variable of `task` initializes that the Task has a problem with. */
function unstarted(view: TrialView, task: TaskDocument, types: readonly StudioNamedValueType[], commit?: Commit): UnstartedValue[] {
  const variables = new Set(task.variables.map(({ id }) => id));
  const linked = new Set((task.links ?? []).flatMap(({ model_field, variable }) => variables.has(variable) ? [model_field] : []));
  return view.resets.flatMap((reset) => {
    const id = addressKey(reset.address);
    const reason = view.problems.find(({ path = [] }) => path[0] === 'links' && path[1] === id)?.reason;
    if (linked.has(id) || !reason) return [];
    return [{
      ...startOf(reset), anchor: startAnchor(id), reason, type: typeName(reset.type, types), valueType: reset.type,
      variables: task.variables.filter((variable) => fits(reset.type, variable.type)).map(({ id, name }) => ({ id, name })),
      initialize: commit && (async (variable: string) => reasonOf(await commit(studioAccessibleName('set', studioName('initializes')), {
        kind: 'structure', changes: [{ op: 'link', model_field: id, variable }],
      }))),
    }];
  });
}

/** The variables of the task document the step reads, each with its Type and Value fields. */
function variables(view: TrialView, task: TaskDocument, types: readonly StudioNamedValueType[], commit?: Commit,
  runs?: TrialRuns, owned: ReadonlyMap<string, Owner> = new Map()): TaskVariable[] {
  const levels = new Map(view.variables.map(({ id, level }) => [id, level]));
  return task.variables.map((variable, index): TaskVariable => {
    const owner = owned.get(variable.id);
    const named = types.find(({ type }) => sameType(type, variable.type));
    const name = typeName(variable.type, types);
    const options = [...named ? [] : [{ value: name, label: name }], ...types.map(({ name }) => ({ value: name, label: name }))];
    const typeChoice = rawField({
      id: `type:${variable.id}`, label: studioName('type'), value: name, type: { kind: 'choice', options },
      changed: changedAt(view, 'task', `/variables/${index}/type`, commit),
      change: (owner ? owner.entry.write : commit) && (async (chosen) => {
        const type = types.find((item) => item.name === chosen)?.type;
        if (!type) return undefined;
        if (owner) return owner.entry.write!(studioAccessibleName('set', studioName('type')),
          { task_variables: updated(owner.entry, owner.variable, (own) => ({ ...own, type })) });
        return reasonOf(await commit!(studioAccessibleName('set', studioName('type')),
          { kind: 'structure', changes: [{ op: 'set_type', member: 'variable', id: variable.id, type }] }));
      }),
    });
    const path = authoredPointer(['task', 'variables', variable.id, 'value']);
    const projection = view.fields.find((field) => field.path === path);
    const sweep = projection?.sweep?.id;
    const value: FieldModel = projection ? {
      ...trialField(view, projection, studioName('value'), runs, {
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
    const count = variable.count ? countProjection ? trialField(view, countProjection, studioName('count'), runs, {
      id: `count:${variable.id}`, kind: expressionKind(variable.count),
      changed: changedAt(view, 'task', `/variables/${index}/count`, commit),
      commit: commit && ((edit) => commit(studioAccessibleName('set', studioName('count')), { kind: 'expression', path: countPath, edit })),
    }) : rawField({ id: `count:${variable.id}`, label: studioName('count'), value: '', type: { kind: 'readOnly' } }) : undefined;
    return {
      id: variable.id, name: variable.name, type: name, valueType: variable.type, typeChoice, value, count,
      level: levels.get(variable.id) === 2 ? 'step' : 'trial', initializes: initializes(view, task, variable, commit, owner),
      entry: owner?.entry.label,
    };
  });
}

/**
 * The edits of the Variables table: those of a variable a Set entry adds (`owned`) store that entry's variables, and so
 * does Add while one entry is selected (`selected`); the rest edit the Task.
 */
export function variableEdits(task: TaskDocument, commit?: Commit, owned: ReadonlyMap<string, Owner> = new Map(),
  selected?: EntryTask): VariableEdits {
  if (!commit) return {};
  const structure = async (name: string, change: Parameters<Commit>[1] & { kind: 'structure' }, removes?: boolean) =>
    reasonOf(await commit(name, change, removes));
  const named = (id: string) => task.variables.find((variable) => variable.id === id)?.name;
  return {
    add: (name) => selected ? selected.write?.(studioName('addVariable'), { task_variables: [...selected.variables, {
      id: fresh(name, [...task.variables, ...selected.variables]), name, type: { base: 'real', shape: [] }, value: { kind: 'literal', value: 0 },
    }] }) : structure(studioName('addVariable'), { kind: 'structure', changes: [{ op: 'add', member: 'variable', name }] }),
    rename: (id, name) => {
      const owner = owned.get(id);
      return owner ? owner.entry.write?.(studioAccessibleName('rename', named(id)),
        { task_variables: updated(owner.entry, owner.variable, (own) => ({ ...own, name })) })
        : structure(studioAccessibleName('rename', named(id)), { kind: 'structure', changes: [{ op: 'rename', member: 'variable', id, name }] });
    },
    remove: (id) => {
      const owner = owned.get(id);
      return owner ? owner.entry.write?.(studioAccessibleName('remove', named(id)), {
        task_variables: updated(owner.entry, owner.variable, () => undefined),
        task_links: owner.entry.links.filter(({ variable }) => variable !== owner.variable),
      }) : structure(studioAccessibleName('remove', named(id)),
        { kind: 'structure', changes: [{ op: 'remove', member: 'variable', id }] }, true);
    },
    move: (id, index) => structure(studioAccessibleName('move', named(id)),
      { kind: 'structure', changes: [{ op: 'move', member: 'variable', id, index }] }),
    list: (id, list) => structure(studioAccessibleName('set', studioName('list')), { kind: 'structure', changes: [
      { op: 'set_variable_count', id, count: list ? { kind: 'literal', value: 2 } : null }] }),
  };
}

/** `name`, or the first of `name-2`, `name-3` and on that no variable has as its id. */
function fresh(name: string, variables: readonly Variable[]) {
  const taken = new Set(variables.map(({ id }) => id));
  let id = name;
  for (let count = 2; taken.has(id); count += 1) id = `${name}-${count}`;
  return id;
}

/** Where the Task view finds the Set entries the trials read: by a variable's origin, and the one the runs select. */
export interface TaskEntries {
  readonly of: (origin: SetVariableOrigin) => EntryTask | undefined;
  readonly selected?: EntryTask;
}

function useVariables(view: TrialView, commit?: Commit, runs?: TrialRuns, entries?: TaskEntries) {
  const catalog = useAuthoringCatalog();
  const task = 'variables' in view.task ? view.task : undefined;
  const types = catalog.data?.tasks.value_types ?? [];
  const owned = owners(view, entries?.of);
  return task && {
    task, owned, variables: variables(view, task, types, commit, runs, owned), unstarted: unstarted(view, task, types, commit),
  };
}

const entryWeight = (weight: WeightPart | { on?: WeightPart; off?: WeightPart } | null | undefined): SetEntryWeight | undefined =>
  weight == null || typeof weight === 'number' ? weight ?? undefined : { on: weight.on ?? 1, off: weight.off ?? 1 };

/**
 * The Sets the step's trials apply, as the service lists them, each weighed by `weigh` with its axis and limited by
 * the condition the service projects at its field.
 */
function setVariables(view: TrialView, weigh?: (axis: string) => TaskSetVariable['weigh'], commit?: Commit,
  runs?: TrialRuns): TaskSetVariable[] {
  return (view.set_fields ?? []).map(({ axis, set, entries, condition }) => {
    const path = authoredPointer(['set_fields', axis, 'condition']);
    const projection = view.fields.find((field) => field.path === path);
    return {
      axis, set, weigh: weigh?.(axis),
      condition: projection && trialField(view, projection, studioName('condition'), runs, {
        id: `condition:${axis}`, kind: expressionKind(condition ?? undefined),
        commit: commit && ((edit) => commit(studioAccessibleName('set', studioName('condition')), { kind: 'expression', path, edit })),
      }),
      entries: [...entries.filter(({ entry }) => entry !== null), ...entries.filter(({ entry }) => entry === null)]
        .map(({ entry, label, weight }) => ({
          id: entry, label: entry === null ? studioName('none') : label || entry, weight: entryWeight(weight),
        })),
    };
  });
}

/** The selected variable, from the window's selection of a task element (`variable:<id>`). */
export const selectedVariable = (selected: string | undefined): TaskSelection | undefined =>
  selected?.startsWith('variable:') ? { kind: 'variable', id: selected.slice('variable:'.length) } : undefined;

/** The Task view of a task document: Variables, and the grid of epochs and signals. */
export function TrialTaskDocument({ view, commit, runs, entries, selected, select, drawn, toSweep, weigh }: {
  view: TrialView; commit?: Commit; runs?: TrialRuns; selected?: string; select(id: string): void;
  /** The Set entries the trials read, whose variables the table marks and whose edits they store. */
  entries?: TaskEntries;
  /** To Sweep on a Set the trials apply, by its axis. */
  toSweep?(axis: string): StudioCommand;
  /** The edit of a weight of a Set the trials apply, by its axis. */
  weigh?(axis: string): TaskSetVariable['weigh'];
  /** Each variable's value in the trial the clock shows, by variable id, when that trial was drawn. */
  drawn?: Readonly<Record<string, string>>;
}) {
  const shown = useVariables(view, commit, runs, entries);
  if (!shown) return null;
  return <TaskDocumentBody variables={<TaskVariables variables={shown.variables}
    sets={setVariables(view, weigh, commit, runs).map((set) => ({ ...set, toSweep: toSweep?.(set.axis) }))} unstarted={shown.unstarted} selected={selectedVariable(selected)}
    select={(selection) => select(`${selection.kind}:${selection.id}`)} edits={variableEdits(shown.task, commit, shown.owned, entries?.selected)}
    drawn={drawn} />}
    grid={<TrialTaskGrid view={view} commit={commit} runs={runs} selected={selected} select={select} />} />;
}

/** The Inspector of a task document's variable: its value with the value's parts, and its type. */
export function TrialVariableInspector({ view, commit, runs, entries, id }: {
  view: TrialView; commit?: Commit; runs?: TrialRuns; entries?: TaskEntries; id: string;
}) {
  const variable = useVariables(view, commit, runs, entries)?.variables.find((item) => item.id === id);
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
