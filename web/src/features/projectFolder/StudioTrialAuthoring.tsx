import clsx from 'clsx';
import { useEffect, useId, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { queryOptions, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  catalogEntries, isExpiredExpressionContext, readStudioTrialAuthoring, readTrialExpressionMenu, useAuthoringCatalog,
} from '@/api/studioAuthoring';
import { shownError } from '@/api/shownErrors';
import {
  studioAccessibleName, studioCountOf, studioDefinitionStatus, studioEvaluateValueSource, studioName,
} from '@/components/activity/studioVocabulary';
import { ChangeMark as ChangedMark } from '@/components/ui/ChangeMark';
import { InlineEdit } from '@/components/ui/InlineEdit';
import { Inspector } from '@/components/ui/Inspector';
import { editedElement } from '@/components/ui/ChoiceMenu';
import { MenuItem } from '@/components/ui/Menu';
import { Tree, type TreeHandle, type TreeNode } from '@/components/ui/Tree';
import {
  evaluateChange, reasonOf, type ChangeMark, type ChangeSource, type FieldModel, type FieldRefusal, type PartMenus,
} from '@/components/values/fieldModel';
import { Drawer, DrawerSection, focusField, type DrawerSections } from '@/features/projectWindow/DrawerForms';
import {
  Table, TableCell, TableFieldCell, TableHeaderCell, TableHeaderRow, TableRow, TableRowHeader, type TableColumn,
} from '@/components/ui/Table';
import { formatMetric } from '@/components/activity/studioPlot';
import { CatalogPicker } from '@/components/ui/CatalogPicker';
import { FieldRow } from '@/components/ui/FieldRow';
import { SettingsForm } from '@/components/values/SettingsForm';
import { changeTip, FieldChange, ValueField } from '@/components/values/ValueField';
import { setIn } from '@/components/values/adapters/jsonSchema';
import { isRecord, rawField } from '@/components/values/adapters/raw';
import { fromFieldView, type FieldViewOptions } from '@/components/values/adapters/fieldView';
import { partMenuKey } from '@/components/values/partMenu';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type {
  FieldKind, JsonValue, ObjectiveTerm, ProjectValueEdit, StudioSavedTrainKey, StudioTrialAuthoring, StudioTrialAuthoringRequest,
  StudioTrialField, TrialTermAggregate,
} from '@/generated/studioContracts';
import { authoredPointer } from './authoredEdits';
import { addressKey } from './model/modelGraph';
import { BatchTrace, TermTrace, TrialTrace } from './task/SignalTrace';
import { epochBoundaries, signalAggregate, signalSteps, type TrialShown } from './task/trialSample';
import { baseOf } from './documentPayload';
import type { ProjectEditController } from './projectEditController';
import { filesRemoval, type ConfirmRunRemoval } from './runRemoval';
import { sendWithStore, type AskStore } from './storeChoice';
import { SweepTargetItem, type SweepTarget } from './StudioTrainAuthoringForm';
import { TaskOutline, type OutlineGroup } from './task/TaskOutline';
import { WindowCell, WindowEditor, type WindowEpochChoice, type WindowHost } from './WindowEditor';

type Change = NonNullable<StudioTrialAuthoringRequest['change']>;
type Relation = Extract<Change, { kind: 'relate' }>;
/** The scope a change is made in. */
export type TrialChangeScope = (change: Change) => StudioTrialAuthoringRequest['scope'] | undefined;
/**
 * Applies a change; `removes` marks one that takes runs out of the project, which the run confirmation guards. A
 * change is stored with whatever Problems it introduces, which the documents' fields mark; it answers with a reason
 * only when the service refuses the write.
 */
export type Commit = (name: string, change: Change, removes?: boolean) => Promise<FieldRefusal | undefined>;
/** A step's Trial authoring, with what the changed values of its Task and Objective documents differ from. */
export type TrialView = StudioTrialAuthoring & {
  readonly changedFrom: { readonly task?: ChangeSource; readonly objective?: ChangeSource };
  /** The menus of the parts of the expression at a field's path, which the service sends as each opens. */
  menus(field: string): PartMenus;
};

const has = (record: object, key: string) => Object.prototype.hasOwnProperty.call(record, key);

function write(path: string, value: JsonValue, present: boolean): ProjectValueEdit {
  return { op: present ? 'replace' : 'add', path, value };
}

/**
 * The changed mark of the value at `path` in the step's Task or Objective document, when it differs from the
 * document's base; its Reset gives it the base's value.
 */
export function changedAt(view: TrialView, document: 'task' | 'objective', path: string, commit?: Commit): ChangeMark | undefined {
  const from = view.changedFrom[document];
  if (!from || !(document === 'task' ? view.task_changed : view.objective_changed)?.includes(path)) return undefined;
  return { from, reset: commit && (() => commit(studioName('reset'), { kind: 'reset', document, paths: [path] })) };
}

/** The kind a stored expression reads as, which its field's mark shows. */
export function expressionKind(value: JsonValue | undefined): FieldKind {
  const kind = isRecord(value) ? value.kind : undefined;
  return kind === 'name' ? 'link' : kind === 'call' || kind === 'unresolved' ? 'formula' : kind === 'draw' ? 'draw' : 'value';
}

/** A field the service projects in a step's Task or Objective, marked with its first stored Problem. */
export function trialField(view: TrialView, projection: StudioTrialField, label: string, runs: TrialRuns | undefined,
  options: FieldViewOptions): FieldModel {
  const [problem] = projection.problems ?? [];
  return {
    ...fromFieldView({ ...projection, label, effective: runs?.selected ? projection.effective : null }, options),
    reason: problem?.reason, span: problem?.span ?? undefined, menus: view.menus(projection.path),
  };
}

/** The read of a run's Trial authoring at a revision. */
const trialAuthoringQuery = (project: string, revision: string, run: StudioSavedTrainKey | undefined) => queryOptions({
  queryKey: ['trial-authoring', project, revision, run] as const,
  queryFn: ({ signal }) => readStudioTrialAuthoring(project, { revision, run: run! }, signal),
});

export function useTrialAuthoring(
  editor: ProjectEditController, project: string, revision: string, run: StudioSavedTrainKey | undefined,
  enabled: boolean, editable: boolean, scope: TrialChangeScope, confirmRemoval: ConfirmRunRemoval, askStore: AskStore,
) {
  const queryClient = useQueryClient();
  const query = useQuery({
    ...trialAuthoringQuery(project, revision, run),
    enabled: enabled && !!run,
    placeholderData: (previous, previousQuery) => previousQuery?.queryKey[1] === project &&
      JSON.stringify(previousQuery.queryKey[3]) === JSON.stringify(run) ? previous : undefined,
  });
  const { project: authored } = useSyncExternalStore(editor.subscribe, editor.getSnapshot).saved;
  const data = query.data;
  /** An Evaluate step's documents change the values it reads; another step's differ from their documents' bases. */
  const view: TrialView | undefined = useMemo(() => {
    if (!data || !run) return undefined;
    const from = (document: string | null): ChangeSource | undefined => {
      if (authored.steps?.[run.step]?.kind === 'evaluate') return { kind: 'evaluate', step: run.step };
      const base = baseOf(authored.documents, document);
      return base === undefined ? undefined : { kind: 'base', document: base };
    };
    /** Reads a menu of this view; once the service no longer holds it, reads the view again and asks that one. */
    const menuKey = (context: string, field: string) => [project, context, field];
    const menus = (field: string): PartMenus => ({
      kind: 'served', key: menuKey(data.expression_context, field),
      read: async (part, signal) => {
        const ask = (expression_context: string) =>
          readTrialExpressionMenu(project, { revision: data.revision, run, expression_context, field, part }, signal);
        try {
          return await ask(data.expression_context);
        } catch (error) {
          if (!isExpiredExpressionContext(error)) throw error;
          const fresh = await queryClient.fetchQuery({ ...trialAuthoringQuery(project, data.revision, run), staleTime: 0 });
          const site = await ask(fresh.expression_context);
          queryClient.setQueryData(partMenuKey(menuKey(fresh.expression_context, field), part), site);
          return site;
        }
      },
    });
    return { ...data, changedFrom: { task: from(data.task_document), objective: from(data.objective_document) }, menus };
  }, [data, authored, run, project, queryClient]);
  /** Stores a change, answering with what the service made of it, or why the write was refused. */
  const send = editable && run && query.data ? (name: string, change: Change, removes?: boolean) => {
    const committedFrom = document.activeElement;
    return editor.command(async (saved) => {
      const request = { revision: saved.revision, run, change, scope: scope(change) };
      // An edit that reaches beyond the selection waits here, holding later commands, until a store is chosen.
      const result = await sendWithStore(askStore, () => editedElement(committedFrom), () => editor.flush(),
        (store_choice) => readStudioTrialAuthoring(saved.project_id, { ...request, store_choice }, editor.signal),
        (answer) => answer.store_choices);
      if (!result) return undefined;
      if (result.files.length || result.created?.length) {
        if (removes && !await confirmRemoval(filesRemoval(saved.project_id, saved.revision, result.files, editor.signal))) return undefined;
        const documents = [result.model, result.task_document, result.objective_document];
        const stored = editor.storeEdit(name, result, {
          reviewedRevision: saved.revision,
          dependencies: [
            { kind: 'step', id: run.step },
            ...documents.flatMap((id) => id ? [{ kind: 'document' as const, id }] : []),
          ],
        });
        await editor.flush();
        const settled = await stored;
        if (settled.status === 'refused') return shownError(settled.error);
        // The commit settles once the view of what it stored is read, so its field never draws the value it replaced.
        await queryClient.prefetchQuery(trialAuthoringQuery(project, editor.getSnapshot().saved.revision, run));
      }
      return result;
    }).catch(shownError);
  } : undefined;
  const commit: Commit | undefined = send && (async (name, change, removes) => {
    const answer = await send(name, change, removes);
    return typeof answer === 'string' ? answer : undefined;
  });
  /** Writes the term on `first`, or relating it to `second`; answers with the new term's id, or why it was refused. */
  const relate = send && (async (first: Relation['first'], second?: Relation['second']) => {
    const answer = await send(studioName('relate'), { kind: 'relate', first, ...second && { second } });
    return typeof answer === 'string' ? { refused: answer } : answer?.related_term ? { term: answer.related_term } : undefined;
  });
  return { view, error: query.error, retry: () => { void query.refetch(); }, commit, relate };
}

export function TrialTaskForm({ view, commit }: { view: TrialView; commit?: Commit }) {
  const [error, setError] = useState<string>();
  const catalog = useAuthoringCatalog();
  const typeLabel = useId();
  const typeChangeId = useId();
  const typeChange = changedAt(view, 'task', authoredPointer(['type']), commit);
  const task = 'type' in view.task ? view.task : undefined;
  const params = task?.params ?? {};
  const type = view.task_types.find((item) => item.type === task?.type);
  const packages = new Map(catalogEntries(catalog.data, 'python_tasks').map((entry) => [entry.key, entry.package]));
  const source = view.task_source && <span className="text-xs text-shell-muted">{studioEvaluateValueSource(view.task_source)}</span>;
  const edit = (name: string, edits: ProjectValueEdit[]) => commit && commit(studioAccessibleName('set', name), { kind: 'task', edits });
  return <div className="grid max-w-2xl gap-4 text-sm">
    {!task && source}
    {task && <section className="grid gap-1" aria-label={studioName('parameters')}>
      <FieldRow label={studioName('task')} labelId={typeLabel}>
        <CatalogPicker aria-labelledby={typeLabel} aria-describedby={typeChange ? typeChangeId : undefined} className="w-full"
          trailing={typeChange && <FieldChange change={typeChange} id={typeChangeId}
            onReset={() => { void typeChange.reset!().then((refusal) => setError(reasonOf(refusal))); }} />}
          disabled={!commit} value={task.type}
          entries={view.task_types.map((item) => ({ key: item.type, label: item.label, package: packages.get(item.type) }))}
          onChoose={async (key) => setError(reasonOf(await edit(studioName('task'), [
            write(authoredPointer(['type']), key, true), write(authoredPointer(['params']), {}, has(task, 'params')),
          ])))} />
      </FieldRow>
      {type && <SettingsForm schema={type.parameter_schema} value={params}
        origin={(path) => {
          const origin = view.task_field_sources?.[path[0]] ?? view.task_source;
          return origin?.kind === 'trained' && !changedAt(view, 'task', authoredPointer(['params', ...path]))
            ? studioEvaluateValueSource(origin) : undefined;
        }}
        changed={(path) => changedAt(view, 'task', authoredPointer(['params', ...path]), commit) ??
          (path.length === 1 ? evaluateChange(view.task_field_sources?.[path[0]] ?? view.task_source ?? undefined) : undefined)}
        write={commit && (async (path, value) => {
          const [key, ...rest] = path;
          const property = type.parameter_schema.properties;
          const name = isRecord(property) && isRecord(property[key]) && typeof property[key].title === 'string'
            ? property[key].title : humanizeLabel(key);
          const edits = [write(authoredPointer(['params', key]), setIn(params[key], rest, value), has(params, key))];
          return edit(name, has(task, 'params') ? edits : [write(authoredPointer(['params']), {}, false), ...edits]);
        })} />}
    </section>}
    {error && <p role="alert">{error}</p>}
  </div>;
}

/** The first problem of each element of the task, by its group and id; a link's is its variable's. */
function outlineReasons(view: StudioTrialAuthoring) {
  const links = 'variables' in view.task ? view.task.links ?? [] : [];
  const reasons = new Map<string, string>();
  for (const { path = [], reason } of view.problems) {
    const [section, id] = path.map(String);
    const linked = section === 'links' ? links.find(({ model_field }) => model_field === id)?.variable : undefined;
    const key = linked ? `variables:${linked}`
      : section === 'variables' || section === 'epochs' || section === 'signals' ? `${section}:${id}` : undefined;
    if (key && !reasons.has(key)) reasons.set(key, reason);
  }
  return reasons;
}

const outlineKinds = { variables: 'variable', epochs: 'epoch', signals: 'signal', modelInputs: 'input' } as const;

/**
 * The drawer's Outline of the Task: a task document's variables, epochs and signals, or a Python task's signals, and
 * the Model's inputs, each with its status mark. Selecting one selects it (`<kind>:<id>`).
 */
export function TrialTaskOutline({ sections, view, selected, select }: {
  sections: DrawerSections; view: StudioTrialAuthoring; selected?: string; select(id: string | null): void;
}) {
  const reasons = outlineReasons(view);
  const entries = (group: OutlineGroup, items: readonly { id: string; name: string }[]) =>
    items.map(({ id, name }) => ({ id, name, reason: reasons.get(`${group}:${id}`) }));
  const task = 'variables' in view.task ? view.task : undefined;
  const groups = {
    variables: task && entries('variables', task.variables),
    epochs: task && entries('epochs', task.epochs),
    signals: task ? entries('signals', task.signals) : view.signals.map(({ id, name }) => ({ id, name: humanizeLabel(name) })),
    modelInputs: view.inputs.map(({ id, name, address, signal_id }) => {
      const signal = view.signals.find((item) => item.id === signal_id)?.name;
      const reason = view.problems.find(({ model_address }) => model_address && addressKey(model_address) === addressKey(address))?.reason;
      return { id, name, muted: !signal_id, reason, detail: signal === name ? undefined : signal };
    }),
  };
  const [kind, ...rest] = selected?.split(':') ?? [];
  const group = (Object.keys(outlineKinds) as OutlineGroup[]).find((key) => outlineKinds[key] === kind);
  return <TaskOutline sections={sections} groups={groups} selected={group && { group, id: rest.join(':') }}
    select={(chosen, id) => select(`${outlineKinds[chosen]}:${id}`)} />;
}

/**
 * The Inspector of a signal, an epoch or a Model input the drawer or the shelf selects; a signal, with a trial or a
 * batch drawn, shows its plot over that trial or its median and percentile bands over the batch.
 */
export function TrialTaskInspector({ view, commit, selected, shown }: {
  view: TrialView; commit?: Commit; selected?: string; shown?: TrialShown;
}) {
  if (!selected) return null;
  const [kind, ...rest] = selected.split(':');
  const key = rest.join(':');
  if (kind === 'signal') {
    const value = view.signals.find(({ id }) => id === key);
    const record = shown?.kind === 'one' ? shown.record : undefined;
    const steps = record && value && signalSteps(record, value.id);
    const aggregate = shown?.kind === 'all' && value ? signalAggregate(shown.aggregate, value.id) : undefined;
    const period = view.period_seconds;
    return value ? <Inspector name={humanizeLabel(value.name)} type={value.type.shape?.join(' × ') || value.type.base}>
      {period != null && record && steps && <section aria-label={studioName('plot')} className="grid gap-1">
        <TrialTrace label={value.name} steps={steps} duration={steps.length * period} boundaries={epochBoundaries(record)} />
      </section>}
      {period != null && aggregate && <section aria-label={studioName('plot')} className="grid gap-1">
        <BatchTrace label={value.name} aggregate={aggregate} duration={aggregate.presence.length * period} boundaries={[]} />
      </section>}
    </Inspector> : null;
  }
  if (kind === 'epoch') {
    const epoch = view.epochs.find(({ id }) => id === key);
    return epoch ? <Inspector name={epoch.name} type={studioName('epochs')} /> : null;
  }
  if (kind !== 'input') return null;
  const input = view.inputs.find(({ id }) => id === key);
  if (!input) return null;
  return <Inspector name={input.name} type={studioName('modelInputs')}>
    <ValueField key={JSON.stringify([input.id, input.signal_id])} model={rawField({
      id: 'signal', label: studioName('signals'), value: input.signal_id ?? '',
      type: { kind: 'choice', options: [
        { value: '', label: studioDefinitionStatus('unbound') },
        ...view.signals.map(({ id, name }) => ({ value: id, label: humanizeLabel(name) })),
      ] },
      change: commit && ((value) => commit('Bind Task Signal', { kind: 'binding', input_id: input.id, signal_id: String(value) || null })),
    })} />
  </Inspector>;
}

/** A term's fields that are values, by the column that shows them; its Window has its own editor. */
type TermField = 'name' | 'enabled' | 'measure' | 'weight' | 'reduction';
type TermColumn = 'name' | 'measure' | 'window' | 'weight' | 'value' | 'covered' | 'enabled';
const authoredColumns = ['name', 'measure', 'window', 'weight', 'enabled'] as const satisfies readonly TermColumn[];
const scoredColumns = ['name', 'measure', 'window', 'weight', 'value', 'covered', 'enabled'] as const satisfies readonly TermColumn[];
const columnNames = {
  name: 'term', enabled: 'enabled', measure: 'measure', window: 'window', weight: 'weight', value: 'value', covered: 'covered',
} as const;
const columnWidths: Record<TermColumn, TableColumn> = {
  name: { least: '6rem', greatest: '16rem' }, measure: { least: '10rem' }, window: { least: '8rem' }, weight: { least: '7rem' },
  value: { least: '6rem', greatest: '9rem' }, covered: { least: '6rem', greatest: '8rem' },
  enabled: { least: '4.5rem', greatest: '4.5rem' },
};

/**
 * What a finished evaluation reports for each term: `shown` is the trial or trials the clock shows, whose value a term's
 * Value is; `run` is each term over every trial the run recorded, whose share of `trials` it covers is its Covered.
 */
export interface TermScores {
  readonly shown: TrialShown | undefined;
  readonly run: Readonly<Record<string, TrialTermAggregate>>;
  readonly trials: number;
}

function scoreOf(scores: TermScores, term: string) {
  const { shown } = scores;
  const value = shown?.kind === 'one' ? shown.record.terms?.[term]?.value
    : shown?.kind === 'all' ? shown.aggregate.terms?.[term]?.value : undefined;
  const covered = scores.run[term]?.covered;
  return {
    value: value == null ? '' : formatMetric(value),
    covered: covered === undefined ? '' : studioCountOf(Math.round(covered * scores.trials), scores.trials),
  };
}

/** The anchor of a term's Weight, which Add Sweep puts the keyboard on. */
export const weightAnchor = (term: string) => `objective:${term}:weight`;

/** The anchor of a term's Measure, which a term just added puts the keyboard on. */
export const measureAnchor = (term: string) => `objective:${term}:measure`;

/** The runs the Objective is shown for: `selected` says the user chose them, whose swept values are marked as differing
 * from the step's; `values` are the chosen values of each sweep, and `choose` chooses one. */
export interface TrialRuns {
  readonly selected: boolean;
  values(sweep: string): ReadonlySet<string>;
  choose?(sweep: string, value: string, extend: boolean): void;
}

const termPath = (index: number) => '/terms/' + index;

function termFields(view: TrialView, term: ObjectiveTerm, index: number, commit?: Commit, runs?: TrialRuns,
  shown?: TrialShown): Record<TermField, FieldModel> {
  const path = termPath(index);
  const changed = (field: keyof ObjectiveTerm) => changedAt(view, 'objective', path + '/' + field, commit);
  const set = (field: keyof ObjectiveTerm, name: string) => commit && ((value: JsonValue) => commit(
    studioAccessibleName('set', name), { kind: 'objective', edits: [write(path + '/' + field, value, has(term, field))] }));
  const expression = (field: 'measure' | 'weight'): FieldModel => {
    const label = humanizeLabel(field);
    const projection = view.fields.find((site) => site.path === authoredPointer(['objective', 'terms', term.id, field]));
    if (!projection) throw new Error(`The Objective's ${field} of term ${term.id} has no field`);
    const kind = expressionKind(term[field] as JsonValue | undefined);
    const sweep = projection.sweep?.id;
    const model = trialField(view, projection, label, runs, {
      id: term.id + field, kind, changed: changed(field),
      commit: commit && ((edit) => commit('Set ' + label, { kind: 'expression', path: projection.path, edit })),
      selectedMembers: sweep ? runs?.values(sweep) : undefined,
      chooseMember: sweep && runs?.choose ? (value, extend) => runs.choose!(sweep, value, extend) : undefined,
    });
    if (field === 'measure') return { ...model, anchor: measureAnchor(term.id) };
    const weights = shown?.kind === 'one' ? shown.record.terms?.[term.id]?.weights ?? undefined : undefined;
    return { ...model, anchor: weightAnchor(term.id), curve: weights };
  };
  return {
    name: rawField({ id: term.id + 'name', label: studioName('name'), type: { kind: 'string' },
      value: term.name ?? '', change: set('name', studioName('name')), changed: changed('name') }),
    enabled: rawField({ id: term.id + 'enabled', label: studioName('enabled'), type: { kind: 'boolean' },
      value: term.enabled ?? true, change: set('enabled', studioName('enabled')), changed: changed('enabled') }),
    measure: expression('measure'),
    weight: expression('weight'),
    reduction: rawField({ id: term.id + 'reduction', label: studioName('reduction'),
      value: term.reduction ?? 'mean', change: set('reduction', humanizeLabel('reduction')), changed: changed('reduction'), type: { kind: 'choice', options:
        ([['mean', 'mean'], ['sum', 'sum'], ['last', 'last'], ['max', 'maximum']] as const)
          .map(([value, name]) => ({ value, label: studioName(name) })) } }),
  };
}

/** The epochs a window can name, each with the occurrences the nominal trial has of it. */
export function windowEpochs(view: StudioTrialAuthoring): WindowEpochChoice[] {
  return view.epochs.map(({ id, name }) => ({
    id, name, occurrences: (view.epoch_instances ?? []).filter((instance) => instance.id === id).length,
  }));
}

/** A term's Window: its structure stored at the term's `window`, each offset the field the service projects for it. */
function termWindow(view: TrialView, term: ObjectiveTerm, index: number, commit?: Commit, runs?: TrialRuns): WindowHost {
  const path = termPath(index) + '/window';
  return {
    epochs: windowEpochs(view),
    offset: (site) => {
      const pointer = authoredPointer(['objective', 'terms', term.id, 'window', ...site]);
      const projection = view.fields.find((field) => field.path === pointer);
      if (!projection) throw new Error(`The Objective's window offset ${pointer} has no field`);
      const label = studioName(site[2]);
      return trialField(view, projection, label, runs, {
        id: term.id + site.join('-'), kind: expressionKind(valueAt(term.window, site)),
        changed: changedAt(view, 'objective', path + '/' + site.join('/'), commit),
        commit: commit && ((edit) => commit(studioAccessibleName('set', label), { kind: 'expression', path: projection.path, edit })),
      });
    },
    change: commit && ((value) => commit(studioAccessibleName('set', studioName('window')),
      { kind: 'objective', edits: [write(path, value as unknown as JsonValue, has(term, 'window'))] })),
    changed: changedAt(view, 'objective', path, commit),
  };
}

function valueAt(root: unknown, path: readonly (string | number)[]): JsonValue | undefined {
  return path.reduce<unknown>((node, key) => isRecord(node) || Array.isArray(node) ? (node as Record<string | number, unknown>)[key] : undefined, root) as JsonValue | undefined;
}

/** A term, and the field of it whose cell is selected. */
export interface TermSelection {
  term: string;
  field?: string;
}

/** The quantities a term can measure: the Model's outputs, the task's variables and its signals, of a real or integer type. */
function measured(view: StudioTrialAuthoring) {
  return [...view.outputs.map(({ value }) => value), ...view.variables, ...view.signals]
    .filter(({ type }) => type.base === 'real' || type.base === 'integer');
}

/** The term that penalises `chosen`'s squared size. */
function newTerm(chosen: ReturnType<typeof measured>[number]) {
  return {
    id: crypto.randomUUID(), name: humanizeLabel(chosen.name),
    measure: { kind: 'call', function: 'sqnorm', args: [{ kind: 'name', reference: chosen.id }] },
    measure_type: { base: 'real', dimensions: Object.fromEntries(
      Object.entries(chosen.type.dimensions ?? {}).map(([name, exponent]) => [name, exponent * 2]),
    ) },
    weight: { kind: 'literal', value: 1 }, enabled: true,
  } satisfies JsonValue;
}

/**
 * The Objective's terms as a table of Term, Measure, Window, Weight and the on-off check. Add Term at the foot of the
 * Term column lists the quantities a term can measure, and the term added is selected with its Measure focused. A
 * term is renamed by clicking its name, moved by dragging its handle, and removed from its row's menu.
 */
export function TrialObjectiveTable({ view, commit, selected, select, runs, scores }: {
  view: TrialView; commit?: Commit; selected?: TermSelection; select(term: string, field?: string): void; runs?: TrialRuns;
  scores?: TermScores;
}) {
  const columns = scores ? scoredColumns : authoredColumns;
  const terms = view.objective?.terms ?? [];
  const box = useRef<HTMLDivElement>(null);
  const [renaming, setRenaming] = useState<string>();
  const [added, setAdded] = useState<string>();
  useEffect(() => {
    if (!added || !terms.some(({ id }) => id === added)) return;
    const measure = box.current?.querySelector<HTMLInputElement>(`[data-field="${CSS.escape(measureAnchor(added))}"] input`);
    if (measure) focusField(measure);
    setAdded(undefined);
  }, [added, terms]);
  const save = (name: string, edits: ProjectValueEdit[], removes?: boolean) => commit?.(name, { kind: 'objective', edits }, removes);
  const move = commit && ((id: string, index: number) => {
    const from = terms.findIndex((term) => term.id === id);
    const reordered = [...terms];
    reordered.splice(index, 0, ...reordered.splice(from, 1));
    return save(studioAccessibleName('move', studioName('term')), [write('/terms', reordered as unknown as JsonValue, true)]);
  });
  const sources = measured(view);
  const add = async (chosen: (typeof sources)[number]) => {
    const value = newTerm(chosen);
    const edits = terms.length ? [write('/terms/-', value, false)] : [write('/terms', [value], has(view.objective ?? {}, 'terms'))];
    if (await save(studioName('addTerm'), edits)) return;
    select(value.id, 'measure');
    setAdded(value.id);
  };
  return <Table aria-label={studioName('objectiveTerms')} held={1} heldEnd columns={columns.map((id) => columnWidths[id])}
    box={{ ref: box, axis: 'both', className: 'h-full min-h-0 [--fade-inset-top:var(--row-height)]' }}
    moveRow={move}
    addRow={{ label: studioName('addTerm'), disabled: !commit || !sources.length, items: sources.map((source) =>
      <MenuItem key={source.id} onClick={() => { void add(source); }}>{humanizeLabel(source.name)}</MenuItem>) }}>
    <thead className="sticky top-0 z-10"><TableHeaderRow>{columns.map((id) =>
      <TableHeaderCell scope="col" key={id} className={clsx(id === 'enabled' && 'text-center')}>
        {studioName(columnNames[id])}
      </TableHeaderCell>)}
    </TableHeaderRow></thead>
    <tbody>{terms.map((term, index) => {
      const fields = termFields(view, term, index, commit, runs);
      const name = term.name ?? studioAccessibleName('term', index + 1);
      const score = scores && scoreOf(scores, term.id);
      const cell = (field: TermColumn) => {
        const chosen = selected?.term === term.id && selected.field === field;
        return { selected: chosen, onFocusCapture: () => { if (!chosen) select(term.id, field); },
          onPointerDown: () => { if (!chosen) select(term.id, field); } };
      };
      return <TableRow key={term.id} selected={selected?.term === term.id && !selected.field} sortable={move && { id: term.id, label: studioAccessibleName('move', name) }}
        menu={commit && { label: studioAccessibleName('term', name), items: <>
          <MenuItem onClick={() => setRenaming(term.id)}>{studioName('rename')}</MenuItem>
          <MenuItem disabled={index === 0} onClick={() => { void move?.(term.id, index - 1); }}>{studioName('moveEarlier')}</MenuItem>
          <MenuItem disabled={index === terms.length - 1} onClick={() => { void move?.(term.id, index + 1); }}>
            {studioName('moveLater')}
          </MenuItem>
          <MenuItem onClick={() => { void save(studioAccessibleName('remove', studioName('term')), [{ op: 'remove', path: termPath(index) }], true); }}>
            {studioName('remove')}
          </MenuItem>
        </> }}>
        {!commit ? <TableRowHeader {...cell('name')}>{name}</TableRowHeader> : <TableFieldCell header {...cell('name')}>
          <span className="flex h-full min-w-0 items-center">
            <InlineEdit form="cell" value={term.name ?? ''} empty={name} editing={renaming === term.id} className="min-w-0 flex-1"
              aria-label={studioAccessibleName('rename', name)}
              onEditingChange={(editing) => setRenaming(editing ? term.id : undefined)}
              onCommit={(next) => {
                if (!commit || !next.trim()) return false;
                if (next.trim() !== term.name) void save(studioAccessibleName('set', studioName('name')),
                  [write(termPath(index) + '/name', next.trim(), has(term, 'name'))]);
              }} />
            <FieldChange change={fields.name.change} id={`${term.id}-name-change`} />
          </span>
        </TableFieldCell>}
        {commit ? <TableFieldCell {...cell('measure')}><ValueField model={fields.measure} form="cell" /></TableFieldCell>
          : <TableCell {...cell('measure')}>{fields.measure.text}</TableCell>}
        <TableFieldCell {...cell('window')}><WindowCell window={term.window} host={termWindow(view, term, index, commit, runs)} /></TableFieldCell>
        {commit ? <TableFieldCell {...cell('weight')}><ValueField model={fields.weight} form="cell" /></TableFieldCell>
          : <TableCell {...cell('weight')}>{fields.weight.text}</TableCell>}
        {score && <>
          <TableCell className="text-right tabular-nums" {...cell('value')}>{score.value}</TableCell>
          <TableCell className="tabular-nums" {...cell('covered')}>{score.covered}</TableCell>
        </>}
        <TableFieldCell className="text-center" {...cell('enabled')}><ValueField model={fields.enabled} form="cell" /></TableFieldCell>
      </TableRow>;
    })}</tbody>
  </Table>;
}

export function TrialObjectiveOutline({ sections, view, commit, selected, select }: {
  sections: DrawerSections; view: StudioTrialAuthoring; commit?: Commit; selected?: string; select(term: string | null): void;
}) {
  const [error, setError] = useState<string>();
  const tree = useRef<TreeHandle>(null);
  const terms = view.objective?.terms ?? [];
  const save = async (name: string, edit: ProjectValueEdit, removes?: boolean) =>
    setError(reasonOf(await commit?.(name, { kind: 'objective', edits: [edit] }, removes)));
  const nodes: TreeNode[] = terms.map((term, index) => ({
    id: term.id, name: term.name ?? studioAccessibleName('term', index + 1), muted: term.enabled === false,
  }));
  return <Drawer>
    <DrawerSection sections={sections} id="objective.outline" name={studioName('outline')}>
      <Tree ref={tree} aria-label={studioName('objectiveTerms')} nodes={nodes} selected={selected ? [selected] : []}
        onSelectedChange={(ids) => select(ids.at(-1) ?? null)}
        onRename={commit && ((id, name) => {
          const index = terms.findIndex((term) => term.id === id);
          if (index < 0 || !name.trim()) return false;
          void save(studioAccessibleName('set', studioName('name')), write('/terms/' + index + '/name', name.trim(), has(terms[index], 'name')));
        })}
        menu={commit && ((node) => {
          const index = terms.findIndex((term) => term.id === node.id);
          return <>
            <MenuItem onClick={() => tree.current?.rename(node.id)}>{studioName('rename')}</MenuItem>
            <MenuItem disabled={index <= 0} onClick={() => {
              const reordered = [...terms];
              [reordered[index - 1], reordered[index]] = [reordered[index], reordered[index - 1]];
              void save(studioName('moveEarlier'), write('/terms', reordered as unknown as JsonValue, true));
            }}>{studioName('moveEarlier')}</MenuItem>
            <MenuItem onClick={() => { void save(studioAccessibleName('remove', studioName('term')), { op: 'remove', path: '/terms/' + index }, true); }}>
              {studioName('remove')}
            </MenuItem>
          </>;
        })} />
      {error && <p role="alert" className="px-2">{error}</p>}
    </DrawerSection>
  </Drawer>;
}

/**
 * The Inspector of the selected term: a Weight that is a Shape draws its curve in the trial the clock shows, and where
 * what is shown recorded the term's value per step, its plot.
 */
export function TrialObjectiveInspector({ view, commit, selected, runs, shown }: {
  view: TrialView; commit?: Commit; selected?: TermSelection; runs?: TrialRuns; shown?: TrialShown;
}) {
  const windowLabel = useId();
  const index = view.objective?.terms?.findIndex(({ id }) => id === selected?.term) ?? -1;
  const term = view.objective?.terms?.[index];
  if (!term) return null;
  const fields = termFields(view, term, index, commit, runs, shown);
  const windowHost = termWindow(view, term, index, commit, runs);
  const order: (TermField | 'window')[] = ['name', 'measure', 'window', 'weight', 'enabled', 'reduction'];
  const name = term.name ?? studioAccessibleName('term', index + 1);
  const period = view.period_seconds;
  return <Inspector name={name} type={term.name ? studioName('term') : undefined}>
    {shown && period != null && <section aria-label={studioName('plot')} className="grid gap-1 empty:hidden">
      <TermTrace label={name} term={term.id} shown={shown} stepSeconds={period} />
    </section>}
    {order.map((field) => field === 'window'
      ? <FieldRow key="window" label={studioName('window')} labelId={windowLabel} actions={windowHost.changed && <ChangedMark
        tip={changeTip(windowHost.changed.from)} reset={windowHost.changed.reset && (() => { void windowHost.changed!.reset!(); })} />}>
        <WindowEditor window={term.window} host={windowHost} labelId={windowLabel} />
      </FieldRow>
      : <ValueField key={term.id + field + JSON.stringify(fields[field].value ?? fields[field].text)} model={fields[field]} />)}
  </Inspector>;
}

/** The anchor of a task variable's Value, which Add Sweep puts the keyboard on. */
export const variableAnchor = (variable: string) => `task:${variable}:value`;

/** The anchor of a task variable's Initializes, where a problem of its links is revealed. */
export const initializesAnchor = (variable: string) => `task:${variable}:initializes`;

/** The anchor of the row of a Model value no variable initializes, keyed by its address, which a problem reveals. */
export const startAnchor = (address: string) => `task:start:${address}`;

/**
 * The fields of the step's Task and Objective a sweep can start from, as Add Sweep items: the task's variables under
 * Task, and the terms' Weights under Objective. Choosing one goes to its field; the sweep is made by the text it commits.
 */
export function TrialSweepItems({ project, revision, run, onChoose }: {
  project: string; revision: string; run?: StudioSavedTrainKey; onChoose(anchor: string, view: 'task' | 'objective'): void;
}) {
  const { data } = useQuery({ ...trialAuthoringQuery(project, revision, run), enabled: !!run });
  if (!data) return null;
  const sweepable = (path: string) => data.fields.some((field) => field.path === path && field.can_sweep && !field.sweep);
  const variables = 'variables' in data.task ? data.task.variables.filter(({ id }) =>
    sweepable(authoredPointer(['task', 'variables', id, 'value']))) : [];
  const terms = (data.objective?.terms ?? []).flatMap((term, index) =>
    sweepable(authoredPointer(['objective', 'terms', term.id, 'weight'])) ? [{ term, index }] : []);
  const task: SweepTarget = { kind: 'group', label: studioName('task'), members: variables.map(({ id, name }) => ({
    kind: 'field', label: name, anchor: variableAnchor(id),
  })) };
  const objective: SweepTarget = { kind: 'group', label: studioName('objective'), members: terms.map(({ term, index }) => ({
    kind: 'group', label: term.name ?? studioAccessibleName('term', index + 1),
    members: [{ kind: 'field', label: humanizeLabel('weight'), anchor: weightAnchor(term.id) }],
  })) };
  return <>
    {variables.length > 0 && <SweepTargetItem target={task} onChoose={(anchor) => onChoose(anchor, 'task')} />}
    {terms.length > 0 && <SweepTargetItem target={objective} onChoose={(anchor) => onChoose(anchor, 'objective')} />}
  </>;
}
