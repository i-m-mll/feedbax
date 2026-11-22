import { appTip } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import { Plus } from 'lucide-react';
import { useId, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { catalogEntries, readStudioTrialAuthoring, useAuthoringCatalog } from '@/api/studioAuthoring';
import { apiErrorMessage } from '@/api/request';
import { studioAccessibleName, studioDefinitionStatus, studioEvaluateValueSource, studioName, studioRunReason } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { Inspector } from '@/components/ui/Inspector';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { StatusMark } from '@/components/ui/StatusMark';
import { Tree, type TreeHandle, type TreeNode } from '@/components/ui/Tree';
import type { ChangeMark, FieldModel } from '@/components/values/fieldModel';
import { Drawer, DrawerSection, type DrawerSections } from '@/features/projectWindow/DrawerForms';
import { Table, TableCell, TableHeaderCell } from '@/components/ui/Table';
import { CatalogPicker } from '@/components/ui/CatalogPicker';
import { FieldRow } from '@/components/ui/FieldRow';
import { SettingsForm } from '@/components/values/SettingsForm';
import { Origin, ValueField } from '@/components/values/ValueField';
import { setIn } from '@/components/values/adapters/jsonSchema';
import { isRecord, rawField } from '@/components/values/adapters/raw';
import { fromFieldView } from '@/components/values/adapters/fieldView';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type {
  JsonValue, ObjectiveTerm, ProjectValueEdit, StudioSavedTrainKey, StudioTrialAuthoring, StudioTrialAuthoringRequest,
} from '@/generated/studioContracts';
import { authoredPointer } from './authoredEdits';
import type { ProjectEditController } from './projectEditController';
import { filesRemoval, type ConfirmRunRemoval } from './runRemoval';
import { ScrollFade } from '@/components/ui/ScrollFade';

type Change = NonNullable<StudioTrialAuthoringRequest['change']>;
/** The scope a change is made in. */
export type TrialChangeScope = (change: Change) => StudioTrialAuthoringRequest['scope'] | undefined;
/** Applies a change; `removes` marks one that takes runs out of the project, which the run confirmation guards. */
type Commit = (name: string, change: Change, removes?: boolean) => Promise<string | undefined>;
const has = (record: object, key: string) => Object.prototype.hasOwnProperty.call(record, key);

function write(path: string, value: JsonValue, present: boolean): ProjectValueEdit {
  return { op: present ? 'replace' : 'add', path, value };
}

/**
 * The changed mark of the value at `path` in the step's Task or Objective document, when it differs from the
 * document's base; its Reset gives it the base's value.
 */
function changedAt(view: StudioTrialAuthoring, document: 'task' | 'objective', path: string, text: string,
  commit?: Commit): ChangeMark | undefined {
  if (!(document === 'task' ? view.task_changed : view.objective_changed)?.includes(path)) return undefined;
  return { text, reset: commit && (() => commit(studioName('reset'), { kind: 'reset', document, paths: [path] })) };
}

export function useTrialAuthoring(
  editor: ProjectEditController, project: string, revision: string, run: StudioSavedTrainKey | undefined,
  enabled: boolean, editable: boolean, scope: TrialChangeScope, confirmRemoval: ConfirmRunRemoval,
) {
  const query = useQuery({
    queryKey: ['trial-authoring', project, revision, run],
    queryFn: ({ signal }) => readStudioTrialAuthoring(project, { revision, run: run! }, signal),
    enabled: enabled && !!run,
    placeholderData: (previous, previousQuery) => previousQuery?.queryKey[1] === project &&
      JSON.stringify(previousQuery.queryKey[3]) === JSON.stringify(run) ? previous : undefined,
  });
  const commit: Commit | undefined = editable && run && query.data ? (name, change, removes) => editor.command(async (saved) => {
    const result = await readStudioTrialAuthoring(saved.project_id, {
      revision: saved.revision, run, change, scope: scope(change),
    }, editor.signal);
    if (result.files.length) {
      if (removes && !await confirmRemoval(filesRemoval(saved.project_id, saved.revision, result.files, editor.signal))) return;
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
      if (settled.status === 'refused') return apiErrorMessage(settled.error);
    } else if (result.problems.length) {
      return result.problems[0].reason;
    }
    return undefined;
  }).catch(apiErrorMessage) : undefined;
  return { view: query.data, error: query.error, retry: () => { void query.refetch(); }, commit };
}

/** `changed` is the text of the changed mark: Changed, or in an evaluation the step that changes the value. */
export function TrialTaskForm({ view, commit, changed }: { view: StudioTrialAuthoring; commit?: Commit; changed: string }) {
  const [error, setError] = useState<string>();
  const catalog = useAuthoringCatalog();
  const typeLabel = useId();
  const typeOrigin = useId();
  const typeChange = changedAt(view, 'task', authoredPointer(['type']), changed, commit);
  const task = 'type' in view.task ? view.task : undefined;
  const params = task?.params ?? {};
  const type = view.task_types.find((item) => item.type === task?.type);
  const packages = new Map(catalogEntries(catalog.data, 'python_tasks').map((entry) => [entry.key, entry.package]));
  const source = view.task_source && <span className="text-xs text-shell-muted">{studioEvaluateValueSource(view.task_source)}</span>;
  const edit = (name: string, edits: ProjectValueEdit[]) => commit && commit(studioAccessibleName('set', name), { kind: 'task', edits });
  return <div className="grid max-w-2xl gap-4 text-sm">
    {!task && source}
    {task && <section className="grid gap-1" aria-label={studioName('parameters')}>
      <FieldRow label={studioName('task')} labelId={typeLabel} below={typeChange && <Origin id={typeOrigin} text={typeChange.text}
        reset={typeChange.reset && (() => { void typeChange.reset!().then(setError); })} />}>
        <CatalogPicker aria-labelledby={typeLabel} aria-describedby={typeChange ? typeOrigin : undefined} className="w-full"
          disabled={!commit} value={task.type}
          entries={view.task_types.map((item) => ({ key: item.type, label: item.label, package: packages.get(item.type) }))}
          onChoose={async (key) => setError(await edit(studioName('task'), [
            write(authoredPointer(['type']), key, true), write(authoredPointer(['params']), {}, has(task, 'params')),
          ]))} />
      </FieldRow>
      {type && <SettingsForm schema={type.parameter_schema} value={params}
        origin={([key]) => {
          const origin = view.task_field_sources?.[key] ?? view.task_source;
          return origin ? studioEvaluateValueSource(origin) : undefined;
        }}
        changed={(path) => changedAt(view, 'task', authoredPointer(['params', ...path]), changed, commit)}
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

/** The Task's elements by kind, as the drawer's sections: what the task produces and what the Model reads. */
function taskGroups(view: StudioTrialAuthoring) {
  const signal = (name: string) => humanizeLabel(name);
  return [
    { id: 'variables', nodes: view.variables.map(({ id, name }) => ({ id: 'variable:' + id, name: humanizeLabel(name) })) },
    { id: 'signals', nodes: view.signals.map(({ id, name }) => ({ id: 'signal:' + id, name: signal(name) })) },
    { id: 'epochs', nodes: view.epochs.map(({ id, name }) => ({ id: 'epoch:' + id, name })) },
    { id: 'perTrial', nodes: view.resets.flatMap(({ address }) => address.element.kind === 'field'
      ? [{ id: 'reset:' + JSON.stringify(address), name: address.element.path.map(String).join(' · ') }] : []) },
    { id: 'modelInputs', nodes: view.inputs.map(({ name, signal: bound }) => ({
      id: 'input:' + name, name, muted: !bound,
    })) },
  ] as const satisfies readonly { id: Parameters<typeof studioName>[0]; nodes: readonly TreeNode[] }[];
}

export function TrialTaskOutline({ sections, view, selected, select }: {
  sections: DrawerSections; view: StudioTrialAuthoring; selected?: string; select(id: string | null): void;
}) {
  return <Drawer>
    {taskGroups(view).map(({ id, nodes }) => nodes.length > 0 && <DrawerSection key={id} sections={sections} id={`task.${id}`}
      name={studioName(id)}>
      <Tree aria-label={studioName(id)} nodes={nodes} selected={selected && nodes.some((node) => node.id === selected) ? [selected] : []}
        onSelectedChange={(ids) => select(ids.at(-1) ?? null)} />
    </DrawerSection>)}
  </Drawer>;
}

export function TrialTaskInspector({ view, commit, selected }: { view: StudioTrialAuthoring; commit?: Commit; selected?: string }) {
  if (!selected) return null;
  const [kind, ...rest] = selected.split(':');
  const key = rest.join(':');
  if (kind === 'variable' || kind === 'signal') {
    const value = (kind === 'variable' ? view.variables : view.signals).find(({ id }) => id === key);
    return value ? <Inspector name={humanizeLabel(value.name)} type={value.type.shape?.join(' × ') || value.type.base} /> : null;
  }
  if (kind === 'epoch') {
    const epoch = view.epochs.find(({ id }) => id === key);
    return epoch ? <Inspector name={epoch.name} type={studioName('epochs')} /> : null;
  }
  if (kind === 'reset') {
    const reset = view.resets.find(({ address }) => JSON.stringify(address) === key);
    return reset && reset.address.element.kind === 'field' ? <Inspector name={reset.address.element.path.map(String).join(' · ')}
      type={reset.variable ?? studioDefinitionStatus('unlinked')} /> : null;
  }
  const input = view.inputs.find(({ name }) => name === key);
  if (!input) return null;
  return <Inspector name={input.name} type={studioName('modelInputs')}>
    <ValueField key={JSON.stringify([input.name, input.signal])} model={rawField({
      id: 'signal', label: studioName('signals'), value: input.signal ?? '',
      type: { kind: 'choice', options: [
        { value: '', label: studioDefinitionStatus('unbound') },
        ...view.signals.map(({ name }) => ({ value: name, label: humanizeLabel(name) })),
      ] },
      change: commit && ((value) => commit('Bind Task Signal', { kind: 'binding', input: input.address, signal: String(value) || null })),
    })} />
  </Inspector>;
}

/** A term's fields, by the column that shows them. */
type TermField = 'name' | 'enabled' | 'measure' | 'window' | 'weight' | 'reduction';
const columns = ['name', 'measure', 'window', 'weight', 'enabled'] as const satisfies readonly TermField[];
const columnNames = { name: 'term', enabled: 'enabled', measure: 'measure', window: 'window', weight: 'weight' } as const;

function termFields(view: StudioTrialAuthoring, term: ObjectiveTerm, index: number, commit?: Commit): Record<TermField, FieldModel> {
  const path = '/terms/' + index;
  const changed = (field: keyof ObjectiveTerm) => changedAt(view, 'objective', path + '/' + field, studioName('changed'), commit);
  const set = (field: keyof ObjectiveTerm, name: string) => commit && ((value: JsonValue) => commit(
    studioAccessibleName('set', name), { kind: 'objective', edits: [write(path + '/' + field, value, has(term, field))] }));
  const expression = (field: 'measure' | 'weight'): FieldModel => {
    const label = humanizeLabel(field);
    const projection = view.fields.find((site) => site.path === authoredPointer(['objective', 'terms', term.id, field]));
    if (!projection) return rawField({ id: term.id + field, label, type: { kind: 'readOnly' }, value: '—' });
    const value = term[field];
    const kind = value?.kind === 'name' ? 'link' : value?.kind === 'call' ? 'formula' : value?.kind === 'draw' ? 'draw' : 'value';
    return fromFieldView({
      label, text: projection.text ?? '',
      site: { ...projection.projection, kinds: projection.projection.kinds.filter((kind) => kind !== 'sweep') },
    }, {
      id: term.id + field, kind, changed: changed(field),
      commit: commit && ((edit) => edit.kind === 'text'
        ? commit('Set ' + label, { kind: 'expression', term: term.id, field, text: edit.text })
        : Promise.resolve(undefined)),
    });
  };
  const epochs = term.epochs ?? [];
  return {
    name: rawField({ id: term.id + 'name', label: studioName('name'), type: { kind: 'string' },
      value: term.name ?? '', change: set('name', studioName('name')), changed: changed('name') }),
    enabled: rawField({ id: term.id + 'enabled', label: studioName('enabled'), type: { kind: 'boolean' },
      value: term.enabled ?? true, change: set('enabled', studioName('enabled')), changed: changed('enabled') }),
    measure: expression('measure'),
    window: rawField({ id: term.id + 'window', label: studioName('window'), changed: changed('epochs'),
      value: epochs.length === 1 ? epochs[0] : epochs.length ? epochs.join(', ') : '',
      type: { kind: 'choice', options: [
        { value: '', label: studioName('wholeTrial') },
        ...epochs.length > 1 ? [{ value: epochs.join(', '), label: epochs.join(', ') }] : [],
        ...view.epochs.map(({ id, name }) => ({ value: id, label: name })),
      ] },
      change: commit && ((epoch) => epoch === epochs.join(', ') && epochs.length > 1 ? undefined
        : commit('Set Window', { kind: 'objective', edits: [write(path + '/epochs', epoch ? [epoch] : [], has(term, 'epochs'))] })) }),
    weight: expression('weight'),
    reduction: rawField({ id: term.id + 'reduction', label: studioName('reduction'),
      value: term.reduction ?? 'mean', change: set('reduction', humanizeLabel('reduction')), changed: changed('reduction'), type: { kind: 'choice', options:
        ([['mean', 'mean'], ['sum', 'sum'], ['last', 'last'], ['max', 'maximum']] as const)
          .map(([value, name]) => ({ value, label: studioName(name) })) } }),
  };
}

/** A term, and the field of it whose cell is selected. */
export interface TermSelection {
  term: string;
  field?: string;
}

/** Add Term: a plus that lists every quantity a term can measure; the term added penalises its squared size. */
export function TrialObjectiveAdd({ view, commit, select }: {
  view: StudioTrialAuthoring; commit?: Commit; select(term: string): void;
}) {
  const [refusal, setRefusal] = useState<string>();
  const terms = view.objective?.terms ?? [];
  const sources = [...view.outputs.map(({ value }) => value), ...view.variables, ...view.signals]
    .filter(({ type }) => type.base === 'real' || type.base === 'integer');
  const add = async (chosen: (typeof sources)[number]) => {
    if (!commit) return;
    const id = crypto.randomUUID();
    const value = {
      id, name: humanizeLabel(chosen.name),
      measure: { kind: 'call', function: 'sqnorm', args: [{ kind: 'name', reference: chosen.id }] },
      measure_type: { base: 'real', dimensions: Object.fromEntries(
        Object.entries(chosen.type.dimensions ?? {}).map(([name, exponent]) => [name, exponent * 2]),
      ) },
      weight: { kind: 'literal', value: 1 }, epochs: [], enabled: true,
    } satisfies JsonValue;
    const edits = terms.length
      ? [write('/terms/-', value, false)]
      : [write('/terms', [value], has(view.objective ?? {}, 'terms'))];
    const reason = await commit(studioName('addTerm'), { kind: 'objective', edits });
    setRefusal(reason);
    if (!reason) select(id);
  };
  return <>
    <Menu>
      <MenuTrigger render={<IconButton variant="ghost" label={studioName('addTerm')} icon={<Plus aria-hidden="true" />}
        disabled={!commit || !sources.length} aria-description={refusal} />} />
      <MenuContent aria-label={studioName('addTerm')}>
        {sources.map((source) => <MenuItem key={source.id} onClick={() => { void add(source); }}>
          {humanizeLabel(source.name)}
        </MenuItem>)}
      </MenuContent>
    </Menu>
    {refusal && <span role="alert" className="inline-flex" {...appTip(studioRunReason({ reason: refusal }))}>
      <StatusMark status="cannot_run" label={refusal} />
    </span>}
  </>;
}

export function TrialObjectiveTable({ view, commit, selected, select }: {
  view: StudioTrialAuthoring; commit?: Commit; selected?: TermSelection; select(term: string, field?: string): void;
}) {
  const terms = view.objective?.terms ?? [];
  return <ScrollFade axis="both" className="h-full min-h-0 [--fade-inset-top:var(--row-height)]">
    <Table aria-label={studioName('objectiveTerms')}>
      <thead className="sticky top-0 z-10"><tr>{columns.map((id) =>
        <TableHeaderCell scope="col" key={id} className={clsx(id === 'enabled' && 'w-0 text-center')}>
          {studioName(columnNames[id])}
        </TableHeaderCell>)}
      </tr></thead>
      <tbody>{terms.map((term, index) => {
        const fields = termFields(view, term, index, commit);
        return <tr key={term.id} className="align-top">
          {columns.map((field) => {
            const chosenCell = selected?.term === term.id && selected.field === field;
            return <TableCell key={field} data-selected={chosenCell || undefined}
              className={clsx('overflow-visible py-1', field === 'measure' && 'min-w-40', chosenCell && 'bg-shell-row-selected')}
              onFocusCapture={() => { if (!chosenCell) select(term.id, field); }}
              onPointerDown={() => { if (!chosenCell) select(term.id, field); }}>
              <ValueField model={fields[field]} form="cell" />
            </TableCell>;
          })}
        </tr>;
      })}</tbody>
    </Table>
  </ScrollFade>;
}

export function TrialObjectiveOutline({ sections, view, commit, selected, select }: {
  sections: DrawerSections; view: StudioTrialAuthoring; commit?: Commit; selected?: string; select(term: string | null): void;
}) {
  const [error, setError] = useState<string>();
  const tree = useRef<TreeHandle>(null);
  const terms = view.objective?.terms ?? [];
  const save = async (name: string, edit: ProjectValueEdit, removes?: boolean) =>
    setError(await commit?.(name, { kind: 'objective', edits: [edit] }, removes));
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

export function TrialObjectiveInspector({ view, commit, selected }: {
  view: StudioTrialAuthoring; commit?: Commit; selected?: TermSelection;
}) {
  const index = view.objective?.terms?.findIndex(({ id }) => id === selected?.term) ?? -1;
  const term = view.objective?.terms?.[index];
  if (!term) return null;
  const fields = termFields(view, term, index, commit);
  const first = (Object.keys(fields) as TermField[]).find((field) => field === selected?.field);
  const order = [...first ? [first] : [], ...(Object.keys(fields) as TermField[]).filter((field) => field !== first)];
  return <Inspector name={term.name ?? studioAccessibleName('term', index + 1)} type={term.name ? studioName('term') : undefined}>
    {order.map((field) => <ValueField key={term.id + field + JSON.stringify(fields[field].value ?? fields[field].text)}
      model={fields[field]} />)}
  </Inspector>;
}
