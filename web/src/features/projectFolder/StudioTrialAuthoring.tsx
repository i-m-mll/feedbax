import { useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { readStudioTrialAuthoring } from '@/api/studioAuthoring';
import { apiErrorMessage } from '@/api/request';
import { studioAccessibleName, studioDefinitionStatus, studioEvaluateValueSource, studioName, studioReason } from '@/components/activity/studioVocabulary';
import { StudioFieldControl } from '@/components/values/StudioFieldControl';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type {
  JsonValue, ObjectiveTerm, ProjectValueEdit, StudioSavedTrainKey, StudioTrialAuthoring, StudioTrialAuthoringRequest,
} from '@/generated/studioContracts';
import { authoredPointer } from './authoredEdits';
import { Field } from './model/ModelInspector';
import type { ProjectEditController } from './projectEditController';

type Change = NonNullable<StudioTrialAuthoringRequest['change']>;
type Commit = (name: string, change: Change) => Promise<string | undefined>;
const control = 'min-w-0 rounded border border-shell-divider bg-shell-chrome px-2 py-1 text-sm';
const button = 'rounded px-2 py-1 text-sm hover:bg-shell-hover disabled:opacity-50';
const has = (record: object, key: string) => Object.prototype.hasOwnProperty.call(record, key);

function parameterValue(value: JsonValue | undefined) {
  return typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean' ? String(value) : '';
}

function write(path: string, value: JsonValue, present: boolean): ProjectValueEdit {
  return { op: present ? 'replace' : 'add', path, value };
}

export function useTrialAuthoring(
  editor: ProjectEditController, project: string, revision: string, run: StudioSavedTrainKey | undefined,
  enabled: boolean, editable: boolean, scope?: StudioTrialAuthoringRequest['scope'],
) {
  const writing = useRef(false);
  const [busy, setBusy] = useState(false);
  const query = useQuery({
    queryKey: ['trial-authoring', project, revision, run],
    queryFn: ({ signal }) => readStudioTrialAuthoring(project, { revision, run: run! }, signal),
    enabled: enabled && !!run, retry: false,
  });
  const commit: Commit | undefined = editable && !busy && run && query.data ? async (name, change) => {
    if (writing.current) return studioReason('saving', studioName('project'));
    writing.current = true;
    setBusy(true);
    try {
      const result = await readStudioTrialAuthoring(project, { revision, run, change, scope }, editor.signal);
      if (result.files.length) {
        const documents = [result.model, result.task_document, result.objective_document];
        editor.editFiles(name, result.files, {
          reviewedRevision: revision,
          dependencies: [
            { kind: 'step', id: run.step },
            ...documents.flatMap((id) => id ? [{ kind: 'document' as const, id }] : []),
          ],
        });
        await editor.flush();
      } else if (result.problems.length) {
        return result.problems[0].reason;
      }
    } catch (error) {
      return apiErrorMessage(error);
    } finally {
      writing.current = false;
      setBusy(false);
    }
  } : undefined;
  return { view: query.data, error: query.error ? apiErrorMessage(query.error) : undefined, commit };
}

export function TrialTaskPicker({ view, commit }: { view: StudioTrialAuthoring; commit?: Commit }) {
  const [error, setError] = useState<string>();
  return <>
    <select className={control} aria-label={studioName('taskDocument')} value={view.task_document ?? ''}
      disabled={!commit || view.tasks.length === 0} onChange={async (event) => {
        const document = event.currentTarget.value;
        if (document) setError(await commit?.(studioAccessibleName('select', studioName('task')), { kind: 'select_task', document }));
      }}>
      {!view.task_document && <option value="">{studioName('inlineTask')}</option>}
      {view.tasks.map(({ document, problems }) => <option key={document} value={document}>
        {document}{problems.length ? ' · ' + problems[0].reason : ''}
      </option>)}
    </select>
    {error && <span role="alert">{error}</span>}
  </>;
}

export function TrialTaskForm({ view, commit }: { view: StudioTrialAuthoring; commit?: Commit }) {
  const [error, setError] = useState<string>();
  const task = view.task;
  const selection = 'params' in task ? task : undefined;
  const type = view.task_types.find((item) => item.type === selection?.type);
  const properties = type?.parameter_schema.properties;
  const params = properties && typeof properties === 'object' && !Array.isArray(properties) ? Object.entries(properties) : [];
  const source = view.task_source && <span className="text-xs text-shell-muted">{studioEvaluateValueSource(view.task_source)}</span>;
  return <div className="grid max-w-2xl gap-4 p-3 text-sm">
    {!type && source}
    {type && <section className="grid gap-2" aria-label={studioName('parameters')}>
      <h3 className="font-medium">{type.label}</h3>
      {params.flatMap(([key, property]) => {
        if (!property || typeof property !== 'object' || Array.isArray(property)) return [];
        const name = typeof property.title === 'string' ? property.title : humanizeLabel(key);
        const value = selection?.params[key] ?? property.default;
        const number = property.type === 'number' || property.type === 'integer';
        const origin = view.task_field_sources?.[key] ?? view.task_source;
        return <Field key={key} label={name} value={parameterValue(value)} type={number ? 'number' : 'text'}
          description={origin ? studioEvaluateValueSource(origin) : undefined}
          parse={(raw) => number ? raw.trim() && Number.isFinite(Number(raw)) ? Number(raw) : undefined : raw}
          commit={commit && (async (value) => setError(await commit('Set ' + name, {
            kind: 'task', edits: [write(authoredPointer(['params', key]), value, !!selection && has(selection.params, key))],
          })))} />;
      })}
    </section>}
    <section aria-label={studioName('signals')}>
      <h3 className="font-medium">{studioName('signals')}</h3>
      <ul>{view.signals.map(({ id, name, type }) => <li key={id}>{humanizeLabel(name)} · {type.shape?.join(' × ') || type.base}</li>)}</ul>
    </section>
    {!!view.epochs.length && <section aria-label={studioName('epochs')}>
      <h3 className="font-medium">{studioName('epochs')}</h3>
      <ol>{view.epochs.map(({ id, name }) => <li key={id}>{name}</li>)}</ol>
    </section>}
    {!!view.resets.length && <section aria-label={studioName('perTrial')}>
      <h3 className="font-medium">{studioName('perTrial')}</h3>
      <ul>{view.resets.map(({ address, variable }) => address.element.kind === 'field' &&
        <li key={JSON.stringify(address)}>{address.element.path.map(String).join(' · ')} · {variable ?? studioDefinitionStatus('unlinked')}</li>)}</ul>
    </section>}
    {!!view.inputs.length && <section className="grid gap-1" aria-label={studioName('modelInputs')}>
      <h3 className="font-medium">{studioName('modelInputs')}</h3>
      {view.inputs.map(({ address, name, signal }) => <label key={name} className="flex items-center gap-2">
        {name}
        <select className={control} aria-label={studioAccessibleName('bind', name)} value={signal ?? ''} disabled={!commit}
          onChange={async (event) => setError(await commit?.('Bind Task Signal', {
            kind: 'binding', input: address, signal: event.currentTarget.value || null,
          }))}>
          <option value="">{studioDefinitionStatus('unbound')}</option>
          {view.signals.map(({ id, name }) => <option key={id} value={name}>{humanizeLabel(name)}</option>)}
        </select>
      </label>)}
    </section>}
    {error && <p role="alert">{error}</p>}
  </div>;
}

function expressionControl(view: StudioTrialAuthoring, term: ObjectiveTerm, field: 'measure' | 'weight', commit?: Commit) {
  const projection = view.fields.find((site) => site.path === authoredPointer(['objective', 'terms', term.id, field]));
  if (!projection) return '—';
  const label = humanizeLabel(field);
  const value = term[field];
  const kind = value?.kind === 'name' ? 'link' : value?.kind === 'call' ? 'formula' : value?.kind === 'draw' ? 'draw' : 'value';
  return <StudioFieldControl valueKind={kind} field={{
    label, text: projection.text ?? '',
    site: { ...projection.projection, kinds: projection.projection.kinds.filter((kind) => kind !== 'sweep') },
  }} commit={commit && ((edit) => edit.kind === 'text'
    ? commit('Set ' + label, { kind: 'expression', term: term.id, field, text: edit.text })
    : Promise.resolve(undefined))} />;
}

export function TrialObjectiveTable({ view, commit, selected, select }: {
  view: StudioTrialAuthoring; commit?: Commit; selected?: string; select(id: string): void;
}) {
  const [error, setError] = useState<string>();
  const [source, setSource] = useState('');
  const terms = view.objective?.terms ?? [];
  const sources = [...view.outputs.map(({ value }) => value), ...view.variables, ...view.signals]
    .filter(({ type }) => type.base === 'real' || type.base === 'integer');
  const chosen = sources.find(({ id }) => id === source) ?? sources[0];
  const save = async (name: string, edit: ProjectValueEdit) =>
    setError(await commit?.(name, { kind: 'objective', edits: [edit] }));
  return <div className="grid gap-3 overflow-auto p-3 text-sm">
    <div className="flex items-center gap-2">
      <select className={control} aria-label={studioName('measureSource')} value={chosen?.id ?? ''} disabled={!commit || !sources.length}
        onChange={(event) => setSource(event.currentTarget.value)}>
        {!sources.length && <option value="" />}
        {sources.map(({ id, name }) => <option key={id} value={id}>{humanizeLabel(name)}</option>)}
      </select>
      <button type="button" className={button} disabled={!commit || !chosen} onClick={async () => {
        if (!commit || !chosen) return;
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
        setError(reason);
        if (!reason) select(id);
      }}>{studioName('addTerm')}</button>
    </div>
    {error && <p role="alert">{error}</p>}
    <table className="w-full text-left" aria-label={studioName('objectiveTerms')}>
      <thead><tr>{(['term', 'measure', 'window', 'weight'] as const).map((id) =>
        <th scope="col" className="border-b border-shell-divider p-2 font-medium" key={id}>{studioName(id)}</th>)}
        <th scope="col" className="border-b border-shell-divider" />
      </tr></thead>
      <tbody>{terms.map((term, index) => {
        const path = '/terms/' + index;
        const set = (field: keyof ObjectiveTerm, value: JsonValue, name: string) =>
          save(name, write(path + '/' + field, value, has(term, field)));
        return <tr key={term.id} className="border-b border-shell-divider align-top" aria-selected={selected === term.id}>
          <td className="p-2">
            <button type="button" className={button} onClick={() => select(term.id)}>{term.name ?? studioAccessibleName('term', index + 1)}</button>
            <label className="flex items-center gap-1"><input type="checkbox" checked={term.enabled ?? true} disabled={!commit}
              onChange={(event) => { void set('enabled', event.currentTarget.checked, studioName('enabled')); }} />{studioName('enabled')}</label>
          </td>
          <td className="min-w-40 p-2">{expressionControl(view, term, 'measure', commit)}</td>
          <td className="p-2"><select className={control} aria-label={studioName('window')} disabled={!commit}
            value={term.epochs?.length === 1 ? term.epochs[0] : term.epochs?.length ? '@multiple' : ''}
            onChange={(event) => { const epoch = event.currentTarget.value; void set('epochs', epoch ? [epoch] : [], 'Set Window'); }}>
            <option value="">{studioName('wholeTrial')}</option>
            {term.epochs && term.epochs.length > 1 && <option value="@multiple" disabled>{term.epochs.join(', ')}</option>}
            {view.epochs.map(({ id, name }) => <option key={id} value={id}>{name}</option>)}
          </select></td>
          <td className="min-w-24 p-2">{expressionControl(view, term, 'weight', commit)}</td>
          <td className="p-2">
            <button type="button" className={button} disabled={!commit || index === 0} aria-label={studioName('moveEarlier')} onClick={() => {
              const reordered = [...terms];
              [reordered[index - 1], reordered[index]] = [reordered[index], reordered[index - 1]];
              void save(studioName('moveEarlier'), write('/terms', reordered as unknown as JsonValue, true));
            }}>↑</button>
            <button type="button" className={button} disabled={!commit} onClick={() => {
              void save(studioAccessibleName('remove', studioName('term')), { op: 'remove', path });
            }}>{studioName('remove')}</button>
          </td>
        </tr>;
      })}</tbody>
    </table>
  </div>;
}

export function TrialObjectiveInspector({ view, commit, selected }: {
  view: StudioTrialAuthoring; commit?: Commit; selected?: string;
}) {
  const [error, setError] = useState<string>();
  const index = view.objective?.terms?.findIndex(({ id }) => id === selected) ?? -1;
  const term = view.objective?.terms?.[index];
  if (!term) return null;
  const save = async (field: 'name' | 'reduction', value: JsonValue) => setError(await commit?.('Set ' + humanizeLabel(field), {
    kind: 'objective', edits: [write('/terms/' + index + '/' + field, value, has(term, field))],
  }));
  return <div className="grid gap-2 p-2 text-sm">
    <Field key={term.id} label={studioName('name')} value={term.name ?? ''} parse={(raw) => raw.trim() || undefined}
      commit={commit && ((value) => { void save('name', value); })} />
    <label>{studioName('reduction')} <select className={control} value={term.reduction ?? 'mean'} disabled={!commit}
      onChange={(event) => { void save('reduction', event.currentTarget.value); }}>
      {([['mean', 'mean'], ['sum', 'sum'], ['last', 'last'], ['max', 'maximum']] as const).map(([value, name]) =>
        <option value={value} key={value}>{studioName(name)}</option>)}
    </select></label>
    {error && <p role="alert">{error}</p>}
  </div>;
}
