import { useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { readStudioTrialAuthoring } from '@/api/studioAuthoring';
import { apiErrorMessage } from '@/api/request';
import { studioAccessibleName, studioDefinitionStatus, studioEvaluateValueSource, studioName, studioReason } from '@/components/activity/studioVocabulary';
import { ArrowUp } from 'lucide-react';
import { Button, IconButton } from '@/components/ui/Button';
import { Checkbox } from '@/components/ui/Checkbox';
import { Select } from '@/components/ui/Select';
import { Table, TableCell, TableHeaderCell } from '@/components/ui/Table';
import { ValueField } from '@/components/values/ValueField';
import { fromFieldView } from '@/components/values/adapters/fieldView';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type {
  JsonValue, ObjectiveTerm, ProjectValueEdit, StudioSavedTrainKey, StudioTrialAuthoring, StudioTrialAuthoringRequest,
} from '@/generated/studioContracts';
import { authoredPointer } from './authoredEdits';
import { Field } from './model/ModelInspector';
import type { ProjectEditController } from './projectEditController';

type Change = NonNullable<StudioTrialAuthoringRequest['change']>;
type Commit = (name: string, change: Change) => Promise<string | undefined>;
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
  return { view: query.data, error: query.error, retry: () => { void query.refetch(); }, commit };
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
        <Select className="min-w-0" aria-label={studioAccessibleName('bind', name)} value={signal ?? ''} disabled={!commit}
          items={[
            { value: '', label: studioDefinitionStatus('unbound') },
            ...view.signals.map(({ name }) => ({ value: name, label: humanizeLabel(name) })),
          ]}
          onValueChange={async (value) => setError(await commit?.('Bind Task Signal', {
            kind: 'binding', input: address, signal: value || null,
          }))} />
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
  return <ValueField model={fromFieldView({
    label, text: projection.text ?? '',
    site: { ...projection.projection, kinds: projection.projection.kinds.filter((kind) => kind !== 'sweep') },
  }, {
    id: term.id + field, kind,
    commit: commit && ((edit) => edit.kind === 'text'
      ? commit('Set ' + label, { kind: 'expression', term: term.id, field, text: edit.text })
      : Promise.resolve(undefined)),
  })} form="cell" />;
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
      <Select className="min-w-0" aria-label={studioName('measureSource')} value={chosen?.id ?? null} disabled={!commit || !sources.length}
        items={sources.map(({ id, name }) => ({ value: id, label: humanizeLabel(name) }))} onValueChange={setSource} />
      <Button disabled={!commit || !chosen} onClick={async () => {
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
      }}>{studioName('addTerm')}</Button>
    </div>
    {error && <p role="alert">{error}</p>}
    <Table aria-label={studioName('objectiveTerms')}>
      <thead><tr>{(['term', 'measure', 'window', 'weight'] as const).map((id) =>
        <TableHeaderCell scope="col" key={id}>{studioName(id)}</TableHeaderCell>)}
        <TableHeaderCell scope="col" />
      </tr></thead>
      <tbody>{terms.map((term, index) => {
        const path = '/terms/' + index;
        const set = (field: keyof ObjectiveTerm, value: JsonValue, name: string) =>
          save(name, write(path + '/' + field, value, has(term, field)));
        return <tr key={term.id} className="align-top" aria-selected={selected === term.id}>
          <TableCell className="py-1">
            <Button variant="ghost" align="start" onClick={() => select(term.id)}>{term.name ?? studioAccessibleName('term', index + 1)}</Button>
            <Checkbox label={studioName('enabled')} checked={term.enabled ?? true} disabled={!commit}
              onCheckedChange={(checked) => { void set('enabled', checked, studioName('enabled')); }} />
          </TableCell>
          <TableCell className="min-w-40 overflow-visible py-1">{expressionControl(view, term, 'measure', commit)}</TableCell>
          <TableCell className="py-1"><Select aria-label={studioName('window')} disabled={!commit}
            value={term.epochs?.length === 1 ? term.epochs[0] : term.epochs?.length ? '@multiple' : ''}
            items={[
              { value: '', label: studioName('wholeTrial') },
              ...term.epochs && term.epochs.length > 1 ? [{ value: '@multiple', label: term.epochs.join(', '), disabled: true }] : [],
              ...view.epochs.map(({ id, name }) => ({ value: id, label: name })),
            ]}
            onValueChange={(epoch) => { void set('epochs', epoch ? [epoch] : [], 'Set Window'); }} /></TableCell>
          <TableCell className="min-w-24 overflow-visible py-1">{expressionControl(view, term, 'weight', commit)}</TableCell>
          <TableCell className="py-1">
            <IconButton label={studioName('moveEarlier')} icon={<ArrowUp aria-hidden="true" />} disabled={!commit || index === 0} onClick={() => {
              const reordered = [...terms];
              [reordered[index - 1], reordered[index]] = [reordered[index], reordered[index - 1]];
              void save(studioName('moveEarlier'), write('/terms', reordered as unknown as JsonValue, true));
            }} />
            <Button variant="ghost" disabled={!commit} onClick={() => {
              void save(studioAccessibleName('remove', studioName('term')), { op: 'remove', path });
            }}>{studioName('remove')}</Button>
          </TableCell>
        </tr>;
      })}</tbody>
    </Table>
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
    <label className="grid gap-1">{studioName('reduction')} <Select aria-label={studioName('reduction')} value={term.reduction ?? 'mean'} disabled={!commit}
      items={([['mean', 'mean'], ['sum', 'sum'], ['last', 'last'], ['max', 'maximum']] as const).map(([value, name]) =>
        ({ value, label: studioName(name) }))}
      onValueChange={(value) => { void save('reduction', value); }} /></label>
    {error && <p role="alert">{error}</p>}
  </div>;
}
