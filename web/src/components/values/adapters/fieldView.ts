import type {
  FieldKind, FieldKindEdit, FieldNameEdit, FieldTextEdit, StudioEvaluateValueSource, StudioFieldView,
  StudioPlanReviewSource,
} from '@/generated/studioContracts';
import { studioEvaluateValueSource, studioName, studioPlanValueSource, studioReason } from '@/components/activity/studioVocabulary';
import type { FieldCommit, FieldModel } from '../fieldModel';

export type ProjectedEdit = FieldTextEdit | FieldKindEdit | FieldNameEdit;

export interface FieldViewOptions {
  id: string;
  commit?: (edit: ProjectedEdit) => Promise<string | undefined>;
  /** The kind of a value that is not a sweep, where the source knows it. */
  kind?: FieldKind;
  /** The field shows the value on a selection of rows rather than the document's. */
  rowSelection?: boolean;
  /** Several rows are selected and the source does not report whether they agree (`mixed`). */
  multipleRows?: boolean;
  selectedMembers?: ReadonlySet<string>;
  chooseMember?: (id: string, extend: boolean) => void;
  planSources?: readonly StudioPlanReviewSource[];
  source?: StudioEvaluateValueSource;
}

function origin({ planSources, source }: FieldViewOptions) {
  if (planSources?.length) return [...new Set(planSources.map(studioPlanValueSource))].join(', ');
  return source && studioEvaluateValueSource(source);
}

/** A field the server projects (`StudioFieldView`): the server parses its text and reports refusals. */
export function fromFieldView(field: StudioFieldView, options: FieldViewOptions): FieldModel {
  const { label, sweep, unit } = field;
  const { rowSelection = false, multipleRows = false } = options;
  const project = options.commit;
  const commit: FieldCommit | undefined = project && ((edit) => edit.kind === 'add_member' || edit.kind === 'remove_member'
    ? Promise.resolve(undefined) : project(edit.kind === 'set' ? { kind: 'text', text: String(edit.value) } : edit));
  const readOnly = !commit;
  const rule = readOnly || !field.can_change_axis ? undefined : commit;
  const kind = sweep ? 'sweep' : options.kind ?? 'value';
  const list = sweep?.kind === 'list' ? sweep : undefined;
  const choices = list?.values ?? field.members ?? [];
  const authored = list ? list.values.map(({ value }) => value).join(', ') + (list.values.length === 1 ? ',' : '')
    : field.text;
  const effective = multipleRows ? undefined : field.effective ?? undefined;
  const base = { names: [], functions: [], kinds: [], type: { kind: 'number' } } as const;
  const part = (id: string, name: Parameters<typeof studioName>[0], text: string,
    write: (text: string) => Promise<string | undefined>, integer = false): FieldModel => ({
    ...base, id, label: studioName(name), text, kind: 'value', type: { kind: integer ? 'integer' : 'number' },
    commit: rule && ((edit) => edit.kind === 'text' ? write(edit.text) : Promise.resolve(undefined)),
  });
  const range = sweep?.kind === 'range' ? sweep.rule : undefined;
  const rangeText = (next: typeof range & object) =>
    `${next.scale === 'log' ? 'geomspace' : 'linspace'}(${next.start} ${unit ?? ''}, ${next.stop} ${unit ?? ''}, ${next.count})`;
  const updateRange = (name: 'start' | 'stop' | 'count' | 'scale', value: string) =>
    commit!({ kind: 'text', text: rangeText({ ...range!, [name]: value }) });
  const sampler = sweep?.kind === 'sampler' && typeof sweep.rule.distribution !== 'string' ? sweep.rule : undefined;
  const parts: FieldModel[] = range ? [
    part('start', 'first', String(range.start), (text) => updateRange('start', text)),
    part('stop', 'last', String(range.stop), (text) => updateRange('stop', text)),
    part('count', 'count', String(range.count), (text) => updateRange('count', text), true),
    {
      ...base, id: 'scale', label: studioName('spacing'), text: null, value: range.scale ?? 'linear', kind: 'value',
      type: { kind: 'choice', options: [
        { value: 'linear', label: studioName('linear') }, { value: 'log', label: studioName('logarithmic') },
      ] },
      commit: rule && ((edit) => edit.kind === 'set' ? updateRange('scale', String(edit.value)) : Promise.resolve(undefined)),
    },
  ] : sampler ? [
    part('count', 'count', String(sampler.count),
      (text) => commit!({ kind: 'text', text: field.text.replace(/\d+\)$/, `${text})`) }), true),
    part('seed', 'seed', String(sampler.seed), (text) => {
      const seed = Number(text);
      return text.trim() && Number.isInteger(seed) && seed >= 0 && seed < 2 ** 32
        ? commit!({ kind: 'text', text: field.text, seed }) : Promise.resolve(studioReason('takesInteger', studioName('seed')));
    }, true),
  ] : [];
  const picked = options.selectedMembers ?? new Set<string>();
  return {
    id: options.id,
    label,
    type: { kind: field.site.site.expected.base === 'integer' ? 'integer' : 'number' },
    unit: unit ?? undefined,
    text: sweep ? authored : field.mixed || rowSelection && multipleRows ? null : rowSelection ? effective ?? authored : authored,
    kind,
    kinds: field.site.kinds.map((value) => ({
      kind: value,
      // The server says whether the selected rows take a sweep of their own (`can_sweep`) or can end theirs.
      enabled: !(readOnly || rowSelection && value !== kind && value !== 'sweep' && value !== 'value' ||
        value === 'sweep' && !sweep && !field.can_sweep ||
        value === 'value' && !!sweep && (!choices.length || !field.can_change_axis)),
    })),
    keeps: sweep ? field.selected_value ?? choices[0]?.id : undefined,
    members: list && {
      values: list.values.map(({ id, value }) => ({ id, text: String(value), selected: picked.has(id) })),
      choose: options.chooseMember,
    },
    parts,
    canList: !!sweep && sweep.kind !== 'list' && sweep.can_list && field.can_change_axis,
    differs: effective ?? undefined,
    origin: [field.conversion, origin(options)].filter(Boolean).join(' · ') || undefined,
    names: field.site.site.names ?? [],
    functions: field.site.functions,
    commit,
  };
}
