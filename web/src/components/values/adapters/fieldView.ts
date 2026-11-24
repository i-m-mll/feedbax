import type {
  FieldKind, FieldKindEdit, FieldNameEdit, FieldTextEdit, StudioEvaluateValueSource, StudioFieldView,
  StudioPlanReviewSource,
} from '@/generated/studioContracts';
import { studioEvaluateValueSource, studioName, studioPlanValueSource, studioReason } from '@/components/activity/studioVocabulary';
import { evaluateChange, valueKind, withReset, type ChangeMark, type FieldModel, type FieldRefusal } from '../fieldModel';

export type ProjectedEdit = FieldTextEdit | FieldKindEdit | FieldNameEdit;

/** An exception a field's source names: the runs it picks, and a command that reveals its row in Exceptions. */
export interface ExceptionMention {
  readonly runs: string;
  readonly reveal: () => void;
}

export type ExceptionMentions = (source: StudioPlanReviewSource) => ExceptionMention | undefined;

export interface FieldViewOptions {
  id: string;
  commit?: (edit: ProjectedEdit) => Promise<FieldRefusal | undefined>;
  /** The kind of a value that is not a sweep, where the source knows it and the field has no structure to read it from. */
  kind?: FieldKind;
  /** The field shows the value on a selection of rows rather than the document's. */
  rowSelection?: boolean;
  /** Several rows are selected and the source does not report whether they agree (`mixed`). */
  multipleRows?: boolean;
  selectedMembers?: ReadonlySet<string>;
  chooseMember?: (id: string, extend: boolean) => void;
  planSources?: readonly StudioPlanReviewSource[];
  exceptions?: ExceptionMentions;
  source?: StudioEvaluateValueSource;
  /** The changed mark, when the value is a difference of the shown document. The selected runs' other value takes its place. */
  changed?: ChangeMark;
}

function origin({ planSources, exceptions, source, changed }: FieldViewOptions) {
  if (!planSources?.length) return { text: source?.kind === 'trained' && !changed ? studioEvaluateValueSource(source) : undefined };
  const mentions = planSources.map((item) => item.kind === 'exception' ? exceptions?.(item) : undefined);
  const named = new Map(planSources.flatMap((item, index) => {
    const mention = mentions[index];
    return mention ? [[JSON.stringify([item.step, item.variant, item.name]), mention] as const] : [];
  }));
  return {
    text: [...new Set(planSources.map((item, index) => studioPlanValueSource(item, mentions[index]?.runs)))].join(', '),
    reveal: named.size === 1 ? [...named.values()][0].reveal : undefined,
  };
}

/** A field the server projects (`StudioFieldView`): the server parses its text and reports refusals. */
export function fromFieldView(field: StudioFieldView, options: FieldViewOptions): FieldModel {
  const { label, sweep, unit } = field;
  const { rowSelection = false, multipleRows = false } = options;
  const project = options.commit;
  const commit = withReset(project && ((edit) => edit.kind === 'add_member' || edit.kind === 'remove_member' || edit.kind === 'reset'
    ? Promise.resolve(undefined) : project(edit.kind === 'set' ? { kind: 'text', text: String(edit.value) } : edit)), options.changed);
  const readOnly = !commit;
  const rule = readOnly || !field.can_change_axis ? undefined : commit;
  const kind = sweep ? 'sweep' : field.structure ? valueKind(field.structure.value, field.site.functions) : options.kind ?? 'value';
  const list = sweep?.kind === 'list' ? sweep : undefined;
  const choices = list?.values ?? field.members ?? [];
  const memberText = (value: (typeof choices)[number]['value']) => typeof value === 'number' ? String(value) : value.text;
  const authored = list ? list.values.map(({ value }) => memberText(value)).join(', ') + (list.values.length === 1 ? ',' : '')
    : field.text;
  const effective = multipleRows ? undefined : field.effective ?? undefined;
  const base = { names: [], functions: [], kinds: [], type: { kind: 'number' } } as const;
  const part = (id: string, name: Parameters<typeof studioName>[0], text: string,
    write: (text: string) => Promise<FieldRefusal | undefined>, integer = false): FieldModel => ({
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
  /** The values the selected runs hold, outside a sweep of the field. */
  const held = !sweep && field.mixed ? field.members ?? [] : [];
  const source = origin(options);
  const text = sweep ? authored : field.mixed || rowSelection && multipleRows ? null : rowSelection ? effective ?? authored : authored;
  const change: ChangeMark | undefined = effective === undefined ? options.changed ?? evaluateChange(options.source)
    : { from: rowSelection && !sweep ? { kind: 'value', text: authored } : { kind: 'selected', text: effective } };
  return {
    id: options.id,
    label,
    type: { kind: field.site.site.expected.base === 'integer' ? 'integer' : 'number' },
    unit: unit ?? undefined,
    text,
    kind,
    kinds: field.site.kinds.map((value) => ({
      kind: value,
      // The server says whether the selected rows take a sweep of their own (`can_sweep`) or can end theirs.
      enabled: !(readOnly || rowSelection && value !== kind && value !== 'sweep' && value !== 'value' ||
        value === 'sweep' && !sweep && !field.can_sweep ||
        value === 'value' && !!sweep && (!choices.length || !field.can_change_axis)),
    })),
    keeps: sweep ? field.selected_value ?? choices[0]?.id : undefined,
    members: list ? {
      values: list.values.map(({ id, value }) => ({ id, text: memberText(value), selected: picked.has(id) })),
      choose: options.chooseMember,
    } : held.length ? { values: held.map(({ id, value }) => ({ id, text: memberText(value), selected: false })) } : undefined,
    parts,
    canList: !!sweep && sweep.kind !== 'list' && sweep.can_list && field.can_change_axis,
    origin: [field.conversion, source.text].filter(Boolean).join(' · ') || undefined,
    change,
    revealOrigin: source.reveal,
    structure: field.structure ?? undefined,
    finest: field.site.site.finest,
    names: field.site.site.names ?? [],
    functions: field.site.functions,
    sites: field.site.parts,
    namePaths: field.site.name_paths,
    commit,
  };
}
