import type {
  FieldKind, FieldKindEdit, FieldNameEdit, FieldTextEdit, StudioFieldFunction, StudioFieldSite, StudioFieldView,
  StudioPlanReviewSource,
} from '@/generated/studioContracts';
import {
  studioEvaluateValueSource, studioName, studioPlanValueSource, studioReason, type EvaluateOrigin,
} from '@/components/activity/studioVocabulary';
import { evaluateChange, valueKind, withReset, type ChangeMark, type FieldModel, type FieldRefusal } from '../fieldModel';

/** The sites and function lists a field view response states once, which its fields name by key. */
export interface FieldSites {
  readonly sites?: Readonly<Record<string, StudioFieldSite>>;
  readonly function_lists?: Readonly<Record<string, readonly StudioFieldFunction[]>>;
}

/** A field's site with its function list, from the response that emitted the field. */
export function fieldSite(tables: FieldSites, key: string) {
  const site = tables.sites?.[key];
  if (!site) throw new Error(`The response emitted a field whose site ${key} it does not hold`);
  const functions = tables.function_lists?.[site.functions];
  if (!functions) throw new Error(`The response holds site ${key} but not its function list ${site.functions}`);
  return { ...site, functions };
}

export type ProjectedEdit = FieldTextEdit | FieldKindEdit | FieldNameEdit;

/** An exception a field's source names: the runs it picks, and a command that reveals its row in Exceptions. */
export interface ExceptionMention {
  readonly runs: string;
  readonly reveal: () => void;
}

export type ExceptionMentions = (source: StudioPlanReviewSource) => ExceptionMention | undefined;

/** How a field names where the plan says its value comes from. */
export interface SourceNaming {
  readonly exceptions?: ExceptionMentions;
}

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
  /** Where the plan says the value comes from. */
  planSources?: SourceNaming & { readonly sources: readonly StudioPlanReviewSource[] };
  source?: EvaluateOrigin;
  /** The changed mark, when the value is a difference of the shown document. The selected runs' other value takes its place. */
  changed?: ChangeMark;
}

function origin({ planSources, source, changed }: FieldViewOptions) {
  if (!planSources?.sources.length) return { text: source?.kind === 'trained' && !changed ? studioEvaluateValueSource(source) : undefined };
  const { sources, exceptions } = planSources;
  const mentions = sources.map((item) => item.kind === 'exception' ? exceptions?.(item) : undefined);
  const named = new Map(sources.flatMap((item, index) => {
    const mention = mentions[index];
    return mention ? [[JSON.stringify([item.step, item.variant, item.key]), mention] as const] : [];
  }));
  return {
    text: [...new Set(sources.map((item, index) => studioPlanValueSource(item, mentions[index]?.runs)))].join(', '),
    reveal: named.size === 1 ? [...named.values()][0].reveal : undefined,
  };
}

/** A field the server projects (`StudioFieldView`): the server parses its text and reports refusals. */
export function fromFieldView(field: StudioFieldView, tables: FieldSites, options: FieldViewOptions): FieldModel {
  const { label, sweep, unit } = field;
  const site = fieldSite(tables, field.site);
  const { rowSelection = false, multipleRows = false } = options;
  const project = options.commit;
  const commit = withReset(project && ((edit) => edit.kind === 'add_member' || edit.kind === 'remove_member' || edit.kind === 'reset'
    ? Promise.resolve(undefined) : project(edit.kind === 'set' ? { kind: 'text', text: String(edit.value) } : edit)), options.changed);
  const readOnly = !commit;
  const rule = readOnly || !field.can_change_axis ? undefined : commit;
  const kind = sweep ? 'sweep' : field.structure ? valueKind(field.structure.value, site.functions) : options.kind ?? 'value';
  const list = sweep?.kind === 'list' ? sweep : undefined;
  const choices = list?.values ?? field.members ?? [];
  /** A member as written: a number in the field's unit, or its own text. */
  const memberText = (value: (typeof choices)[number]['value']) =>
    typeof value !== 'number' ? value.text : unit ? `${value} ${unit}` : String(value);
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
      ...base, id: 'scale', label: studioName('spacing'), text: null, value: range.scale, kind: 'value',
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
  const converted = field.conversion;
  const text = sweep ? authored : field.mixed || rowSelection && multipleRows ? null : rowSelection ? effective ?? authored : authored;
  const change: ChangeMark | undefined = effective === undefined ? options.changed ?? evaluateChange(options.source)
    : { from: rowSelection && !sweep ? { kind: 'value', text: authored } : { kind: 'selected', text: effective } };
  return {
    id: options.id,
    label,
    type: { kind: site.expected.base === 'integer' ? 'integer' : 'number' },
    unit: unit ?? undefined,
    text,
    kind,
    kinds: site.kinds.map((value) => ({
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
    paired: converted ? `${Number(converted.value.toPrecision(8))} ${converted.unit}` : undefined,
    origin: source.text || undefined,
    change,
    revealOrigin: source.reveal,
    structure: field.structure ?? undefined,
    finest: site.finest,
    drawPer: site.draw_per,
    names: site.names ?? [],
    functions: site.functions,
    menus: { kind: 'listed', parts: site.parts },
    namePaths: site.name_paths,
    commit,
  };
}
