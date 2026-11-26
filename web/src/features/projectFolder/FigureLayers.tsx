import { ChartLine, Database, LayoutPanelTop, Minus, Palette, RectangleHorizontal, ScatterChart, Spline } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { studioName, studioParameterLabel, studioReason, type StudioNameId } from '@/components/activity/studioVocabulary';
import { InspectorAddRow } from '@/components/ui/Inspector';
import { MenuItem } from '@/components/ui/Menu';
import { Tree } from '@/components/ui/Tree';
import { rawField } from '@/components/values/adapters/raw';
import type { FieldModel, FieldOption } from '@/components/values/fieldModel';
import { ValueField } from '@/components/values/ValueField';
import type {
  ColorEncoding, ContinuousColorScale, DatasetPort, FieldDescriptor, JsonValue, OrdinalColorScale, StoredFigureDocument,
  StoredFigurePanel, StudioAnalyzeFigureValues, StudioFigureColorScaleValues, StudioFigureLayerValues, StoredPointsLayer, StoredRuleLayer, StoredSpanLayer, StoredTrajectoryLayer, VariableBinding,
} from '@/generated/studioContracts';

type Figure = StoredFigureDocument;
type Layer = Figure['layers'][string];
type ColorScale = ContinuousColorScale | OrdinalColorScale;
type Encoding = string | VariableBinding;
type Position = Encoding | number;
type Selection = NonNullable<VariableBinding['select']>[string];
type AddedMark = 'points' | 'rule' | 'span' | 'trajectory';
/** The fields a role's bound data offers, by name. */
export type RoleFields = Readonly<Record<string, FieldDescriptor>>;

/** The fields a dataset port offers a figure: its variables, its selected variables and its coordinates. */
export function offeredFields(port: Pick<DatasetPort, 'variables' | 'selected_variables' | 'coordinates'>): RoleFields {
  return { ...port.variables, ...port.selected_variables, ...port.coordinates };
}

const markGlyphs: Record<Layer['mark'], ReactNode> = {
  line: <ChartLine />, band: <ChartLine />, points: <ScatterChart />, rule: <Minus />, span: <RectangleHorizontal />,
  trajectory: <Spline />,
};
const addedMarks: readonly AddedMark[] = ['points', 'rule', 'span', 'trajectory'];
const axes: readonly FieldOption[] = [{ value: 'x', label: studioName('xAxis') }, { value: 'y', label: studioName('yAxis') }];
const palettes: Record<ColorScale['kind'], readonly string[]> = {
  continuous: ['viridis', 'cividis', 'plasma', 'turbo'],
  ordinal: ['category10', 'plotly', 'dark24'],
};
const member = 'member';

/** An edit of one member of a figure, addressed by its path in the figure. */
export type FigureMemberEdit =
  | { readonly op: 'replace'; readonly path: readonly string[]; readonly value: JsonValue }
  | { readonly op: 'remove'; readonly path: readonly string[] }
  | { readonly op: 'order'; readonly path: readonly string[]; readonly order: readonly string[] };

/** A figure's edit: the label of its Undo entry and the member edits that make it. */
export type FigureEdit = (label: string, edits: readonly FigureMemberEdit[]) => void;

/** The figure's fields whose members are keyed, each written member by member. */
const keyedFields = ['layers', 'layout', 'color_scales'] as const;

/**
 * The member edits that turn `before` into `after`: a keyed collection's members each replaced or removed, and its order
 * written when the members both hold come in another order; any other field replaced or removed whole.
 */
export function figureEdits(before: Figure, after: Figure): FigureMemberEdit[] {
  const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
  const fields = [...new Set([...Object.keys(before), ...Object.keys(after)])] as (keyof Figure)[];
  return fields.flatMap((field): FigureMemberEdit[] => {
    const was = before[field];
    const is = after[field];
    if (same(was, is)) return [];
    if (is === undefined) return [{ op: 'remove', path: [field] }];
    if (!(keyedFields as readonly string[]).includes(field) || was === undefined) {
      return [{ op: 'replace', path: [field], value: is as JsonValue }];
    }
    const old = was as Record<string, JsonValue>;
    const next = is as Record<string, JsonValue>;
    const kept = Object.keys(next).filter((key) => key in old);
    const moved = kept.join('\0') !== Object.keys(old).filter((key) => key in next).join('\0');
    return [
      ...Object.keys(old).filter((key) => !(key in next)).map((key): FigureMemberEdit => ({ op: 'remove', path: [field, key] })),
      ...Object.entries(next).filter(([key, value]) => !same(old[key], value))
        .map(([key, value]): FigureMemberEdit => ({ op: 'replace', path: [field, key], value })),
      ...moved ? [{ op: 'order' as const, path: [field], order: kept }] : [],
    ];
  });
}

/** What the inspector shows the fields of: a layer, a named colour scale, or a panel. */
type Part = { kind: 'layer' | 'scale' | 'panel'; name: string };
const partId = ({ kind, name }: Part) => JSON.stringify([kind, name]);
const partOf = (id: string): Part => { const [kind, name] = JSON.parse(id) as [Part['kind'], string]; return { kind, name }; };

const variableOf = (encoding: Encoding) => typeof encoding === 'string' ? encoding : encoding.variable;
const selectOf = (encoding: Encoding | undefined) => encoding && typeof encoding === 'object' ? encoding.select ?? {} : {};
/** An encoding naming `variable`, keeping the selection a bound encoding makes. */
const renamed = (encoding: Encoding | undefined, variable: string): Encoding =>
  encoding && typeof encoding === 'object' ? { ...encoding, variable } : variable;
/** `encoding` with its selection of `dim` set to `value`, or cleared when `value` is undefined. */
function selecting(encoding: Encoding, dim: string, value: Selection | undefined): Encoding {
  const { [dim]: _, ...rest } = selectOf(encoding);
  const select = value === undefined ? rest : { ...rest, [dim]: value };
  return Object.keys(select).length ? { variable: variableOf(encoding), select } : variableOf(encoding);
}
const colorOf = (layer: Layer): ColorEncoding | undefined =>
  'x' in layer.encode ? layer.encode.color ?? undefined : undefined;
const isSpatial = (layer: Layer) => layer.mark === 'trajectory' && !!layer.encode.z;

/** A coordinate value typed as text: a number, a Boolean, or the text itself. */
function typedValue(text: string): string | number | boolean {
  if (text === 'true' || text === 'false') return text === 'true';
  const number = Number(text);
  return text && Number.isFinite(number) ? number : text;
}

/** How an encoding reads as an axis label: its field's label, and each coordinate it selects. */
function encodingLabel(encoding: Encoding, fields: RoleFields) {
  const variable = variableOf(encoding);
  const base = fields[variable]?.label ?? variable;
  const select = Object.entries(selectOf(encoding)).map(([dim, value]) =>
    `${dim} ${value !== null && typeof value === 'object' ? value.index : String(value)}`);
  return [base, ...select].join(' · ');
}

/** `base` while no part of the figure is keyed by it, then the first free of `base-2`, `base-3`, …. */
function freeKey(base: string, taken: (key: string) => boolean) {
  let key = base;
  for (let number = 2; taken(key); number += 1) key = `${base}-${number}`;
  return key;
}

/** A new layer of `mark`, reading what the figure's first layer reads, so it draws before any of its fields is edited. */
function newLayer(mark: AddedMark, figure: Figure): Layer {
  const first = Object.values(figure.layers)[0];
  const place = { role: first?.role ?? figure.data[0], panel: first?.panel ?? Object.keys(figure.layout)[0] };
  if (mark === 'rule') return { ...place, mark, encode: { axis: 'y', at: 0 } };
  if (mark === 'span') return { ...place, mark, encode: { axis: 'x', start: 0, end: 1 } };
  const read = { ...first?.select && { select: first.select }, ...first?.group && { group: first.group } };
  const position = first && 'x' in first.encode ? { x: first.encode.x, y: first.encode.y } : { x: '', y: '' };
  if (mark === 'trajectory') return { ...place, ...read, mark, encode: position };
  const encode: StoredPointsLayer['encode'] = position;
  if (first?.mark === 'band') encode.error_y = { lower: first.encode.lower, upper: first.encode.upper };
  if (first?.mark === 'points' && first.encode.error_y) encode.error_y = first.encode.error_y;
  return { ...place, ...read, mark, encode };
}

/**
 * The figure's layers, named colour scales and panels as attachments: a row for each, the `+` row that adds a mark, a
 * colour scale or a data role, and the chosen row's fields. A layer's field choices are the fields of the data its role
 * is bound to.
 */
/** Adds a layer as one Undo entry, and gives the key the Service allocated it. */
export type AddFigureLayer = (label: string, layer: Layer) => Promise<string | undefined>;

export function FigureLayers({ figure, values, fields, edit, addLayer, refusal }: {
  figure: Figure;
  /**
   * The figure's effective values, as the Service reads its stored form: each layer's name, and its looks. They are of
   * the reviewed figure, so a layer or scale an edit has just added waits for the review after it.
   */
  values: StudioAnalyzeFigureValues;
  /** The fields a role's bound data offers, when it is bound to data whose fields are known. */
  fields: (role: string) => RoleFields;
  edit?: FigureEdit;
  addLayer?: AddFigureLayer;
  /** Why the figure's last layer edit was refused. */
  refusal?: string;
}) {
  const [chosen, setChosen] = useState<Part>();
  const scales = figure.color_scales ?? {};
  const layer = chosen?.kind === 'layer' ? figure.layers[chosen.name] : undefined;
  const scale = chosen?.kind === 'scale' ? scales[chosen.name] : undefined;
  const panel = chosen?.kind === 'panel' ? figure.layout[chosen.name] : undefined;
  const change = edit && ((label: string, next: Figure) => edit(label, figureEdits(figure, next)));
  const add = async (mark: AddedMark) => {
    const key = await addLayer?.(studioName(mark), newLayer(mark, figure));
    if (key) setChosen({ kind: 'layer', name: key });
  };
  const addScale = () => {
    const name = freeKey('scale', (name) => name in scales);
    change?.(studioName('colorScale'), { ...figure, color_scales: { ...scales, [name]: { kind: 'continuous' } } });
    setChosen({ kind: 'scale', name });
  };
  const addRole = () => {
    const role = freeKey(`data-${figure.data.length + 1}`, (name) => figure.data.includes(name));
    change?.(studioName('dataRole'), { ...figure, data: [...figure.data, role] });
  };
  const remove = ({ kind, name }: Part) => {
    if (kind === 'layer') {
      const { [name]: _, ...rest } = figure.layers;
      change?.(studioName('remove'), { ...figure, layers: rest });
    }
    if (kind === 'scale') {
      const { [name]: _, ...rest } = scales;
      change?.(studioName('remove'), { ...figure, color_scales: rest });
    }
  };
  const removable = ({ kind, name }: Part) => kind === 'layer' ? Object.keys(figure.layers).length > 1
    : kind === 'scale' && !Object.values(figure.layers).some((item) => colorOf(item)?.scale === name);
  const layerValues = chosen?.kind === 'layer' ? values.layers[chosen.name] : undefined;
  const scaleValues = chosen?.kind === 'scale' ? values.color_scales[chosen.name] : undefined;
  const shownFields = layer && layerValues && chosen ? layerFields(figure, chosen.name, layer, layerValues, fields(layer.role), change)
    : scale && scaleValues && chosen ? scaleFields(figure, chosen.name, scale, scaleValues, change)
      : panel && chosen ? panelFields(figure, chosen.name, panel, change) : [];
  return <>
    <Tree aria-label={studioName('layers')} selected={chosen && (layer || scale || panel) ? [partId(chosen)] : []}
      nodes={[
        ...Object.entries(figure.layers).flatMap(([key, item]) => values.layers[key] ? [{
          id: partId({ kind: 'layer', name: key }), name: values.layers[key].name, glyph: markGlyphs[item.mark],
          trailing: <span>{studioName(item.mark)}</span>,
        }] : []),
        ...Object.keys(scales).map((name) => ({
          id: partId({ kind: 'scale', name }), name, glyph: <Palette />, trailing: <span>{studioName('colorScale')}</span>,
        })),
        ...Object.entries(figure.layout).map(([key, item]) => ({
          id: partId({ kind: 'panel', name: key }), name: item.title || key, glyph: <LayoutPanelTop />,
          trailing: <span>{studioName('panel')}</span>,
        })),
      ]}
      onSelectedChange={(ids) => { const id = ids.at(-1); setChosen(id === undefined ? undefined : partOf(id)); }}
      menu={edit ? (node) => removable(partOf(node.id))
        ? <MenuItem onClick={() => remove(partOf(node.id))}>{studioName('remove')}</MenuItem> : undefined : undefined} />
    {edit && <InspectorAddRow label={studioName('addMark')}>
      {addedMarks.map((mark) => <MenuItem key={mark} glyph={markGlyphs[mark]} onClick={() => void add(mark)}>
        {studioName(mark)}
      </MenuItem>)}
      <MenuItem glyph={<Palette />} onClick={addScale}>{studioName('colorScale')}</MenuItem>
      <MenuItem glyph={<Database />} onClick={addRole}>{studioName('dataRole')}</MenuItem>
    </InspectorAddRow>}
    {refusal && <p role="alert">{refusal}</p>}
    {shownFields.length > 0 && <div className="grid min-w-0 gap-1 pt-1">
      {shownFields.map((model) =>
        <ValueField key={JSON.stringify([chosen && partId(chosen), model.id, model.value])} model={model} />)}
    </div>}
  </>;
}

type Change = (label: string, figure: Figure) => void;

function fieldMaker(change: Change | undefined) {
  return (id: StudioNameId | { id: string; label: string }, type: FieldModel['type'], value: JsonValue | undefined,
    next?: (value: JsonValue) => Figure | string) => {
    const { id: key, label } = typeof id === 'string' ? { id, label: studioName(id) } : id;
    return rawField({
      id: key, label, type, value,
      change: change && next && ((value) => {
        const result = next(value);
        if (typeof result === 'string') return result;
        change(label, result);
        return undefined;
      }),
    });
  };
}

const text = (value: JsonValue) => String(value).trim() || null;
const none = () => studioName('noneChoice');

function layerFields(figure: Figure, key: string, layer: Layer, own: StudioFigureLayerValues, offered: RoleFields,
  change?: Change): FieldModel[] {
  const field = fieldMaker(change);
  const names = Object.keys(offered);
  const replaced = (next: Layer, document: Figure = figure): Figure => labelled(figure, layer, offered,
    { ...document, layers: { ...document.layers, [key]: next } });
  const panels = Object.keys(figure.layout);
  const options = (values: readonly string[], current?: string): FieldOption[] =>
    [...new Set([...values, ...current ? [current] : []])].map((value) => ({ value, label: value }));
  const choice = (values: readonly string[], current?: string) => ({ kind: 'choice' as const, options: options(values, current) });
  const optional = (values: readonly string[], current?: string) =>
    ({ kind: 'choice' as const, options: [{ value: none(), label: none() }, ...options(values, current)] });
  const common: FieldModel[] = [
    ...panels.length > 1 ? [field('panel', choice(panels), layer.panel,
      (panel) => replaced({ ...layer, panel: String(panel) }))] : [],
    ...figure.data.length > 1 ? [field('dataRole', choice(figure.data), layer.role,
      (role) => replaced({ ...layer, role: String(role) }))] : [],
  ];
  const style = layer.style ?? {};
  const color = colorOf(layer);
  const label = field('label', { kind: 'string' }, style.label ?? undefined,
    (label) => replaced({ ...layer, style: { ...style, label: text(label) } } as Layer));
  const fixedColor = field('color', { kind: 'string' }, style.color ?? undefined,
    (value) => replaced({ ...layer, style: { ...style, color: text(value) } } as Layer));
  const looks: FieldModel[] = [label, ...color ? [] : [fixedColor]];
  const position = (id: StudioNameId, value: Position, set: (position: Position) => Layer) =>
    field(id, { kind: 'string' }, typeof value === 'number' ? value : variableOf(value), (entered) => {
      const typed = String(entered).trim();
      const number = Number(typed);
      if (typed && Number.isFinite(number)) return replaced(set(number));
      if (names.includes(typed)) return replaced(set(renamed(typeof value === 'number' ? undefined : value, typed)));
      return studioReason('takesNumberOrField', studioName(id));
    });
  const axis = (encode: StoredRuleLayer['encode'] | StoredSpanLayer['encode']) =>
    field('axis', { kind: 'choice', options: axes }, encode.axis,
      (value) => replaced({ ...layer, encode: { ...encode, axis: value } } as Layer));
  if (layer.mark === 'rule') {
    const { encode } = layer;
    return [...common, axis(encode), position('at', encode.at, (at) => ({ ...layer, encode: { ...encode, at } })), ...looks];
  }
  if (layer.mark === 'span') {
    const { encode } = layer;
    return [...common, axis(encode),
      position('start', encode.start, (start) => ({ ...layer, encode: { ...encode, start } })),
      position('end', encode.end, (end) => ({ ...layer, encode: { ...encode, end } })),
      ...looks,
      field('fillAlpha', { kind: 'number' }, own.fill_alpha, (alpha) => {
        const value = Number(alpha);
        return value >= 0 && value <= 1 ? replaced({ ...layer, style: { ...layer.style, fill_alpha: value } })
          : studioReason('takesNumber', studioName('fillAlpha'));
      })];
  }

  const encode = layer.encode;
  const withEncode = (next: Partial<typeof encode>) => ({ ...layer, encode: { ...encode, ...next } } as Layer);
  /** The dimensions a layer's group and selection already fix, which a channel's selection does not offer. */
  const fixed = new Set([
    ...Object.keys(layer.select ?? {}),
    ...(layer.group ?? []).flatMap((name) => name === member ? offered[member]?.dims ?? [] : [name]),
  ]);
  /** A channel's selection of each coordinate of its field that the layer does not fix and that is not trial time. */
  const selections = (id: StudioNameId, value: Encoding | undefined, set: (encoding: Encoding) => Layer) => {
    if (!value) return [];
    const dims = offered[variableOf(value)]?.dims ?? [];
    const free = dims.filter((dim) => !fixed.has(dim) && !offered[dim]?.trial_time && dim in offered);
    return free.map((dim) => {
      const selected = selectOf(value)[dim];
      const shown = selected !== null && typeof selected === 'object' ? selected.index : selected;
      return field({ id: `${id}.${dim}`, label: studioParameterLabel(studioName(id), dim) },
        { kind: 'string' }, shown ?? undefined, (entered) => {
          const typed = String(entered).trim();
          return replaced(set(selecting(value, dim, typed ? typedValue(typed) : undefined)));
        });
    });
  };
  const channel = (id: 'xAxis' | 'yAxis', key: 'x' | 'y') => [
    field(id, choice(names, variableOf(encode[key]) || undefined), variableOf(encode[key]) || undefined,
      (value) => replaced(withEncode({ [key]: renamed(encode[key], String(value)) }))),
    ...selections(id, encode[key], (next) => withEncode({ [key]: next })),
  ];
  const group = field('groupBy', { kind: 'string' }, (layer.group ?? []).join(', '), (entered) => {
    const parts = String(entered).split(',').map((part) => part.trim()).filter(Boolean);
    const unknown = parts.find((part) => !names.includes(part));
    if (unknown !== undefined) return studioReason('takesField', studioName('groupBy'));
    return replaced({ ...layer, group: parts });
  });
  const coloring = [
    field('colorBy', optional(names, color && variableOf(color.field)), color ? variableOf(color.field) : none(), (value) => {
      if (value === none()) {
        const { color: _, ...rest } = encode;
        return replaced({ ...layer, encode: rest } as Layer);
      }
      const scaleNames = Object.keys(figure.color_scales ?? {});
      const scale = color?.scale ?? scaleNames[0] ?? freeKey('scale', () => false);
      const scales = scale in (figure.color_scales ?? {}) ? figure.color_scales
        : { ...figure.color_scales, [scale]: { kind: 'continuous' as const } };
      const { color: _, ...unfixed } = style;
      return replaced({
        ...layer, style: unfixed, encode: { ...encode, color: { field: renamed(color?.field, String(value)), scale } },
      } as Layer, { ...figure, color_scales: scales });
    }),
    ...color ? [field('colorScale', choice(Object.keys(figure.color_scales ?? {}), color.scale), color.scale,
      (scale) => replaced(withEncode({ color: { ...color, scale: String(scale) } })))] : [],
  ];
  const read = [group, ...channel('xAxis', 'x'), ...channel('yAxis', 'y')];
  if (layer.mark === 'points') {
    const { encode } = layer;
    const bounds = encode.error_y;
    const bound = (id: 'lower' | 'upper') => field(id, optional(names, bounds ? variableOf(bounds[id]) : undefined),
      bounds ? variableOf(bounds[id]) : none(), (value) => {
        if (value === none()) {
          const { error_y: _, ...rest } = encode;
          return replaced({ ...layer, encode: rest });
        }
        const variable = String(value);
        const next = bounds ? { ...bounds, [id]: renamed(bounds[id], variable) } : { lower: variable, upper: variable };
        return replaced({ ...layer, encode: { ...encode, error_y: next } });
      });
    return [...common, ...read, bound('lower'), bound('upper'), ...coloring, ...looks,
      field('markerSize', { kind: 'number' }, layer.style?.marker_size ?? undefined, (size) => {
        const value = Number(size);
        return value > 0 ? replaced({ ...layer, style: { ...style, marker_size: value } })
          : studioReason('takesPositive', studioName('markerSize'));
      })];
  }
  if (layer.mark === 'trajectory') {
    const { encode } = layer;
    const z = encode.z ?? undefined;
    const setZ = (next: Encoding | undefined) => {
      const { z: _, ...planar } = encode;
      return replaced({ ...layer, encode: next ? { ...planar, z: next } : planar });
    };
    return [...common, ...read,
      field('zAxis', optional(names, z && variableOf(z)), z ? variableOf(z) : none(),
        (value) => setZ(value === none() ? undefined : renamed(z, String(value)))),
      ...selections('zAxis', z, (next) => ({ ...layer, encode: { ...encode, z: next } })),
      ...coloring, ...looks];
  }
  return [...common, ...coloring, ...looks];
}

const spatialAxes = ['x', 'y', 'z'] as const;

/**
 * `after` with the axis labels of `layer`'s panel kept to its encodings while the panel is 3D: a panel that becomes 3D
 * is labelled by the fields its trajectory reads, a label that is still the label of the encoding it was taken from
 * follows the encoding, and a panel that stops being 3D loses its z axis.
 */
function labelled(before: Figure, layer: Layer, offered: RoleFields, after: Figure): Figure {
  const spatialIn = (document: Figure) => Object.values(document.layers)
    .find((item) => item.panel === layer.panel && isSpatial(item)) as StoredTrajectoryLayer | undefined;
  const was = spatialIn(before);
  const is = spatialIn(after);
  const panel = after.layout[layer.panel];
  if (!panel || (!was && !is)) return after;
  const labels = panel.axes_labels ?? {};
  const label = (trajectory: StoredTrajectoryLayer, axis: typeof spatialAxes[number]) => {
    const encoding = trajectory.encode[axis];
    return encoding ? encodingLabel(encoding, offered) : null;
  };
  const next: StoredFigurePanel = !is ? { ...panel, axes_labels: { ...labels, z: null }, z_axis: null }
    : { ...panel, axes_labels: Object.fromEntries(spatialAxes.map((axis) =>
      [axis, !was || labels[axis] == null || labels[axis] === label(was, axis) ? label(is, axis) : labels[axis]])) };
  return { ...after, layout: { ...after.layout, [layer.panel]: next } };
}

function scaleFields(figure: Figure, name: string, scale: ColorScale, own: StudioFigureColorScaleValues, change?: Change): FieldModel[] {
  const field = fieldMaker(change);
  const set = (next: ColorScale): Figure => ({ ...figure, color_scales: { ...figure.color_scales, [name]: next } });
  const choices = (values: readonly string[]) => ({ kind: 'choice' as const, options: values.map((value) => ({ value, label: value })) });
  const domainText = scale.domain?.map(String).join(', ');
  return [
    field('kind', choices(['continuous', 'ordinal']), scale.kind,
      (kind) => kind === scale.kind ? set(scale)
        : set({ kind: kind as ColorScale['kind'], ...scale.title && { title: scale.title }, ...scale.legend !== undefined && { legend: scale.legend } })),
    field('palette', choices(palettes[scale.kind]), own.palette,
      (palette) => set({ ...scale, palette } as ColorScale)),
    field('domain', { kind: 'string' }, domainText, (entered) => {
      const parts = String(entered).split(',').map((part) => part.trim()).filter(Boolean);
      if (!parts.length) return set({ ...scale, domain: null });
      if (scale.kind === 'ordinal') {
        const values = parts.map(typedValue);
        return new Set(values.map((value) => JSON.stringify(value))).size === values.length
          ? set({ ...scale, domain: values }) : studioReason('takesDistinct', studioName('domain'));
      }
      const [low, high] = parts.map(Number);
      return parts.length === 2 && Number.isFinite(low) && Number.isFinite(high) && low < high
        ? set({ ...scale, domain: [low, high] }) : studioReason('takesIncreasingPair', studioName('domain'));
    }),
    field('title', { kind: 'string' }, scale.title ?? undefined, (title) => set({ ...scale, title: text(title) })),
    field('legend', { kind: 'boolean' }, own.legend, (legend) => set({ ...scale, legend: legend === true })),
  ];
}

function panelFields(figure: Figure, key: string, panel: StoredFigurePanel, change?: Change): FieldModel[] {
  const field = fieldMaker(change);
  const set = (next: StoredFigurePanel): Figure => ({ ...figure, layout: { ...figure.layout, [key]: next } });
  const labels = panel.axes_labels ?? {};
  const spatial = Object.values(figure.layers).some((item) => item.panel === key && isSpatial(item));
  const axisLabel = (id: 'xAxis' | 'yAxis' | 'zAxis', key: 'x' | 'y' | 'z') =>
    field(id, { kind: 'string' }, labels[key] ?? undefined,
      (value) => set({ ...panel, axes_labels: { ...labels, [key]: text(value) } }));
  return [
    field('title', { kind: 'string' }, panel.title ?? undefined, (title) => set({ ...panel, title: text(title) })),
    axisLabel('xAxis', 'x'), axisLabel('yAxis', 'y'), ...spatial ? [axisLabel('zAxis', 'z')] : [],
  ];
}
