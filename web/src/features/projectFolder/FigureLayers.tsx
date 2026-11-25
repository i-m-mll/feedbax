import { ChartLine, Database, Minus, RectangleHorizontal, ScatterChart, Spline } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { studioName, studioReason, type StudioNameId } from '@/components/activity/studioVocabulary';
import { InspectorAddRow } from '@/components/ui/Inspector';
import { MenuItem } from '@/components/ui/Menu';
import { Tree } from '@/components/ui/Tree';
import { rawField } from '@/components/values/adapters/raw';
import type { FieldModel, FieldOption } from '@/components/values/fieldModel';
import { ValueField } from '@/components/values/ValueField';
import type {
  FigureDocument, JsonValue, PointsLayer, RuleLayer, SpanLayer, VariableBinding,
} from '@/generated/studioContracts';
import { generatedName } from './generatedName';

type Layer = FigureDocument['layers'][number];
type Encoding = string | VariableBinding;
type Position = Encoding | number;
type AddedMark = 'points' | 'rule' | 'span';

const markGlyphs: Record<Layer['mark'], ReactNode> = {
  line: <ChartLine />, band: <ChartLine />, points: <ScatterChart />, rule: <Minus />, span: <RectangleHorizontal />,
  trajectory: <Spline />,
};
const addedMarks: readonly AddedMark[] = ['points', 'rule', 'span'];
const axes: readonly FieldOption[] = [{ value: 'x', label: studioName('xAxis') }, { value: 'y', label: studioName('yAxis') }];

/** A figure's edit: the label of its Undo entry and the figure after it. */
export type FigureEdit = (label: string, figure: FigureDocument) => void;

const variableOf = (encoding: Encoding) => typeof encoding === 'string' ? encoding : encoding.variable;
/** An encoding naming `variable`, keeping the selection a bound encoding makes. */
const renamed = (encoding: Encoding | undefined, variable: string): Encoding =>
  encoding && typeof encoding === 'object' ? { ...encoding, variable } : variable;

/** A new layer of `mark`, reading what the figure's first layer reads, so it draws before any of its fields is edited. */
function newLayer(mark: AddedMark, figure: FigureDocument): Layer {
  const first = figure.layers[0];
  const name = generatedName(mark, (name) => figure.layers.some((layer) => layer.name === name));
  const place = { name, role: first?.role ?? figure.data[0], panel: first?.panel ?? figure.layout[0].name };
  if (mark === 'rule') return { ...place, mark, encode: { axis: 'y', at: 0 } };
  if (mark === 'span') return { ...place, mark, encode: { axis: 'x', start: 0, end: 1 } };
  const read = { ...first?.select && { select: first.select }, ...first?.group && { group: first.group } };
  const encode: PointsLayer['encode'] = first && 'x' in first.encode
    ? { x: first.encode.x, y: first.encode.y } : { x: '', y: '' };
  if (first?.mark === 'band') encode.error_y = { lower: first.encode.lower, upper: first.encode.upper };
  if (first?.mark === 'points' && first.encode.error_y) encode.error_y = first.encode.error_y;
  return { ...place, ...read, mark, encode };
}

/**
 * The figure's layers as attachments: a row for each, the `+` row that adds a points, rule or span layer or a data
 * role, and the chosen layer's fields. A layer's field choices are the fields of the data its role is bound to.
 */
export function FigureLayers({ figure, fields, edit, refusal }: {
  figure: FigureDocument;
  /** The fields a role's bound data offers, when it is bound to data whose fields are known. */
  fields: (role: string) => readonly string[];
  edit?: FigureEdit;
  /** Why the figure's last layer edit was refused. */
  refusal?: string;
}) {
  const [chosen, setChosen] = useState<string>();
  const layer = figure.layers.find(({ name }) => name === chosen);
  const write = (label: string, layers: Layer[]) => edit?.(label, { ...figure, layers });
  const replace = (label: string, next: Layer) =>
    write(label, figure.layers.map((item) => item.name === next.name ? next : item));
  const add = (mark: AddedMark) => {
    const next = newLayer(mark, figure);
    write(studioName(mark), [...figure.layers, next]);
    setChosen(next.name);
  };
  const addRole = () => {
    const role = generatedName(`data-${figure.data.length + 1}`, (name) => figure.data.includes(name));
    edit?.(studioName('dataRole'), { ...figure, data: [...figure.data, role] });
  };
  const remove = (name: string) => write(studioName('remove'), figure.layers.filter((item) => item.name !== name));
  return <>
    <Tree aria-label={studioName('layers')} selected={layer ? [layer.name] : []}
      nodes={figure.layers.map((item) => ({
        id: item.name, name: item.name, glyph: markGlyphs[item.mark], trailing: <span>{studioName(item.mark)}</span>,
      }))}
      onSelectedChange={(ids) => setChosen(ids.at(-1))}
      menu={edit && figure.layers.length > 1 ? (node) =>
        <MenuItem onClick={() => remove(node.id)}>{studioName('remove')}</MenuItem> : undefined} />
    {edit && <InspectorAddRow label={studioName('addMark')}>
      {addedMarks.map((mark) => <MenuItem key={mark} glyph={markGlyphs[mark]} onClick={() => add(mark)}>
        {studioName(mark)}
      </MenuItem>)}
      <MenuItem glyph={<Database />} onClick={addRole}>{studioName('dataRole')}</MenuItem>
    </InspectorAddRow>}
    {refusal && <p role="alert">{refusal}</p>}
    {layer && <div className="grid min-w-0 gap-1 pt-1">
      {layerFields(figure, layer, fields(layer.role), edit && ((label, next) => replace(label, next)))
        .map((model) => <ValueField key={JSON.stringify([layer.name, model.id, model.value])} model={model} />)}
    </div>}
  </>;
}

type LayerChange = (label: string, layer: Layer) => void;

function layerFields(figure: FigureDocument, layer: Layer, names: readonly string[], change?: LayerChange): FieldModel[] {
  const field = (id: StudioNameId, type: FieldModel['type'], value: JsonValue | undefined, next?: (value: JsonValue) => Layer | string) =>
    rawField({
      id, label: studioName(id), type, value,
      change: change && next && ((value) => {
        const result = next(value);
        if (typeof result === 'string') return result;
        change(studioName(id), result);
        return undefined;
      }),
    });
  const options = (values: readonly string[], current?: string): FieldOption[] =>
    [...new Set([...values, ...current ? [current] : []])].map((value) => ({ value, label: value }));
  const choice = (values: readonly string[], current?: string) => ({ kind: 'choice' as const, options: options(values, current) });
  const common: FieldModel[] = [
    ...figure.layout.length > 1 ? [field('panel', choice(figure.layout.map(({ name }) => name)), layer.panel,
      (panel) => ({ ...layer, panel: String(panel) }))] : [],
    ...figure.data.length > 1 ? [field('dataRole', choice(figure.data), layer.role,
      (role) => ({ ...layer, role: String(role) }))] : [],
  ];
  const style = layer.style ?? {};
  const text = (value: JsonValue) => String(value).trim() || null;
  const looks: FieldModel[] = [
    field('label', { kind: 'string' }, style.label ?? undefined,
      (label) => ({ ...layer, style: { ...style, label: text(label) } }) as Layer),
    field('color', { kind: 'string' }, style.color ?? undefined,
      (color) => ({ ...layer, style: { ...style, color: text(color) } }) as Layer),
  ];
  const position = (id: StudioNameId, value: Position, set: (position: Position) => Layer) =>
    field(id, { kind: 'string' }, typeof value === 'number' ? value : variableOf(value), (entered) => {
      const typed = String(entered).trim();
      const number = Number(typed);
      if (typed && Number.isFinite(number)) return set(number);
      if (names.includes(typed)) return set(renamed(typeof value === 'number' ? undefined : value, typed));
      return studioReason('takesNumberOrField', studioName(id));
    });
  const axis = (encode: RuleLayer['encode'] | SpanLayer['encode']) =>
    field('axis', { kind: 'choice', options: axes }, encode.axis, (value) => ({ ...layer, encode: { ...encode, axis: value } }) as Layer);
  switch (layer.mark) {
    case 'points': {
      const { encode } = layer;
      const bounds = encode.error_y;
      const none = studioName('noneChoice');
      const bound = (id: 'lower' | 'upper') => field(id,
        { kind: 'choice', options: [{ value: none, label: none }, ...options(names, bounds ? variableOf(bounds[id]) : undefined)] },
        bounds ? variableOf(bounds[id]) : none, (value) => {
          if (value === none) return { ...layer, encode: { x: encode.x, y: encode.y } };
          const variable = String(value);
          const next = bounds ? { ...bounds, [id]: renamed(bounds[id], variable) } : { lower: variable, upper: variable };
          return { ...layer, encode: { ...encode, error_y: next } };
        });
      const channel = (id: 'xAxis' | 'yAxis', key: 'x' | 'y') => field(id, choice(names, variableOf(encode[key]) || undefined),
        variableOf(encode[key]) || undefined, (value) => ({ ...layer, encode: { ...encode, [key]: renamed(encode[key], String(value)) } }));
      return [...common, channel('xAxis', 'x'), channel('yAxis', 'y'), bound('lower'), bound('upper'), ...looks,
        field('markerSize', { kind: 'number' }, layer.style?.marker_size ?? undefined, (size) => {
          const value = Number(size);
          return value > 0 ? { ...layer, style: { ...style, marker_size: value } } : studioReason('takesPositive', studioName('markerSize'));
        })];
    }
    case 'rule': {
      const { encode } = layer;
      return [...common, axis(encode), position('at', encode.at, (at) => ({ ...layer, encode: { ...encode, at } })), ...looks];
    }
    case 'span': {
      const { encode } = layer;
      return [...common, axis(encode),
        position('start', encode.start, (start) => ({ ...layer, encode: { ...encode, start } })),
        position('end', encode.end, (end) => ({ ...layer, encode: { ...encode, end } })),
        ...looks,
        field('fillAlpha', { kind: 'number' }, layer.style?.fill_alpha ?? 0.15, (alpha) => {
          const value = Number(alpha);
          return value >= 0 && value <= 1 ? { ...layer, style: { ...layer.style, fill_alpha: value } }
            : studioReason('takesNumber', studioName('fillAlpha'));
        })];
    }
    default:
      return [...common, ...looks];
  }
}
