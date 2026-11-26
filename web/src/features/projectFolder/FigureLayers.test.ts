import { expect, it } from 'vitest';
import type { StoredFigureDocument } from '@/generated/studioContracts';
import { figureEdits } from './FigureLayers';

const line = (y: string) => ({ role: 'states', panel: 'main', mark: 'line', encode: { x: 'time', y } }) as const;
const figure: StoredFigureDocument = {
  data: ['states'],
  layout: { main: { row: 0, col: 0 } },
  layers: Object.fromEntries(Array.from({ length: 40 }, (_, index) => [`line-${index}`, line(`y${index}`)])),
};

it('writes one member edit for one changed layer, however many layers the figure has', () => {
  const after = { ...figure, layers: { ...figure.layers, 'line-7': line('speed') } };
  expect(figureEdits(figure, after)).toEqual([{ op: 'replace', path: ['layers', 'line-7'], value: line('speed') }]);
});

it('removes a layer by its key, and writes the order only when kept layers move', () => {
  const { 'line-3': _, ...rest } = figure.layers;
  expect(figureEdits(figure, { ...figure, layers: rest })).toEqual([{ op: 'remove', path: ['layers', 'line-3'] }]);
  const { 'line-0': first, ...others } = figure.layers;
  const moved = { ...others, 'line-0': first };
  expect(figureEdits(figure, { ...figure, layers: moved })).toEqual([
    { op: 'order', path: ['layers'], order: Object.keys(moved) },
  ]);
});
