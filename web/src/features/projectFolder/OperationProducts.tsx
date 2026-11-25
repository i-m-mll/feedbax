import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { requestJson } from '@/api/request';
import { StudioRegion, studioReadState } from '@/components/activity/StudioRegion';
import { studioAccessibleName, studioFormat, studioName } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { Section } from '@/components/ui/Section';
import { Table, TableCell, TableHeaderCell, TableRow, TableRowHeader } from '@/components/ui/Table';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import {
  parseContract, type DatasetDescriptor, type FieldDescriptor, type JsonValue, type StudioSavedLightHistory, type StudioSavedLightResult,
} from '@/generated/studioContracts';
import { LightResultChoice } from './SavedLightProducts';

/** An Analyze operation whose current or previous result the Results read. */
export interface OperationResult {
  readonly name: string;
  readonly label: string;
  readonly history: StudioSavedLightHistory;
}

export const productPageSize = 128;
const shownNumber = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 6 });
const count = new Intl.NumberFormat();
const hasProducts = ({ products }: StudioSavedLightResult) => Object.keys(products).length > 0;
const columnBounds = { least: '4rem', greatest: '16rem' };

function fieldName(name: string, field: FieldDescriptor) {
  const label = field.label ?? humanizeLabel(name);
  return field.units ? studioFormat('withUnit', { name: label, unit: field.units }) : label;
}

function shownValue(value: JsonValue) {
  return typeof value === 'number' ? shownNumber.format(value) : typeof value === 'string' ? value : JSON.stringify(value);
}

/** The index along each of `sizes` of the `flat`th value in row-major order. */
export function rowMajorIndex(flat: number, sizes: readonly number[]) {
  const index = new Array<number>(sizes.length);
  let rest = flat;
  for (let axis = sizes.length - 1; axis >= 0; axis -= 1) {
    index[axis] = rest % sizes[axis];
    rest = Math.floor(rest / sizes[axis]);
  }
  return index;
}

/** What a dimension's positions are read as: its one-dimensional coordinate's inline values, or none. */
function dimensionLabels(descriptor: DatasetDescriptor, dimension: string) {
  const coordinate = Object.values(descriptor.coords ?? {})
    .find(({ dims, inline_values }) => dims.length === 1 && dims[0] === dimension && inline_values);
  return coordinate?.inline_values ?? undefined;
}

function FieldValues({ url, descriptor, field }: { url: string; descriptor: DatasetDescriptor; field: string }) {
  const [offset, setOffset] = useState(0);
  const dims = descriptor.variables[field].dims;
  const page = useQuery({
    queryKey: ['operation-dataset-page', url, field, offset],
    queryFn: async ({ signal }) => parseContract('StudioOperationDatasetPage', await requestJson(
      `${url}/values?${new URLSearchParams({ field, offset: String(offset), limit: String(productPageSize) })}`, { signal })),
    placeholderData: (previous) => previous?.field === field ? previous : undefined,
  });
  const data = page.data;
  const labels = dims.map((dimension) => dimensionLabels(descriptor, dimension));
  const sizes = dims.map((dimension) => descriptor.dimensions[dimension]);
  const through = data ? data.offset + data.values.length : 0;
  return <StudioRegion state={studioReadState('values', data, page.error, () => { void page.refetch(); })}>
    {data && <div className="grid gap-1">
      <div className="flex items-center gap-2">
        <Button disabled={data.offset === 0} onClick={() => setOffset(Math.max(0, data.offset - productPageSize))}>
          {studioName('previousPage')}
        </Button>
        <Button disabled={through >= data.total} onClick={() => setOffset(through)}>{studioName('nextPage')}</Button>
        <span className="tabular-nums text-shell-muted">{studioFormat('pageRange', {
          from: count.format(data.total ? data.offset + 1 : 0), through: count.format(through), total: count.format(data.total),
        })}</span>
      </div>
      <Table aria-label={studioAccessibleName('values', fieldName(field, descriptor.variables[field]))} columns={[...dims.map(() => columnBounds), columnBounds]}
        aria-busy={page.isPlaceholderData || undefined}>
        <thead><tr>
          {dims.map((dimension) => <TableHeaderCell key={dimension}>{humanizeLabel(dimension)}</TableHeaderCell>)}
          <TableHeaderCell>{fieldName(field, descriptor.variables[field])}</TableHeaderCell>
        </tr></thead>
        <tbody>{data.values.map((value, row) => {
          const index = rowMajorIndex(data.offset + row, sizes);
          return <TableRow key={data.offset + row}>
            {index.map((position, axis) => <TableCell key={dims[axis]} className="tabular-nums">
              {labels[axis] ? shownValue(labels[axis]![position]) : position}
            </TableCell>)}
            <TableCell className="tabular-nums">{shownValue(value)}</TableCell>
          </TableRow>;
        })}</tbody>
      </Table>
    </div>}
  </StudioRegion>;
}

function FieldTable({ name, fields, descriptor, selected, onSelect }: {
  name: 'variables' | 'labels';
  fields: Record<string, FieldDescriptor>;
  descriptor: DatasetDescriptor;
  selected?: string;
  onSelect?(field: string): void;
}) {
  const shape = (field: FieldDescriptor) => field.dims
    .map((dimension) => `${humanizeLabel(dimension)} ${count.format(descriptor.dimensions[dimension])}`).join(' × ');
  return <Table aria-label={studioName(name)} held={1} columns={[columnBounds, columnBounds, columnBounds]}>
    <thead><tr>
      <TableHeaderCell>{studioName(name)}</TableHeaderCell>
      <TableHeaderCell>{studioName('dimensions')}</TableHeaderCell>
      <TableHeaderCell>{studioName('units')}</TableHeaderCell>
    </tr></thead>
    <tbody>{Object.entries(fields).map(([key, field]) => <TableRow key={key} selected={key === selected}>
      <TableRowHeader>
        {onSelect ? <Button variant="bare" size="content" pressed={key === selected} onClick={() => onSelect(key)}>
          {field.label ?? humanizeLabel(key)}
        </Button> : field.label ?? humanizeLabel(key)}
      </TableRowHeader>
      <TableCell className="tabular-nums">{shape(field)}</TableCell>
      <TableCell>{field.units ?? ''}</TableCell>
    </TableRow>)}</tbody>
  </Table>;
}

/** One product's dataset: its variables and coordinates with their dimensions and units, and a page of one variable. */
function ProductData({ role, url }: { role: string; url: string }) {
  const descriptor = useQuery({
    queryKey: ['operation-dataset', url],
    queryFn: async ({ signal }) => parseContract('DatasetDescriptor', await requestJson(url, { signal })),
    staleTime: Infinity,
  });
  const [chosen, setChosen] = useState<string>();
  const data = descriptor.data;
  const variables = Object.keys(data?.variables ?? {});
  const field = chosen && variables.includes(chosen) ? chosen : variables[0];
  return <Section level="field" name={humanizeLabel(role)} bodyClassName="grid gap-2">
    <StudioRegion state={studioReadState('values', data, descriptor.error, () => { void descriptor.refetch(); })}>
      {data && <>
        <FieldTable name="variables" fields={data.variables} descriptor={data} selected={field} onSelect={setChosen} />
        {Object.keys(data.coords ?? {}).length > 0 && <FieldTable name="labels" fields={data.coords!} descriptor={data} />}
        {field && <FieldValues key={field} url={url} descriptor={data} field={field} />}
      </>}
    </StudioRegion>
  </Section>;
}

/** The Results of an Analyze step: each operation's products, read as data. */
export function OperationProducts({ operations }: { operations: readonly OperationResult[] }) {
  return <div className="grid gap-3">{operations.map(({ name, label, history }) =>
    <Section key={name} level="group" name={label} bodyClassName="grid gap-2">
      <LightResultChoice history={history} holds={hasProducts} className="grid gap-2">
        {(result) => Object.entries(result.products).map(([role, url]) => <ProductData key={url} role={role} url={url} />)}
      </LightResultChoice>
    </Section>)}
  </div>;
}

/** Whether an operation's current or previous result has a product for the Results to read. */
export function holdsProducts(history: StudioSavedLightHistory) {
  return [history.result, history.previous].some((result) => result && hasProducts(result));
}
