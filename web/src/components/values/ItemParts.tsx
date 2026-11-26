import { studioName } from '@/components/activity/studioVocabulary';
import type { StudioFieldName } from '@/generated/studioContracts';
import type { FieldEdit, FieldModel } from './fieldModel';
import { ValueField } from './ValueField';

/** The repetition a bracket's cell reads by, as its site names it. */
const repetition = 'i';

/** Which item of a list a value reads: the repetition's, one by its number, or none yet. */
export type ListItem = 'each' | number | undefined;

/** A value that reads a list: the list, and the item it reads. */
export interface ListRead {
  readonly list: StudioFieldName;
  readonly item: ListItem;
}

/** The list the field's value reads at its root, as `targets[i]`, `targets[0]` or `targets` alone, when it reads one. */
export function listRead(model: FieldModel): ListRead | undefined {
  const value = model.structure?.value;
  const list = (reference: string) => model.names.find(({ id, list_count }) => id === reference && list_count);
  if (value?.kind === 'name') {
    const read = list(value.reference);
    return read && { list: read, item: undefined };
  }
  if (value?.kind !== 'call' || value.function !== 'index') return undefined;
  const [source, item] = value.args ?? [];
  const read = source?.kind === 'name' ? list(source.reference) : undefined;
  if (!read) return undefined;
  return {
    list: read,
    item: item?.kind === 'name' && item.reference === repetition ? 'each'
      : item?.kind === 'literal' && Number.isInteger(item.value) ? item.value as number : undefined,
  };
}

/** Whether the field's site is inside a bracket, where `i` is the repetition. */
export const inRepeat = (model: FieldModel) => model.names.some(({ id }) => id === repetition);

/** The text that reads `item` of the list named `name`. */
const itemText = (name: string, item: ListItem) =>
  item === undefined ? name : `${name}[${item === 'each' ? repetition : item}]`;

/**
 * A list's read as its parts under the field: the Variable it reads, among the site's lists, and the Item, Each Repeat
 * inside a bracket or one item by its number. Choosing either writes the value's text.
 */
export function ItemParts({ model, read }: { model: FieldModel; read: ListRead }) {
  const lists = model.names.filter(({ list_count }) => list_count);
  const count = read.list.list_count ?? 0;
  const commit = model.commit;
  const write = (name: string, item: ListItem) => commit
    ? commit({ kind: 'text', text: itemText(name, item) }) : Promise.resolve(undefined);
  const chosen = (edit: FieldEdit) => edit.kind === 'set' ? edit.value : undefined;
  const base = { names: model.names, functions: model.functions, kinds: [], kind: 'value', text: null } as const;
  const parts: FieldModel[] = [
    {
      ...base, id: model.id + '.list', label: studioName('variable'), value: read.list.id,
      type: { kind: 'choice', options: lists.map(({ id, name }) => ({ value: id, label: name })) },
      commit: commit && ((edit) => {
        const list = lists.find(({ id }) => id === chosen(edit));
        return list && list.id !== read.list.id ? write(list.name, read.item) : Promise.resolve(undefined);
      }),
    },
    {
      ...base, id: model.id + '.item', label: studioName('item'), value: read.item ?? null,
      type: { kind: 'choice', options: [
        ...inRepeat(model) ? [{ value: 'each', label: studioName('eachRepeat') }] : [],
        ...Array.from({ length: count }, (_, item) => ({ value: item, label: String(item) })),
      ] },
      commit: commit && ((edit) => {
        const item = chosen(edit);
        return (item === 'each' || typeof item === 'number') && item !== read.item
          ? write(read.list.name, item) : Promise.resolve(undefined);
      }),
    },
  ];
  return <div className="grid gap-1">
    {parts.map((part) => <ValueField key={part.id + JSON.stringify(part.value)} model={part} />)}
  </div>;
}
