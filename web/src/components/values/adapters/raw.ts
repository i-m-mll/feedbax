import type { JsonValue } from '@/generated/studioContracts';
import { studioReason } from '@/components/activity/studioVocabulary';
import type { FieldCommit, FieldModel, FieldType } from '../fieldModel';

/** Writes a value the client holds raw; answers with a reason when it refuses. */
export type RawChange = (value: JsonValue) => Promise<string | undefined> | string | undefined | void;

function parse(type: FieldType, label: string, text: string): { value: JsonValue } | { reason: string } {
  if (type.kind === 'number' || type.kind === 'integer') {
    const value = Number(text);
    if (!text.trim() || !Number.isFinite(value)) return { reason: studioReason('takesNumber', label) };
    if (type.kind === 'integer' && !Number.isInteger(value)) return { reason: studioReason('takesInteger', label) };
    return { value };
  }
  return { value: text };
}

export function rawText(value: JsonValue | undefined) {
  return value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value);
}

/** A field whose value the client holds as JSON: it parses text by the field's type. Raw fields offer Value only. */
export function rawField({ id, label, type, value, unit, origin, change, parts }: {
  id: string; label: string; type: FieldType; value: JsonValue | undefined; unit?: string; origin?: string;
  change?: RawChange; parts?: readonly FieldModel[];
}): FieldModel {
  const write = async (next: JsonValue): Promise<string | undefined> => (await change!(next)) || undefined;
  const commit: FieldCommit | undefined = change && (async (edit) => {
    if (edit.kind === 'set') return write(edit.value);
    if (edit.kind !== 'text') return undefined;
    const parsed = parse(type, label, edit.text);
    return 'reason' in parsed ? parsed.reason : write(parsed.value);
  });
  return {
    id, label, type, unit, origin, value: value ?? undefined, text: rawText(value), kind: 'value',
    kinds: [{ kind: 'value', enabled: !!commit }], differs: false, names: [], functions: [], parts, commit,
  };
}

/** The parts of an array value, each writing its element back into the array. */
export function arrayParts(id: string, label: string, value: readonly JsonValue[], change: RawChange | undefined,
  part: (index: number, item: JsonValue, write: RawChange | undefined) => FieldModel): FieldModel[] {
  return value.map((item, index) => part(index, item, change && ((next) =>
    change(value.map((old, position) => position === index ? next : old)))));
}

/** The parts of an object value, each writing its key back into the object. */
export function objectWriter(value: Readonly<Record<string, JsonValue>>, change: RawChange | undefined) {
  return (key: string): RawChange | undefined => change && ((next) => change({ ...value, [key]: next }));
}

export function isRecord(value: unknown): value is Record<string, JsonValue> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}
