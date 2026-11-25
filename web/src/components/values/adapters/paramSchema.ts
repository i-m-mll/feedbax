import type { JsonValue, ParamSchema } from '@/generated/studioContracts';
import { humanizeLabel } from '../humanizeLabel';
import type { FieldModel, FieldOption, FieldType } from '../fieldModel';
import { arrayParts, isRecord, objectWriter, rawField, type RawChange } from './raw';

function primitiveType(value: JsonValue): FieldType {
  return typeof value === 'number' ? { kind: 'number' } : typeof value === 'boolean' ? { kind: 'boolean' }
    : typeof value === 'string' ? { kind: 'string' } : { kind: 'readOnly' };
}

function element(id: string, label: string, value: JsonValue, change: RawChange | undefined): FieldModel {
  if (Array.isArray(value)) {
    return rawField({ id, label, type: { kind: 'vector' }, value, change,
      parts: arrayParts(id, label, value, change, (index, item, write) => element(`${id}.${index}`, String(index + 1), item, write)) });
  }
  return rawField({ id, label, type: primitiveType(value), value, change });
}

/**
 * A parameter declared by a `ParamSchema` with its raw value: a component's parameter, or an Analyze operation's,
 * whose choices may be the operation's variables (`options`).
 */
export function fromParamSchema(schema: ParamSchema, value: JsonValue | undefined, change?: RawChange,
  { id = schema.name, options = schema.options?.map((option) => ({ value: option, label: option })) }:
  { id?: string; options?: readonly FieldOption[] } = {}): FieldModel {
  const label = humanizeLabel(schema.name);
  const current = value ?? (schema.default as JsonValue | undefined) ?? null;
  if (options?.length) return rawField({ id, label, value: current, change, type: { kind: 'choice', options } });
  switch (schema.type) {
    case 'bool': return rawField({ id, label, type: { kind: 'boolean' }, value: current, change });
    case 'int': return rawField({ id, label, type: { kind: 'integer' }, value: current, change });
    case 'float': return rawField({ id, label, type: { kind: 'number' }, value: current, change });
    case 'str': case 'enum': return rawField({ id, label, type: { kind: 'string' }, value: current, change });
    case 'array': case 'bounds2d': {
      if (!Array.isArray(current)) return rawField({ id, label, type: { kind: 'readOnly' }, value: current });
      return rawField({ id, label, type: { kind: schema.type === 'bounds2d' ? 'vector' : 'list' }, value: current, change,
        parts: arrayParts(id, label, current, change, (index, item, write) => element(`${id}.${index}`, String(index + 1), item, write)) });
    }
    case 'object': {
      if (!schema.nested_schema || !isRecord(current)) return rawField({ id, label, type: { kind: 'readOnly' }, value: current });
      const write = objectWriter(current, change);
      return rawField({ id, label, type: { kind: 'group' }, value: current, change,
        parts: schema.nested_schema.map((nested) => fromParamSchema(nested, current[nested.name], write(nested.name),
          { id: `${id}.${nested.name}` })) });
    }
  }
}
