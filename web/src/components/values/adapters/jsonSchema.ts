import type { JsonValue } from '@/generated/studioContracts';
import { humanizeLabel } from '../humanizeLabel';
import type { FieldModel } from '../fieldModel';
import { arrayParts, isRecord, objectWriter, rawField, type RawChange } from './raw';

/** One property of a JSON schema (a task's `parameter_schema`, a method's `settings_schema`) with its value. */
export function fromJsonSchema(key: string, property: JsonValue, value: JsonValue | undefined, change?: RawChange,
  id = key): FieldModel {
  const schema = isRecord(property) ? property : {};
  const label = typeof schema.title === 'string' ? schema.title : humanizeLabel(key);
  const current = value ?? schema.default ?? null;
  if (Array.isArray(schema.enum)) {
    return rawField({ id, label, value: current, change,
      type: { kind: 'choice', options: schema.enum.map((option) => ({ value: option, label: String(option) })) } });
  }
  switch (schema.type) {
    case 'boolean': return rawField({ id, label, type: { kind: 'boolean' }, value: current, change });
    case 'integer': return rawField({ id, label, type: { kind: 'integer' }, value: current, change });
    case 'number': return rawField({ id, label, type: { kind: 'number' }, value: current, change });
    case 'string': return rawField({ id, label, type: { kind: 'string' }, value: current, change });
    case 'array': {
      if (!Array.isArray(current)) break;
      const fixed = Array.isArray(schema.prefixItems) ||
        typeof schema.minItems === 'number' && schema.minItems === schema.maxItems;
      const items = Array.isArray(schema.prefixItems) ? schema.prefixItems : undefined;
      return rawField({ id, label, type: { kind: fixed ? 'vector' : 'list' }, value: current, change,
        parts: arrayParts(id, label, current, change, (index, item, write) =>
          fromJsonSchema(String(index + 1), items?.[index] ?? schema.items ?? null, item, write, `${id}.${index}`)) });
    }
    case 'object': {
      if (!isRecord(schema.properties) || !isRecord(current)) break;
      const write = objectWriter(current, change);
      return rawField({ id, label, type: { kind: 'group' }, value: current, change,
        parts: Object.entries(schema.properties).map(([name, nested]) =>
          fromJsonSchema(name, nested, current[name], write(name), `${id}.${name}`)) });
    }
  }
  return rawField({ id, label, type: { kind: 'readOnly' }, value: current });
}
