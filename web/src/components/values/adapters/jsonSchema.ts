import type { JsonValue } from '@/generated/studioContracts';
import { humanizeLabel } from '../humanizeLabel';
import type { FieldModel, FieldOption } from '../fieldModel';
import { isRecord, rawField, type RawChange } from './raw';

type Schema = Readonly<Record<string, JsonValue>>;

/** Writes the value at a path below the form's root; answers with a reason when the source refuses. */
export type SchemaWrite = (path: readonly string[], value: JsonValue) => ReturnType<RawChange>;

export interface SchemaContext {
  /** The schema whose `$defs` resolve `$ref`. */
  readonly root: JsonValue;
  readonly write?: SchemaWrite;
  /** A field the server projects for the value at this path (a sweepable number), drawn in place of the schema's. */
  readonly projected?: (path: readonly string[]) => FieldModel | undefined;
  /** Where the value at this path comes from, when not from the document itself. */
  readonly origin?: (path: readonly string[]) => string | undefined;
}

const without = (schema: Schema, ...keys: string[]) =>
  Object.fromEntries(Object.entries(schema).filter(([key]) => !keys.includes(key)));

/** Follows `$ref` into `$defs`. A definition's own title names its class, not the field, so the field's is kept. */
function resolve(schema: JsonValue, root: JsonValue): Schema {
  if (!isRecord(schema)) return {};
  const ref = schema.$ref;
  if (typeof ref !== 'string' || !ref.startsWith('#/$defs/') || !isRecord(root) || !isRecord(root.$defs)) return schema;
  return { ...without(resolve(root.$defs[ref.slice('#/$defs/'.length)] ?? null, root), 'title'), ...without(schema, '$ref') };
}

function jsonType(value: JsonValue) {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  if (typeof value === 'number') return Number.isInteger(value) ? 'integer' : 'number';
  return typeof value;
}

function accepts(schema: Schema, value: JsonValue) {
  const type = jsonType(value);
  return schema.type === type || type === 'integer' && schema.type === 'number' ||
    Array.isArray(schema.enum) && schema.enum.includes(value);
}

/** The property every member fixes to a constant: a discriminated union's tag. */
function tag(members: readonly Schema[]) {
  const consts = (member: Schema) => Object.entries(isRecord(member.properties) ? member.properties : {})
    .filter(([, property]) => isRecord(property) && 'const' in property).map(([name]) => name);
  const [first, ...rest] = members.map(consts);
  return first?.find((name) => rest.every((names) => names.includes(name)));
}

function constOf(member: Schema, name: string) {
  const property = isRecord(member.properties) ? member.properties[name] : undefined;
  return isRecord(property) ? property.const ?? null : null;
}

function defaults(member: Schema): Record<string, JsonValue> {
  return Object.fromEntries(Object.entries(isRecord(member.properties) ? member.properties : {})
    .flatMap(([name, property]) => isRecord(property) && 'default' in property ? [[name, property.default]] : []));
}

export function pointer(path: readonly string[]) {
  return path.map((part) => '/' + part.replace(/~/g, '~0').replace(/\//g, '~1')).join('');
}

/** `value` with the value at `path` replaced. */
export function setIn(value: JsonValue | undefined, path: readonly string[], next: JsonValue): JsonValue {
  if (!path.length) return next;
  const [head, ...rest] = path;
  if (Array.isArray(value)) return value.map((item, index) => String(index) === head ? setIn(item, rest, next) : item);
  const record = isRecord(value) ? value : {};
  return { ...record, [head]: setIn(record[head], rest, next) };
}

/**
 * The field for one value declared by a JSON schema: a method's `settings_schema`, a figure template's, a task's
 * `parameter_schema`. An optional value (`anyOf` with null) is its non-null member; a union of types is the member
 * the value has; a discriminated union is a choice of its tag that swaps the group beneath it.
 */
export function fromJsonSchema(key: string, property: JsonValue, value: JsonValue | undefined, context: SchemaContext,
  path: readonly string[] = [key]): FieldModel {
  const projection = context.projected?.(path);
  if (projection) return projection;
  const outer = resolve(property, context.root);
  const label = typeof outer.title === 'string' ? outer.title : humanizeLabel(key);
  const id = pointer(path);
  const change: RawChange | undefined = context.write && ((next) => context.write!(path, next));
  const origin = context.origin?.(path);
  const current = value ?? outer.default ?? null;
  const union = Array.isArray(outer.anyOf) ? outer.anyOf : Array.isArray(outer.oneOf) ? outer.oneOf : undefined;
  let schema = outer;
  if (union) {
    const members = union.map((member) => without(resolve(member, context.root), 'title')).filter((member) => member.type !== 'null');
    const name = members.length > 1 ? tag(members) : undefined;
    if (name) return discriminated(label, id, name, members, current, path, context, change);
    const member = members.length === 1 ? members[0] : members.find((candidate) => accepts(candidate, current));
    schema = { ...(member ?? {}), ...without(outer, 'anyOf', 'oneOf') };
  }
  if (Array.isArray(schema.enum)) {
    return rawField({ id, label, value: current, change, origin,
      type: { kind: 'choice', options: schema.enum.map((option) => ({ value: option, label: String(option) })) } });
  }
  switch (schema.type) {
    case 'boolean': return rawField({ id, label, type: { kind: 'boolean' }, value: current, change, origin });
    case 'integer': return rawField({ id, label, type: { kind: 'integer' }, value: current, change, origin });
    case 'number': return rawField({ id, label, type: { kind: 'number' }, value: current, change, origin });
    case 'string': return rawField({ id, label, type: { kind: 'string' }, value: current, change, origin });
    case 'array': {
      if (!Array.isArray(current)) break;
      const fixed = Array.isArray(schema.prefixItems) ||
        typeof schema.minItems === 'number' && schema.minItems === schema.maxItems;
      const items = Array.isArray(schema.prefixItems) ? schema.prefixItems : undefined;
      return rawField({ id, label, type: { kind: fixed ? 'vector' : 'list' }, value: current, change,
        parts: current.map((item, index) => fromJsonSchema(String(index + 1), items?.[index] ?? schema.items ?? null,
          item, context, [...path, String(index)])) });
    }
    case 'object': {
      if (!isRecord(current)) break;
      const declared = isRecord(schema.properties) ? Object.entries(schema.properties) : [];
      const extra = schema.additionalProperties ?? null;
      const free = isRecord(extra)
        ? Object.keys(current).filter((name) => !declared.some(([key]) => key === name)).map((name) => [name, extra] as const)
        : [];
      return rawField({ id, label, type: { kind: 'group' }, value: current, change,
        parts: [...declared, ...free].map(([name, nested]) =>
          fromJsonSchema(name, nested, current[name], context, [...path, name])) });
    }
  }
  return rawField({ id, label, type: { kind: 'readOnly' }, value: current });
}

function discriminated(label: string, id: string, name: string, members: readonly Schema[], current: JsonValue,
  path: readonly string[], context: SchemaContext, change: RawChange | undefined): FieldModel {
  const chosen = isRecord(current) ? members.find((member) => constOf(member, name) === current[name]) : undefined;
  const options: FieldOption[] = members.map((member) => ({
    value: constOf(member, name), label: String(constOf(member, name)),
  }));
  const choice = rawField({ id: pointer([...path, name]), label: humanizeLabel(name), value: chosen ? constOf(chosen, name) : null,
    type: { kind: 'choice', options },
    change: change && ((next) => {
      const member = members.find((candidate) => constOf(candidate, name) === next);
      return member && change({ ...defaults(member), [name]: next });
    }) });
  const rest = chosen && isRecord(chosen.properties) && isRecord(current)
    ? Object.entries(chosen.properties).filter(([key]) => key !== name)
      .map(([key, nested]) => fromJsonSchema(key, nested, current[key], context, [...path, key]))
    : [];
  return rawField({ id, label, type: { kind: 'group' }, value: current, parts: [choice, ...rest] });
}
