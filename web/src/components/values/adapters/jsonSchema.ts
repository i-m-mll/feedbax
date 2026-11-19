import type { JsonValue } from '@/generated/studioContracts';
import { humanizeLabel } from '../humanizeLabel';
import type { FieldModel, FieldOption } from '../fieldModel';
import { isRecord, rawField, type RawChange } from './raw';

type Schema = Readonly<Record<string, JsonValue>>;

/** Writes the value at a path below the form's root; answers with a reason when the source refuses. */
export type SchemaWrite = (path: readonly string[], value: JsonValue) => ReturnType<RawChange>;

/**
 * Puts the class whose key is `key` at a class-valued path, in place of the current one; the source decides what of
 * the current value the class keeps. Answers with a reason when the source refuses.
 */
export type SchemaChoose = (path: readonly string[], key: JsonValue) => ReturnType<RawChange>;

export interface SchemaContext {
  /** The schema whose `$defs` resolve `$ref`. */
  readonly root: JsonValue;
  readonly write?: SchemaWrite;
  /** Without it, a class-valued field shows its class and cannot change it. */
  readonly choose?: SchemaChoose;
  /** A field the server projects for the value at this path (a sweepable number), drawn in place of the schema's. */
  readonly projected?: (path: readonly string[]) => FieldModel | undefined;
  /** Where the value at this path comes from, when not from the document itself. */
  readonly origin?: (path: readonly string[]) => string | undefined;
}

const without = (schema: Schema, ...keys: string[]) =>
  Object.fromEntries(Object.entries(schema).filter(([key]) => !keys.includes(key)));

/**
 * Follows `$ref` into `$defs`. A definition's own title names its class, not the field, so the field's is kept unless
 * `own` asks for the class's.
 */
function resolve(schema: JsonValue, root: JsonValue, own = false): Schema {
  if (!isRecord(schema)) return {};
  const ref = schema.$ref;
  if (typeof ref !== 'string' || !ref.startsWith('#/$defs/') || !isRecord(root) || !isRecord(root.$defs)) return schema;
  const definition = resolve(root.$defs[ref.slice('#/$defs/'.length)] ?? null, root, own);
  return { ...(own ? definition : without(definition, 'title')), ...without(schema, '$ref') };
}

/** A union's members other than null, a nested union's members in its place, and the property that tags them. */
function unionOf(schema: Schema, root: JsonValue): { members: Schema[]; name?: string } | undefined {
  const union = Array.isArray(schema.anyOf) ? schema.anyOf : Array.isArray(schema.oneOf) ? schema.oneOf : undefined;
  if (!union) return undefined;
  const discriminator = isRecord(schema.discriminator) ? schema.discriminator.propertyName : undefined;
  let name = typeof discriminator === 'string' ? discriminator : undefined;
  const members: Schema[] = [];
  for (const member of union) {
    const resolved = resolve(member, root, true);
    const inner = unionOf(resolved, root);
    if (inner) {
      members.push(...inner.members);
      name ??= inner.name;
    } else if (resolved.type !== 'null') members.push(resolved);
  }
  return { members, name: name ?? (members.length > 1 ? tag(members) : undefined) };
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

function fixed(property: JsonValue): JsonValue | undefined {
  if (!isRecord(property)) return undefined;
  if ('const' in property) return property.const;
  return Array.isArray(property.enum) && property.enum.length === 1 ? property.enum[0] : undefined;
}

/** The property every member fixes to a constant: a discriminated union's tag. */
function tag(members: readonly Schema[]) {
  const consts = (member: Schema) => Object.entries(isRecord(member.properties) ? member.properties : {})
    .filter(([, property]) => fixed(property) !== undefined).map(([name]) => name);
  const [first, ...rest] = members.map(consts);
  return first?.find((name) => rest.every((names) => names.includes(name)));
}

function constOf(member: Schema, name: string) {
  return fixed(isRecord(member.properties) ? member.properties[name] ?? null : null) ?? null;
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
 * the value has; a discriminated union is a class: a choice of its tag whose parameters are drawn beneath it.
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
  const union = unionOf(outer, context.root);
  let schema = outer;
  if (union) {
    const { members, name } = union;
    if (name) return classField(label, id, name, members, current, path, context, origin);
    const member = members.length === 1 ? members[0] : members.find((candidate) => accepts(candidate, current));
    schema = { ...without(member ?? {}, 'title'), ...without(outer, 'anyOf', 'oneOf', 'discriminator') };
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

/** A value that is one of several classes: a choice of class, the chosen class's parameters as its parts. */
function classField(label: string, id: string, name: string, members: readonly Schema[], current: JsonValue,
  path: readonly string[], context: SchemaContext, origin: string | undefined): FieldModel {
  const chosen = isRecord(current) ? members.find((member) => constOf(member, name) === current[name]) : undefined;
  const options: FieldOption[] = members.map((member) => {
    const value = constOf(member, name);
    return { value, label: typeof member.title === 'string' ? member.title : String(value) };
  });
  const parameters = chosen && isRecord(chosen.properties) && isRecord(current)
    ? Object.entries(chosen.properties).filter(([key]) => key !== name)
    : [];
  const parts = parameters.map(([key, nested]) => fromJsonSchema(key, nested, (current as Record<string, JsonValue>)[key], context, [...path, key]));
  if (parts.length === 1) {
    const [[key, nested]] = parameters;
    const own = resolve(nested, context.root);
    parts[0] = { ...parts[0], label, parameter: typeof own.title === 'string' ? own.title : humanizeLabel(key) };
  }
  return rawField({
    id, label, origin, value: chosen ? constOf(chosen, name) : null, type: { kind: 'class', options }, parts,
    change: context.choose && ((key) => context.choose!(path, key)),
  });
}
