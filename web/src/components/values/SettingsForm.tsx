import type { JsonValue } from '@/generated/studioContracts';
import { fromJsonSchema, type SchemaContext } from './adapters/jsonSchema';
import { ValueField } from './ValueField';

export interface SettingsFormProps extends Omit<SchemaContext, 'root'> {
  /** A JSON schema of an object: a method's `settings_schema`, a figure template's, a task's `parameter_schema`. */
  readonly schema: JsonValue;
  readonly value: JsonValue | undefined;
  /** The `data-field` of each top-level value's row, by which a problem's place reveals it. */
  readonly anchor?: (path: readonly string[]) => string;
}

/** One `ValueField` row per declared value, a nested object as a group of indented rows. */
export function SettingsForm({ schema, value, anchor, ...context }: SettingsFormProps) {
  const root = fromJsonSchema('', schema, value ?? {}, { root: schema, ...context }, []);
  return <>{(root.type.kind === 'group' ? root.parts ?? [] : [root]).map((model) => anchor && model.path
    ? <div key={model.id} data-field={anchor(model.path)} className="contents"><ValueField model={model} /></div>
    : <ValueField key={model.id} model={model} />)}</>;
}
