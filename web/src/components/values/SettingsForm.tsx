import type { JsonValue } from '@/generated/studioContracts';
import { fromJsonSchema, type SchemaContext } from './adapters/jsonSchema';
import type { FieldModel } from './fieldModel';
import { ValueField } from './ValueField';

export interface SettingsFormProps extends Omit<SchemaContext, 'root'> {
  /** A JSON schema of an object: a method's `settings_schema`, a figure template's, a task's `parameter_schema`. */
  readonly schema: JsonValue;
  readonly value: JsonValue | undefined;
}

/** The fields of a settings form: one per declared value, a nested object as a group whose parts are its values. */
export function settingsFields({ schema, value, ...context }: SettingsFormProps): readonly FieldModel[] {
  const root = fromJsonSchema('', schema, value ?? {}, { root: schema, ...context }, []);
  return root.type.kind === 'group' ? root.parts ?? [] : [root];
}

/** One `ValueField` row per declared value, a nested object as a group of indented rows. */
export function SettingsForm(props: SettingsFormProps) {
  return <>{settingsFields(props).map((model) => <ValueField key={model.id} model={model} />)}</>;
}
