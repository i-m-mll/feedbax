export type SchemaOrigin = 'declared' | 'inferred_static' | 'curated_fallback' | 'unknown';

export interface ValueSchema {
  id: string;
  label: string;
  kind: string;
  dtype?: string | null;
  shape?: unknown[] | null;
  rank?: number | null;
  units?: string | null;
  frame?: string | null;
  origin: SchemaOrigin;
  metadata: Record<string, unknown>;
}

export interface PortSchema {
  id: string;
  label: string;
  node_id?: string | null;
  component_type?: string | null;
  port: string;
  direction: 'input' | 'output';
  value_schema: ValueSchema;
  bound_task_data_id?: string | null;
  origin: SchemaOrigin;
  metadata: Record<string, unknown>;
}

export interface TaskDataSchema {
  id: string;
  label: string;
  kind: string;
  path: string;
  bindable: boolean;
  value_schema: ValueSchema;
  origin: SchemaOrigin;
  metadata: Record<string, unknown>;
}

export interface SchemaValidationIssue {
  type: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
  location?: Record<string, string> | null;
}

export interface StudioSchemaRegistry {
  kind: 'studio_schema_registry';
  schema_version: string;
  generated_at?: string;
  workspace_id?: string | null;
  scenario_id?: string | null;
  ports: PortSchema[];
  task_data: TaskDataSchema[];
  issues: SchemaValidationIssue[];
  metadata: Record<string, unknown>;
}
