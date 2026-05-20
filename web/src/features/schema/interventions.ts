import type { TapSpec } from '@/types/graph';
import type {
  SelectorTargetSchema,
  StudioInterventionTransformSpec,
  StudioSchemaRegistry,
  StudioValidationIssue,
  StudioValueSpec,
  ValueSchema,
} from '@/types/workspace';

const NUMERIC_DTYPES = new Set([
  'float',
  'float16',
  'float32',
  'float64',
  'int',
  'int16',
  'int32',
  'int64',
  'uint',
  'uint16',
  'uint32',
  'uint64',
  'number',
  'scalar',
  'vector',
  'array',
  'tensor',
]);

const SUPPORTED_INTERVENTION_OPERATIONS = new Set([
  'clamp',
  'noise',
  'constant',
  'offset',
  'scale',
]);

function issue(
  type: string,
  message: string,
  path: string,
  severity: StudioValidationIssue['severity'] = 'error'
): StudioValidationIssue {
  return { type, message, severity, location: { path } };
}

function selectorCompact(spec: StudioInterventionTransformSpec): string | null {
  const compact = spec.target_selector?.compact;
  return typeof compact === 'string' && compact.length > 0 ? compact : null;
}

function selectorSchemaTargetId(spec: StudioInterventionTransformSpec): string | null {
  const id = spec.target_selector?.metadata?.schema_target_id;
  return typeof id === 'string' && id.length > 0 ? id : null;
}

function resolveSelectorTarget(
  spec: StudioInterventionTransformSpec,
  registry: Pick<StudioSchemaRegistry, 'selector_targets'>
): SelectorTargetSchema | null {
  const schemaTargetId = selectorSchemaTargetId(spec);
  const compact = selectorCompact(spec);
  return (
    registry.selector_targets.find((target) => target.id === schemaTargetId) ??
    registry.selector_targets.find((target) => target.selector === compact) ??
    null
  );
}

function isNumericSchema(schema: ValueSchema): boolean | null {
  if (!schema.dtype) return null;
  const dtype = schema.dtype.toLowerCase();
  if (NUMERIC_DTYPES.has(dtype)) return true;
  if (dtype.includes('float') || dtype.includes('int')) return true;
  return false;
}

function shapeCompatible(source: unknown[] | null | undefined, target: unknown[] | null | undefined) {
  if (!source || !target) return null;
  if (source.length !== target.length) return false;
  return source.every((sourceDim, index) => {
    const targetDim = target[index];
    return (
      sourceDim === targetDim ||
      sourceDim === null ||
      targetDim === null ||
      sourceDim === undefined ||
      targetDim === undefined ||
      sourceDim === 'any' ||
      targetDim === 'any' ||
      sourceDim === '*' ||
      targetDim === '*' ||
      sourceDim === -1 ||
      targetDim === -1
    );
  });
}

function valueSpecShape(value: StudioValueSpec | null | undefined): unknown[] | null {
  return Array.isArray(value?.shape) ? value.shape : null;
}

function hasValue(spec: StudioInterventionTransformSpec): boolean {
  return spec.value !== undefined && spec.value !== null;
}

function hasNoiseScale(spec: StudioInterventionTransformSpec): boolean {
  const parameters = spec.parameters ?? {};
  return (
    hasValue(spec) ||
    parameters.scale !== undefined ||
    parameters.std !== undefined ||
    parameters.noise_std !== undefined
  );
}

function hasClampBound(spec: StudioInterventionTransformSpec): boolean {
  return Boolean(spec.bounds && ('min' in spec.bounds || 'max' in spec.bounds));
}

function numericCompatibilityIssues(
  operation: string,
  schema: ValueSchema,
  path: string,
  label: string
): StudioValidationIssue[] {
  const numeric = isNumericSchema(schema);
  if (numeric === false) {
    return [
      issue(
        'intervention_target_dtype_mismatch',
        `${operation} intervention target ${label} has non-numeric dtype ${schema.dtype}`,
        path
      ),
    ];
  }
  if (numeric === null || schema.origin === 'unknown') {
    return [
      issue(
        'intervention_target_unknown_schema',
        `${operation} intervention target ${label} cannot be fully checked from static schema data`,
        path,
        'warning'
      ),
    ];
  }
  return [];
}

export function validateInterventionSchema(
  taps: TapSpec[] | null | undefined,
  registry: Pick<StudioSchemaRegistry, 'selector_targets'>
): StudioValidationIssue[] {
  const issues: StudioValidationIssue[] = [];

  for (const [index, tap] of (taps ?? []).entries()) {
    if (tap.type !== 'intervention') continue;
    const path = `taps.${index}.transform.intervention`;
    const intervention = tap.transform?.intervention;
    if (!intervention) {
      issues.push(
        issue(
          'intervention_missing_spec',
          `Intervention tap ${tap.id} needs a typed target and operation`,
          path
        )
      );
      continue;
    }

    const operation = intervention.operation;
    if (!SUPPORTED_INTERVENTION_OPERATIONS.has(operation)) {
      issues.push(
        issue(
          'intervention_unknown_operation',
          `Intervention operation ${operation || 'None'} is not supported by schema validation`,
          `${path}.operation`
        )
      );
      continue;
    }

    const target = resolveSelectorTarget(intervention, registry);
    if (!target) {
      issues.push(
        issue(
          'intervention_unknown_target',
          `Intervention target ${selectorCompact(intervention) ?? 'None'} is not in the selector schema registry`,
          `${path}.target_selector`
        )
      );
      continue;
    }

    issues.push(
      ...numericCompatibilityIssues(operation, target.value_schema, path, target.label)
    );

    if (operation === 'clamp' && !hasClampBound(intervention)) {
      issues.push(
        issue(
          'intervention_missing_bounds',
          `Clamp intervention for ${target.label} needs at least one bound`,
          `${path}.bounds`
        )
      );
    }

    if ((operation === 'constant' || operation === 'offset' || operation === 'scale') && !hasValue(intervention)) {
      issues.push(
        issue(
          'intervention_missing_value',
          `${operation} intervention for ${target.label} needs a value spec`,
          `${path}.value`
        )
      );
    }

    if (operation === 'noise' && !hasNoiseScale(intervention)) {
      issues.push(
        issue(
          'intervention_missing_noise_scale',
          `Noise intervention for ${target.label} needs a scale or std value`,
          `${path}.value`
        )
      );
    }

    const valueShape = valueSpecShape(intervention.value);
    const compatible = shapeCompatible(valueShape, target.value_schema.shape);
    if (compatible === false) {
      issues.push(
        issue(
          'intervention_value_shape_mismatch',
          `Intervention value shape ${JSON.stringify(valueShape)} does not match ${target.label} shape ${JSON.stringify(target.value_schema.shape)}`,
          `${path}.value.shape`
        )
      );
    } else if (
      (operation === 'constant' || operation === 'offset') &&
      hasValue(intervention) &&
      compatible === null &&
      target.value_schema.shape
    ) {
      issues.push(
        issue(
          'intervention_value_unknown_shape',
          `Intervention value for ${target.label} cannot be shape-checked from static schema data`,
          `${path}.value`,
          'warning'
        )
      );
    }
  }

  return issues;
}
