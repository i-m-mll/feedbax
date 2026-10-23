import { useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import { createPortal } from 'react-dom';
import { RotateCcw, SlidersHorizontal, X } from 'lucide-react';
import {
  LEGACY_FRONTEND_VALUE_SPEC_SCHEMA_V1,
  LEGACY_VALUE_SPEC_SCHEMA_V1,
  VALUE_SPEC_DISTRIBUTIONS,
  VALUE_SPEC_FUNCTION_TEMPLATES,
  VALUE_SPEC_MODE_OPTIONS,
  VALUE_SPEC_SCOPE_OPTIONS,
  VALUE_SPEC_SCHEMA_VERSION,
  normalizeStudioValueSpec,
  setValueSpecDistributionFamily,
  setValueSpecDistributionParameter,
  setValueSpecEnumerable,
  setValueSpecFunction,
  setValueSpecFunctionParameter,
  setValueSpecMode,
  setValueSpecScope,
  setValueSpecConstantValue,
  valueSpecEnumerableCount,
  valueSpecChipLabel,
  valueSpecValidationErrors,
  type StudioValueSpecEnumerable,
  type StudioValueSpecVariationScope,
} from '@/features/scenario/valueSpecs';
import type {
  StudioTaskTimelineSignalSpec,
  StudioValueSpec,
  StudioValueSpecMode,
  StudioValueSpecSamplingScope,
} from '@/types/workspace';

const VALUE_SPEC_POPOVER_WIDTH_PX = 640;
const VALUE_SPEC_POPOVER_MARGIN_PX = 12;

type OwnerKind =
  | 'task_signal'
  | 'epoch_length'
  | 'task_param'
  | 'component_param'
  | 'objective'
  | 'intervention';

type SemanticKind =
  | 'task_signal'
  | 'protocol_target'
  | 'epoch_length'
  | 'static_shape'
  | 'static_leaf'
  | 'state_init';

export interface ValueSpecFieldDescriptor {
  id: string;
  label: string;
  ownerKind: OwnerKind;
  semanticKind: SemanticKind;
  valueSchema?: Record<string, unknown> | null;
  allowedModes: StudioValueSpecMode[];
  allowedScopes?: StudioValueSpecSamplingScope[];
  defaultScope?: StudioValueSpecSamplingScope | null;
  functionIds?: string[];
  loweringTarget?: string;
  disabled?: boolean;
}

interface ValueSpecFieldProps {
  descriptor: ValueSpecFieldDescriptor;
  value: unknown;
  onChange: (value: unknown) => void;
  forceValueSpec?: boolean;
  compact?: boolean;
}

export function isStudioValueSpec(value: unknown): value is StudioValueSpec {
  const version = (value as { schema_version?: unknown } | null)?.schema_version;
  return (
    value !== null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    (version === VALUE_SPEC_SCHEMA_VERSION ||
      version === LEGACY_VALUE_SPEC_SCHEMA_V1 ||
      version === LEGACY_FRONTEND_VALUE_SPEC_SCHEMA_V1) &&
    typeof (value as { mode?: unknown }).mode === 'string' &&
    typeof (value as { metadata?: unknown }).metadata === 'object'
  );
}

export function literalFromValueSpec(value: unknown): unknown {
  if (!isStudioValueSpec(value)) return value;
  const spec = normalizeStudioValueSpec(value);
  return spec.mode === 'constant' && spec.variation?.scope !== 'sweep' ? spec.value : spec;
}

function formatValue(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value === null || value === undefined) return '';
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function humanizeLabel(label: string): string {
  return label
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function displayTextForValue(value: unknown): string {
  if (!isStudioValueSpec(value)) return formatValue(value);
  if (value.mode === 'constant') return formatValue(value.value);
  const distribution = value.distribution as Record<string, unknown> | null | undefined;
  const parameters = distribution?.parameters as Record<string, unknown> | undefined;
  if (value.mode === 'distribution' && distribution?.family === 'uniform' && parameters) {
    const min = parameters.min;
    const max = parameters.max;
    if (min !== undefined && max !== undefined) return `${min}-${max}`;
  }
  return valueSpecChipLabel(value);
}

function parseInlineText(rawValue: string, currentValue: unknown): unknown {
  const trimmed = rawValue.trim();
  if (isStudioValueSpec(currentValue)) {
    const distribution = currentValue.distribution as Record<string, unknown> | null | undefined;
    const parameters = distribution?.parameters as Record<string, unknown> | undefined;
    if (currentValue.mode === 'distribution' && distribution?.family === 'uniform' && parameters) {
      const match = trimmed.match(/^(-?\d+(?:\.\d+)?)\s*(?:-|,|:|\.\.)\s*(-?\d+(?:\.\d+)?)$/);
      if (match) {
        return {
          ...currentValue,
          distribution: {
            ...distribution,
            parameters: {
              ...parameters,
              min: Number(match[1]),
              max: Number(match[2]),
            },
          },
        };
      }
      return currentValue;
    }
    if (currentValue.mode === 'constant') {
      return setValueSpecConstantValue(currentValue, parseStructuredInput(trimmed, currentValue.value));
    }
    return currentValue;
  }
  return parseStructuredInput(trimmed, currentValue);
}

function parseScalarInput(rawValue: string, currentValue: unknown): unknown {
  if (typeof currentValue === 'number') {
    const parsed = Number.parseFloat(rawValue);
    return Number.isFinite(parsed) ? parsed : currentValue;
  }
  if (typeof currentValue === 'boolean') return rawValue === 'true';
  if (currentValue === null || currentValue === undefined) {
    const trimmed = rawValue.trim();
    const parsed = Number.parseFloat(trimmed);
    return Number.isFinite(parsed) && String(parsed) === trimmed ? parsed : rawValue;
  }
  return rawValue;
}

function parseStructuredInput(rawValue: string, currentValue: unknown): unknown {
  if (Array.isArray(currentValue) || isRecord(currentValue)) {
    try {
      return JSON.parse(rawValue);
    } catch {
      return currentValue;
    }
  }
  return parseScalarInput(rawValue, currentValue);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

const VALUE_INPUT_ROW_CLASS = 'grid grid-cols-[5.5rem_minmax(0,1fr)] items-center gap-2 text-xs';
const VALUE_INPUT_LABEL_CLASS = 'truncate text-right font-medium text-slate-500';
const VALUE_INPUT_CLASS = 'h-8 min-w-0 rounded border border-slate-200 bg-white px-2 text-xs text-slate-700';

function defaultLiteralForDescriptor(descriptor: ValueSpecFieldDescriptor, fallback: unknown) {
  if (fallback !== undefined) return fallback;
  if (descriptor.semanticKind === 'epoch_length') return { min: 0, max: 1 };
  if (descriptor.valueSchema?.shape && Array.isArray(descriptor.valueSchema.shape)) {
    const trailing = descriptor.valueSchema.shape.at(-1);
    if (typeof trailing === 'number' && trailing > 1) {
      return Array.from({ length: trailing }, () => 0);
    }
  }
  return 0;
}

function createValueSpec(
  descriptor: ValueSpecFieldDescriptor,
  value: unknown,
  mode: StudioValueSpecMode = 'constant'
): StudioValueSpec {
  const schema = descriptor.valueSchema ?? {};
  const literal = isStudioValueSpec(value) ? value.value : value;
  return setValueSpecMode(
    {
      schema_version: VALUE_SPEC_SCHEMA_VERSION,
      value_form: 'literal',
      variation: { scope: 'fixed', enumerable: null, metadata: {} },
      mode: 'constant',
      value: defaultLiteralForDescriptor(descriptor, literal),
      dtype: typeof schema.dtype === 'string' ? schema.dtype : null,
      shape: Array.isArray(schema.shape) ? schema.shape : null,
      units: typeof schema.units === 'string' ? schema.units : null,
      frame: typeof schema.frame === 'string' ? schema.frame : null,
      metadata: {
        value_field_id: descriptor.id,
        value_field_owner_kind: descriptor.ownerKind,
        value_field_semantic_kind: descriptor.semanticKind,
        lowering_target: descriptor.loweringTarget ?? null,
        value_schema: descriptor.valueSchema ?? null,
      },
    },
    mode
  );
}

function commitValueSpec(
  valueSpec: StudioValueSpec,
  forceValueSpec: boolean,
  descriptor: ValueSpecFieldDescriptor
) {
  const metadata = {
    ...valueSpec.metadata,
    value_field_id: descriptor.id,
    value_field_owner_kind: descriptor.ownerKind,
    value_field_semantic_kind: descriptor.semanticKind,
    lowering_target: descriptor.loweringTarget ?? null,
  };
  const next = normalizeStudioValueSpec({ ...valueSpec, metadata });
  const errors = valueSpecValidationErrors(next, descriptor.allowedScopes);
  if (errors.length > 0) return valueSpec;
  if (!forceValueSpec && next.mode === 'constant' && next.variation.scope !== 'sweep') {
    return next.value;
  }
  return next;
}

function StructuredValueInput({
  value,
  onChange,
  label = 'Literal',
}: {
  value: unknown;
  onChange: (value: unknown) => void;
  label?: string;
}) {
  if (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((item) => typeof item === 'number')
  ) {
    return (
      <div className="grid grid-cols-2 gap-2">
        {value.map((item, index) => (
          <label key={index} className={VALUE_INPUT_ROW_CLASS}>
            <span className={VALUE_INPUT_LABEL_CLASS}>{index}</span>
            <input
              type="number"
              value={item}
              step="any"
              onChange={(event) => {
                const next = [...value];
                next[index] = Number(event.target.value);
                onChange(next);
              }}
              className={VALUE_INPUT_CLASS}
            />
          </label>
        ))}
      </div>
    );
  }
  if (isRecord(value) && ('active' in value || 'inactive' in value)) {
    return (
      <div className="grid grid-cols-2 gap-2">
        {(['inactive', 'active'] as const).map((key) => (
          <label key={key} className={VALUE_INPUT_ROW_CLASS}>
            <span className={VALUE_INPUT_LABEL_CLASS}>
              {key === 'inactive' ? 'Inactive' : 'Active'}
            </span>
            <input
              type={typeof value[key] === 'number' ? 'number' : 'text'}
              value={formatValue(value[key] ?? (key === 'active' ? 1 : 0))}
              step={typeof value[key] === 'number' ? 'any' : undefined}
              onChange={(event) =>
                onChange({ ...value, [key]: parseStructuredInput(event.target.value, value[key]) })
              }
              className={VALUE_INPUT_CLASS}
            />
          </label>
        ))}
      </div>
    );
  }
  if (Array.isArray(value) || isRecord(value)) {
    return (
      <label className="grid grid-cols-[5.5rem_minmax(0,1fr)] items-start gap-2 text-xs">
        <span className={`${VALUE_INPUT_LABEL_CLASS} pt-1.5`}>{humanizeLabel(label)}</span>
        <textarea
          value={formatValue(value)}
          onChange={(event) => onChange(parseStructuredInput(event.target.value, value))}
          className="min-h-16 min-w-0 rounded border border-slate-200 bg-white px-2 py-1.5 font-mono text-[11px] text-slate-700"
        />
      </label>
    );
  }
  if (typeof value === 'boolean') {
    return (
      <label className={VALUE_INPUT_ROW_CLASS}>
        <span className={VALUE_INPUT_LABEL_CLASS}>{humanizeLabel(label)}</span>
        <input
          type="checkbox"
          checked={value}
          onChange={(event) => onChange(event.target.checked)}
          className="h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
        />
      </label>
    );
  }
  return (
    <label className={VALUE_INPUT_ROW_CLASS}>
      <span className={VALUE_INPUT_LABEL_CLASS}>{humanizeLabel(label)}</span>
      <input
        type={typeof value === 'number' ? 'number' : 'text'}
        value={formatValue(value ?? '')}
        step={typeof value === 'number' ? 'any' : undefined}
        onChange={(event) => onChange(parseScalarInput(event.target.value, value))}
        className={VALUE_INPUT_CLASS}
      />
    </label>
  );
}

function ParameterEditor({
  parameters,
  onChange,
}: {
  parameters: Record<string, unknown>;
  onChange: (name: string, value: unknown) => void;
}) {
  return (
    <div className="grid gap-2">
      {Object.entries(parameters).map(([name, value]) => (
        <StructuredValueInput
          key={name}
          label={name}
          value={value}
          onChange={(nextValue) => onChange(name, nextValue)}
        />
      ))}
    </div>
  );
}

function ValueSpecPreview({ valueSpec }: { valueSpec: StudioValueSpec }) {
  valueSpec = normalizeStudioValueSpec(valueSpec);
  if (valueSpec.variation?.scope === 'sweep') {
    const enumerable = valueSpec.variation.enumerable;
    const count = valueSpecEnumerableCount(valueSpec);
    const ticks =
      enumerable?.form === 'range' && typeof enumerable.count === 'number'
        ? Array.from({ length: Math.min(enumerable.count, 8) }, (_, index) => index)
        : Array.from({ length: Math.min(count ?? 4, 8) }, (_, index) => index);
    return (
      <div className="space-y-2">
        <div className="flex h-9 items-end gap-1">
          {ticks.map((_, index) => (
            <span
              key={index}
              className="block flex-1 rounded-sm bg-amber-400/80"
              style={{ height: `${35 + ((index * 17) % 55)}%` }}
            />
          ))}
        </div>
        <div className="text-[10px] font-medium text-amber-700">
          {count === null ? 'Axis ready' : `${count} axis value${count === 1 ? '' : 's'}`}
        </div>
      </div>
    );
  }
  if (valueSpec.mode === 'distribution') {
    return (
      <div className="grid grid-cols-8 gap-1">
        {Array.from({ length: 8 }, (_, index) => (
          <span
            key={index}
            className="block rounded-sm bg-emerald-100"
            style={{ height: `${8 + ((index * 7) % 18)}px` }}
          />
        ))}
      </div>
    );
  }
  if (valueSpec.mode === 'function' || valueSpec.mode === 'schedule') {
    return (
      <div className="flex h-9 items-end gap-1">
        {[0.15, 0.2, 0.35, 0.55, 0.75, 0.9, 0.9, 0.9].map((height, index) => (
          <span
            key={index}
            className="block flex-1 rounded-sm bg-emerald-500/70"
            style={{ height: `${height * 100}%` }}
          />
        ))}
      </div>
    );
  }
  return (
    <div className="h-9 rounded-sm bg-slate-100 px-2 py-2 font-mono text-[11px] text-slate-500">
      {valueSpecChipLabel(valueSpec)}
    </div>
  );
}

function popoverStyle(anchorRect: DOMRect | null): CSSProperties {
  if (typeof window === 'undefined') return {};
  const width = Math.min(VALUE_SPEC_POPOVER_WIDTH_PX, window.innerWidth - VALUE_SPEC_POPOVER_MARGIN_PX * 2);
  const left = Math.min(
    Math.max(VALUE_SPEC_POPOVER_MARGIN_PX, anchorRect?.left ?? VALUE_SPEC_POPOVER_MARGIN_PX),
    window.innerWidth - width - VALUE_SPEC_POPOVER_MARGIN_PX
  );
  const below = (anchorRect?.bottom ?? 80) + 8;
  const maxHeight = window.innerHeight - below - VALUE_SPEC_POPOVER_MARGIN_PX;
  if (maxHeight >= 360) {
    return { left, top: below, width, maxHeight };
  }
  const height = Math.min(560, window.innerHeight - VALUE_SPEC_POPOVER_MARGIN_PX * 2);
  const top = Math.max(VALUE_SPEC_POPOVER_MARGIN_PX, (anchorRect?.top ?? 80) - height - 8);
  return { left, top, width, maxHeight: height };
}

function EnumerableEditor({
  valueSpec,
  onChange,
}: {
  valueSpec: StudioValueSpec;
  onChange: (valueSpec: StudioValueSpec) => void;
}) {
  const spec = normalizeStudioValueSpec(valueSpec);
  const enumerable =
    spec.variation?.enumerable ??
    ({
      form: 'list',
      values: [spec.value ?? 0],
    } satisfies StudioValueSpecEnumerable);
  const setEnumerable = (next: StudioValueSpecEnumerable) => onChange(setValueSpecEnumerable(spec, next));
  const values = enumerable.form === 'list' && Array.isArray(enumerable.values) ? enumerable.values : [0];
  return (
    <div className="rounded border border-amber-200 bg-amber-50/70 p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="text-xs font-semibold text-amber-800">Sweep axis</div>
        <select
          value={enumerable.form}
          onChange={(event) => {
            const form = event.target.value as StudioValueSpecEnumerable['form'];
            if (form === 'range') {
              setEnumerable({ form, start: 0, stop: 1, count: 5, scale: 'linear' });
            } else if (form === 'sampler') {
              setEnumerable({
                form,
                sampler: spec.distribution ?? { family: 'uniform', parameters: { min: 0, max: 1 } },
                n: 5,
              });
            } else {
              setEnumerable({ form, values });
            }
          }}
          className="h-7 rounded border border-amber-200 bg-white px-2 text-xs text-amber-900"
        >
          <option value="list">List</option>
          <option value="range">Range</option>
          <option value="sampler">Sampler</option>
        </select>
      </div>
      {enumerable.form === 'list' && (
        <div className="space-y-2">
          {values.map((item, index) => (
            <div key={index} className="grid grid-cols-[minmax(0,1fr)_2rem] gap-2">
              <StructuredValueInput
                label={`Value ${index + 1}`}
                value={item}
                onChange={(nextValue) => {
                  const nextValues = [...values];
                  nextValues[index] = nextValue;
                  setEnumerable({ form: 'list', values: nextValues });
                }}
              />
              <button
                type="button"
                onClick={() =>
                  setEnumerable({
                    form: 'list',
                    values: values.filter((_, valueIndex) => valueIndex !== index),
                  })
                }
                className="h-8 rounded border border-amber-200 bg-white text-xs text-amber-700"
                title="Remove value"
              >
                <X className="mx-auto h-3.5 w-3.5" />
              </button>
            </div>
          ))}
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setEnumerable({ form: 'list', values: [...values, values.at(-1) ?? 0] })}
              className="rounded border border-amber-300 bg-white px-2 py-1 text-xs font-medium text-amber-800"
            >
              Add value
            </button>
            <input
              placeholder="Paste CSV"
              onBlur={(event) => {
                const parsed = event.currentTarget.value
                  .split(',')
                  .map((item) => item.trim())
                  .filter(Boolean)
                  .map((item) => {
                    const numeric = Number(item);
                    return Number.isFinite(numeric) ? numeric : item;
                  });
                if (parsed.length > 0) {
                  setEnumerable({ form: 'list', values: parsed });
                  event.currentTarget.value = '';
                }
              }}
              className="h-7 min-w-0 flex-1 rounded border border-amber-200 bg-white px-2 text-xs text-amber-900"
            />
          </div>
        </div>
      )}
      {enumerable.form === 'range' && (
        <div className="grid grid-cols-2 gap-2">
          {(['start', 'stop', 'count'] as const).map((key) => (
            <label key={key} className="grid gap-1 text-xs text-amber-800">
              <span>{humanizeLabel(key)}</span>
              <input
                type="number"
                step={key === 'count' ? 1 : 'any'}
                min={key === 'count' ? 1 : undefined}
                value={Number(enumerable[key] ?? (key === 'count' ? 5 : 0))}
                onChange={(event) =>
                  setEnumerable({
                    ...enumerable,
                    form: 'range',
                    [key]: key === 'count' ? Math.max(1, Math.round(Number(event.target.value))) : Number(event.target.value),
                  })
                }
                className="h-8 rounded border border-amber-200 bg-white px-2 text-xs text-amber-900"
              />
            </label>
          ))}
          <label className="grid gap-1 text-xs text-amber-800">
            <span>Scale</span>
            <select
              value={enumerable.scale ?? 'linear'}
              onChange={(event) =>
                setEnumerable({
                  ...enumerable,
                  form: 'range',
                  scale: event.target.value as 'linear' | 'log',
                })
              }
              className="h-8 rounded border border-amber-200 bg-white px-2 text-xs text-amber-900"
            >
              <option value="linear">Linear</option>
              <option value="log">Log</option>
            </select>
          </label>
        </div>
      )}
      {enumerable.form === 'sampler' && (
        <div className="space-y-2">
          <label className="grid gap-1 text-xs text-amber-800">
            <span>Samples</span>
            <input
              type="number"
              min={1}
              step={1}
              value={enumerable.n ?? 5}
              onChange={(event) =>
                setEnumerable({ ...enumerable, form: 'sampler', n: Math.max(1, Math.round(Number(event.target.value))) })
              }
              className="h-8 rounded border border-amber-200 bg-white px-2 text-xs text-amber-900"
            />
          </label>
          <div className="rounded bg-white/80 px-2 py-1.5 text-[11px] text-amber-800">
            Sampler uses the current distribution payload and emits `n` axis values for P2 expansion.
          </div>
        </div>
      )}
    </div>
  );
}

function SchedulePointsEditor({
  schedule,
  fallback,
  onChange,
}: {
  schedule: Record<string, unknown>;
  fallback: unknown;
  onChange: (schedule: Record<string, unknown>) => void;
}) {
  const points = Array.isArray(schedule.points) ? schedule.points : [[0, fallback ?? 0]];
  return (
    <div className="space-y-2">
      {points.map((point, index) => {
        const tuple = Array.isArray(point) ? point : [index, point];
        return (
          <div key={index} className="grid grid-cols-[5rem_minmax(0,1fr)_2rem] gap-2">
            <input
              type="number"
              step="any"
              value={Number(tuple[0] ?? index)}
              onChange={(event) => {
                const next = [...points];
                next[index] = [Number(event.target.value), tuple[1] ?? fallback ?? 0];
                onChange({ ...schedule, points: next });
              }}
              className={VALUE_INPUT_CLASS}
              aria-label={`Schedule point ${index + 1} coordinate`}
            />
            <StructuredValueInput
              label={`Point ${index + 1}`}
              value={tuple[1] ?? fallback ?? 0}
              onChange={(nextValue) => {
                const next = [...points];
                next[index] = [tuple[0] ?? index, nextValue];
                onChange({ ...schedule, points: next });
              }}
            />
            <button
              type="button"
              onClick={() => onChange({ ...schedule, points: points.filter((_, pointIndex) => pointIndex !== index) })}
              className="h-8 rounded border border-slate-200 bg-white text-slate-500"
              title="Remove point"
            >
              <X className="mx-auto h-3.5 w-3.5" />
            </button>
          </div>
        );
      })}
      <button
        type="button"
        onClick={() => onChange({ ...schedule, points: [...points, [points.length, fallback ?? 0]] })}
        className="rounded border border-slate-200 bg-white px-2 py-1 text-xs font-medium text-slate-600"
      >
        Add point
      </button>
    </div>
  );
}

function ValueSpecModal({
  descriptor,
  initialValue,
  anchorRect,
  forceValueSpec,
  onCancel,
  onCommit,
}: {
  descriptor: ValueSpecFieldDescriptor;
  initialValue: unknown;
  anchorRect: DOMRect | null;
  forceValueSpec: boolean;
  onCancel: () => void;
  onCommit: (value: unknown) => void;
}) {
  const initialMode = isStudioValueSpec(initialValue)
    ? normalizeStudioValueSpec(initialValue).mode
    : descriptor.allowedModes[0] ?? 'constant';
  const [activeMode, setActiveMode] = useState<StudioValueSpecMode>(initialMode);
  const [draftsByMode, setDraftsByMode] = useState<Record<string, StudioValueSpec>>(() => {
    const seed = isStudioValueSpec(initialValue)
      ? normalizeStudioValueSpec(initialValue)
      : createValueSpec(descriptor, initialValue, initialMode);
    return { [seed.mode]: seed };
  });
  const draft =
    draftsByMode[activeMode] ??
    createValueSpec(descriptor, isStudioValueSpec(initialValue) ? initialValue.value : initialValue, activeMode);
  const scopes = (descriptor.allowedScopes ?? ['run', 'sweep']) as StudioValueSpecVariationScope[];
  const functionTemplates = VALUE_SPEC_FUNCTION_TEMPLATES.filter((template) => {
    if (descriptor.functionIds && !descriptor.functionIds.includes(template.id)) return false;
    if (descriptor.allowedScopes && !template.domains.some((domain) => descriptor.allowedScopes!.includes(domain))) {
      return false;
    }
    return true;
  });
  const activeTemplate =
    functionTemplates.find((template) => template.id === draft.function_id) ?? functionTemplates[0];
  const functionParameters = draft.parameters ?? activeTemplate?.defaultParameters ?? {};
  const distributionParameters =
    (draft.distribution?.parameters as Record<string, unknown> | undefined) ?? {};
  const schedule = draft.schedule ?? {};
  const validationErrors = valueSpecValidationErrors(draft, scopes);
  const variationScope = normalizeStudioValueSpec(draft).variation.scope;
  const enumerableCount = valueSpecEnumerableCount(draft);
  const updateDraft = (next: StudioValueSpec) => {
    setDraftsByMode((current) => ({ ...current, [next.mode]: next }));
  };
  const switchMode = (mode: StudioValueSpecMode) => {
    setDraftsByMode((current) => {
      if (current[mode]) return { ...current, [activeMode]: draft };
      return { ...current, [activeMode]: draft, [mode]: setValueSpecMode(draft, mode) };
    });
    setActiveMode(mode);
  };
  const resetDrafts = () => {
    const seed = isStudioValueSpec(initialValue)
      ? normalizeStudioValueSpec(initialValue)
      : createValueSpec(descriptor, initialValue, initialMode);
    setActiveMode(seed.mode);
    setDraftsByMode({ [seed.mode]: seed });
  };
  if (typeof document === 'undefined') return null;
  return createPortal(
    <div className="fixed inset-0 z-50">
      <button
        type="button"
        className="absolute inset-0 cursor-default bg-slate-900/30 backdrop-blur-[1px]"
        onClick={onCancel}
        aria-label="Cancel value editor"
      />
      <div
        className="absolute overflow-y-auto rounded border border-slate-200 bg-white p-4 shadow-xl"
        style={popoverStyle(anchorRect)}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0 truncate text-sm font-semibold text-slate-800">
            {descriptor.label}
          </div>
          <button
            type="button"
            onClick={onCancel}
            className="inline-flex h-8 w-8 items-center justify-center rounded text-slate-500 hover:bg-slate-100"
            title="Cancel"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-1">
          {VALUE_SPEC_MODE_OPTIONS.filter((option) => descriptor.allowedModes.includes(option.value)).map(
            (option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => switchMode(option.value)}
                className={
                  activeMode === option.value
                    ? 'rounded border border-emerald-300 bg-emerald-50 px-2.5 py-1.5 text-xs font-medium text-emerald-700'
                    : 'rounded border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-500 hover:border-emerald-200'
                }
              >
                {option.label}
              </button>
            )
          )}
        </div>
        <div className="mt-3 grid gap-4 lg:grid-cols-[minmax(0,1fr)_11rem]">
          <div className="min-w-0 space-y-3">
            {draft.mode === 'constant' && (
              <StructuredValueInput
                value={draft.value}
                onChange={(nextValue) => updateDraft(setValueSpecConstantValue(draft, nextValue))}
              />
            )}
            {draft.mode === 'function' && activeTemplate && (
              <div className="space-y-3">
                <label className="grid gap-1 text-xs text-slate-500">
                  <span>Function</span>
                  <select
                    value={draft.function_id ?? activeTemplate.id}
                    onChange={(event) => updateDraft(setValueSpecFunction(draft, event.target.value))}
                    className="h-8 rounded border border-slate-200 bg-white px-2 text-xs text-slate-700"
                  >
                    {functionTemplates.map((template) => (
                      <option key={template.id} value={template.id}>
                        {template.label}
                      </option>
                    ))}
                  </select>
                </label>
                <ParameterEditor
                  parameters={functionParameters}
                  onChange={(name, nextValue) =>
                    updateDraft(setValueSpecFunctionParameter(draft, name, nextValue))
                  }
                />
              </div>
            )}
            {draft.mode === 'distribution' && (
              <div className="space-y-3">
                <label className="grid gap-1 text-xs text-slate-500">
                  <span>Distribution</span>
                  <select
                    value={String(draft.distribution?.family ?? 'uniform')}
                    onChange={(event) =>
                      updateDraft(setValueSpecDistributionFamily(draft, event.target.value))
                    }
                    className="h-8 rounded border border-slate-200 bg-white px-2 text-xs text-slate-700"
                  >
                    {VALUE_SPEC_DISTRIBUTIONS.map((family) => (
                      <option key={family} value={family}>
                        {family}
                      </option>
                    ))}
                  </select>
                </label>
                <ParameterEditor
                  parameters={distributionParameters}
                  onChange={(name, nextValue) =>
                    updateDraft(setValueSpecDistributionParameter(draft, name, nextValue))
                  }
                />
              </div>
            )}
            {draft.mode === 'schedule' && (
              <div className="space-y-3">
                <label className="grid gap-1 text-xs text-slate-500">
                  <span>Domain</span>
                  <select
                    value={String(schedule.domain ?? descriptor.defaultScope ?? 'epoch')}
                    onChange={(event) =>
                      updateDraft({
                        ...draft,
                        schedule: { ...schedule, domain: event.target.value },
                      })
                    }
                    className="h-8 rounded border border-slate-200 bg-white px-2 text-xs text-slate-700"
                  >
                    {VALUE_SPEC_SCOPE_OPTIONS.filter((option) =>
                      ['trial', 'epoch', 'timestep', 'run', 'sweep'].includes(option.value)
                    ).map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="grid gap-1 text-xs text-slate-500">
                  <span>Points</span>
                  <SchedulePointsEditor
                    schedule={schedule}
                    fallback={draft.value}
                    onChange={(nextSchedule) =>
                      updateDraft({
                        ...draft,
                        schedule: nextSchedule,
                      })
                    }
                  />
                </label>
              </div>
            )}
            {draft.mode === 'expression' && (
              <label className="grid gap-1 text-xs text-slate-500">
                <span>Expression</span>
                <input
                  value={draft.expression ?? ''}
                  onChange={(event) => updateDraft({ ...draft, expression: event.target.value })}
                  className="h-8 rounded border border-slate-200 bg-white px-2 font-mono text-xs text-slate-700"
                />
              </label>
            )}
            {scopes.length > 0 && (
              <label className="grid gap-1 text-xs text-slate-500">
                <span>Variation</span>
                <select
                  value={variationScope}
                  onChange={(event) =>
                    updateDraft(setValueSpecScope(draft, event.target.value))
                  }
                  className="h-8 rounded border border-slate-200 bg-white px-2 text-xs text-slate-700"
                >
                  {VALUE_SPEC_SCOPE_OPTIONS.filter((option) => scopes.includes(option.value)).map(
                    (option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    )
                  )}
                </select>
              </label>
            )}
            {variationScope === 'sweep' && (
              <EnumerableEditor valueSpec={draft} onChange={updateDraft} />
            )}
          </div>
          <div className="rounded border border-slate-200 bg-slate-50/70 p-3">
            <ValueSpecPreview valueSpec={draft} />
            <div className="mt-2 truncate text-[11px] font-medium text-slate-500">
              {valueSpecChipLabel(draft)}
            </div>
            {variationScope === 'sweep' && (
              <div className="mt-2 rounded bg-amber-50 px-2 py-1.5 text-[11px] text-amber-800">
                Adds an axis to the Train matrix
                {enumerableCount === null
                  ? '.'
                  : ` with ${enumerableCount} value${enumerableCount === 1 ? '' : 's'}.`}
              </div>
            )}
            {variationScope === 'replicate' && (
              <div className="mt-2 rounded bg-slate-100 px-2 py-1.5 text-[11px] text-slate-600">
                Replicate variation resamples separately for each replicate.
              </div>
            )}
            {variationScope === 'run' && draft.mode === 'distribution' && (
              <div className="mt-2 rounded bg-slate-100 px-2 py-1.5 text-[11px] text-slate-600">
                Run variation samples once and shares the value across replicates.
              </div>
            )}
            {validationErrors.length > 0 && (
              <div className="mt-2 rounded bg-rose-50 px-2 py-1.5 text-[11px] text-rose-700">
                {validationErrors.join('. ')}
              </div>
            )}
          </div>
        </div>
        <div className="mt-4 flex justify-between gap-2">
          <button
            type="button"
            onClick={resetDrafts}
            className="inline-flex items-center gap-1 rounded border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Revert
          </button>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={onCancel}
              className="rounded border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={validationErrors.length > 0}
              onClick={() => onCommit(commitValueSpec(draft, forceValueSpec, descriptor))}
              className="rounded border border-emerald-600 bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:border-slate-300 disabled:bg-slate-300"
            >
              Done
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}

export function ValueSpecField({
  descriptor,
  value,
  onChange,
  forceValueSpec = false,
  compact = false,
}: ValueSpecFieldProps) {
  const [open, setOpen] = useState(false);
  const [anchorRect, setAnchorRect] = useState<DOMRect | null>(null);
  const fieldRef = useRef<HTMLDivElement | null>(null);
  const [localText, setLocalText] = useState(() => displayTextForValue(value));
  useEffect(() => {
    setLocalText(displayTextForValue(value));
  }, [value]);
  useEffect(() => {
    if (!open) return;
    const updateAnchor = () => setAnchorRect(fieldRef.current?.getBoundingClientRect() ?? null);
    updateAnchor();
    window.addEventListener('resize', updateAnchor);
    window.addEventListener('scroll', updateAnchor, true);
    return () => {
      window.removeEventListener('resize', updateAnchor);
      window.removeEventListener('scroll', updateAnchor, true);
    };
  }, [open]);
  const commitLocalText = () => {
    const nextValue = parseInlineText(localText, value);
    if (nextValue !== value) onChange(nextValue);
    setLocalText(displayTextForValue(nextValue));
  };
  return (
    <>
      <div
        ref={fieldRef}
        className={
          compact
            ? 'flex h-6 w-full min-w-0 items-center overflow-hidden rounded border border-slate-200 bg-white text-[10px] text-slate-700 focus-within:border-emerald-300'
            : 'flex min-h-8 w-full min-w-0 items-center overflow-hidden rounded-lg border border-slate-200 bg-white text-sm text-slate-800 focus-within:border-emerald-300'
        }
      >
        <input
          value={localText}
          disabled={descriptor.disabled}
          onChange={(event) => setLocalText(event.target.value)}
          onBlur={commitLocalText}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.currentTarget.blur();
            }
            if (event.key === 'Escape') {
              setLocalText(displayTextForValue(value));
              event.currentTarget.blur();
            }
          }}
          className={
            compact
              ? 'h-full min-w-0 flex-1 bg-transparent px-1.5 text-center font-medium outline-none disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-slate-400'
              : 'min-h-8 min-w-0 flex-1 bg-transparent px-3 py-2 outline-none disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-slate-400'
          }
          aria-label={humanizeLabel(descriptor.label)}
        />
        <button
          type="button"
          disabled={descriptor.disabled}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => {
            setAnchorRect(fieldRef.current?.getBoundingClientRect() ?? null);
            setOpen(true);
          }}
          className={
            compact
              ? 'inline-flex h-full w-6 shrink-0 items-center justify-center border-l border-slate-100 text-slate-500 hover:bg-slate-50 hover:text-slate-800 disabled:cursor-not-allowed disabled:text-slate-300'
              : 'inline-flex self-stretch w-9 shrink-0 items-center justify-center border-l border-slate-100 text-slate-500 hover:bg-slate-50 hover:text-slate-800 disabled:cursor-not-allowed disabled:text-slate-300'
          }
          title={`Edit ${humanizeLabel(descriptor.label)} value spec`}
        >
          <SlidersHorizontal className={compact ? 'h-3 w-3' : 'h-3.5 w-3.5'} />
        </button>
      </div>
      {open && (
        <ValueSpecModal
          descriptor={descriptor}
          initialValue={value}
          anchorRect={anchorRect}
          forceValueSpec={forceValueSpec}
          onCancel={() => setOpen(false)}
          onCommit={(nextValue) => {
            onChange(nextValue);
            setOpen(false);
          }}
        />
      )}
    </>
  );
}

export function descriptorForTaskSignal(
  signal: StudioTaskTimelineSignalSpec
): ValueSpecFieldDescriptor {
  const valueSchema = isRecord(signal.metadata.value_schema)
    ? (signal.metadata.value_schema as Record<string, unknown>)
    : null;
  const scopes = Array.isArray(signal.metadata.value_spec_scopes)
    ? signal.metadata.value_spec_scopes.filter((scope): scope is StudioValueSpecSamplingScope =>
        ['fixed', 'snapshot', 'run', 'replicate', 'trial', 'epoch', 'timestep', 'sweep'].includes(
          String(scope)
        )
      )
    : ['fixed', 'run', 'sweep', 'trial', 'epoch', 'timestep'];
  const modes = Array.isArray(signal.metadata.value_spec_modes)
    ? signal.metadata.value_spec_modes.filter((mode): mode is StudioValueSpecMode =>
        ['constant', 'reference', 'expression', 'function', 'distribution', 'schedule'].includes(
          String(mode)
        )
      )
    : ['constant', 'expression', 'function', 'distribution'];
  return {
    id: `task_signal:${signal.id}`,
    label: signal.label,
    ownerKind: 'task_signal',
    semanticKind: signal.kind === 'target' ? 'protocol_target' : 'task_signal',
    valueSchema,
    allowedModes: modes,
    allowedScopes: scopes,
    defaultScope: scopes.includes('trial') ? 'trial' : scopes[0] ?? 'run',
    functionIds:
      signal.id === 'target_position'
        ? ['delayed_reach_target_position', 'ramp', 'piecewise_linear']
        : signal.id === 'movement_target'
          ? ['delayed_reach_movement_target', 'ramp', 'piecewise_linear']
          : ['step', 'pulse', 'piecewise_linear'],
    loweringTarget: signal.kind === 'target' ? 'trial_data' : 'timestep_signal',
  };
}
