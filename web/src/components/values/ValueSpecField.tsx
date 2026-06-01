import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { RotateCcw, SlidersHorizontal, X } from 'lucide-react';
import { DIVIDER_HEIGHT, useLayoutStore } from '@/stores/layoutStore';
import {
  VALUE_SPEC_DISTRIBUTIONS,
  VALUE_SPEC_FUNCTION_TEMPLATES,
  VALUE_SPEC_MODE_OPTIONS,
  VALUE_SPEC_SCOPE_OPTIONS,
  setValueSpecDistributionFamily,
  setValueSpecDistributionParameter,
  setValueSpecFunction,
  setValueSpecFunctionParameter,
  setValueSpecMode,
  setValueSpecScope,
  setValueSpecConstantValue,
  valueSpecChipLabel,
} from '@/features/scenario/valueSpecs';
import type {
  StudioTaskTimelineSignalSpec,
  StudioValueSpec,
  StudioValueSpecMode,
  StudioValueSpecSamplingScope,
} from '@/types/workspace';

const VALUE_SPEC_MODAL_TOP_PX = 92;
const VALUE_SPEC_MODAL_MIN_HEIGHT_PX = 380;
const VALUE_SPEC_MODAL_MIN_BOTTOM_PX = 32;

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
  return (
    value !== null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    (value as { schema_version?: unknown }).schema_version === 'feedbax.studio.value.v1' &&
    typeof (value as { mode?: unknown }).mode === 'string' &&
    typeof (value as { metadata?: unknown }).metadata === 'object'
  );
}

export function literalFromValueSpec(value: unknown): unknown {
  if (!isStudioValueSpec(value)) return value;
  return value.mode === 'constant' ? value.value : value;
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
      schema_version: 'feedbax.studio.value.v1',
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
  const next = { ...valueSpec, metadata };
  if (!forceValueSpec && next.mode === 'constant') return next.value;
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

function useViewportHeight() {
  const [viewportHeight, setViewportHeight] = useState(() =>
    typeof window === 'undefined' ? 720 : window.innerHeight
  );
  useEffect(() => {
    const update = () => setViewportHeight(window.innerHeight);
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);
  return viewportHeight;
}

function ValueSpecModal({
  descriptor,
  initialValue,
  forceValueSpec,
  onCancel,
  onCommit,
}: {
  descriptor: ValueSpecFieldDescriptor;
  initialValue: unknown;
  forceValueSpec: boolean;
  onCancel: () => void;
  onCommit: (value: unknown) => void;
}) {
  const bottomHeight = useLayoutStore((state) => state.bottomHeight);
  const viewportHeight = useViewportHeight();
  const initialMode = isStudioValueSpec(initialValue)
    ? initialValue.mode
    : descriptor.allowedModes[0] ?? 'constant';
  const [activeMode, setActiveMode] = useState<StudioValueSpecMode>(initialMode);
  const [draftsByMode, setDraftsByMode] = useState<Record<string, StudioValueSpec>>(() => {
    const seed = isStudioValueSpec(initialValue)
      ? initialValue
      : createValueSpec(descriptor, initialValue, initialMode);
    return { [seed.mode]: seed };
  });
  const draft =
    draftsByMode[activeMode] ??
    createValueSpec(descriptor, isStudioValueSpec(initialValue) ? initialValue.value : initialValue, activeMode);
  const scopes = descriptor.allowedScopes ?? ['run', 'sweep'];
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
  const maxBottomForUsableModal = Math.max(
    VALUE_SPEC_MODAL_MIN_BOTTOM_PX,
    viewportHeight - VALUE_SPEC_MODAL_TOP_PX - VALUE_SPEC_MODAL_MIN_HEIGHT_PX
  );
  const modalBottom = Math.min(
    Math.max(VALUE_SPEC_MODAL_MIN_BOTTOM_PX, bottomHeight + DIVIDER_HEIGHT),
    maxBottomForUsableModal
  );
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
      ? initialValue
      : createValueSpec(descriptor, initialValue, initialMode);
    setActiveMode(seed.mode);
    setDraftsByMode({ [seed.mode]: seed });
  };
  if (typeof document === 'undefined') return null;
  return createPortal(
    <div
      className="fixed inset-x-0 z-50"
      style={{ top: `${VALUE_SPEC_MODAL_TOP_PX}px`, bottom: `${modalBottom}px` }}
    >
      <button
        type="button"
        className="absolute inset-0 cursor-default bg-slate-900/30 backdrop-blur-[1px]"
        onClick={onCancel}
        aria-label="Cancel value editor"
      />
      <div className="absolute left-1/2 top-1/2 max-h-[calc(100%-2rem)] w-[min(40rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded border border-slate-200 bg-white p-4 shadow-xl">
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
                  <textarea
                    value={formatValue(schedule.points ?? [[0, draft.value ?? 0]])}
                    onChange={(event) => {
                      let points: unknown = schedule.points ?? [[0, draft.value ?? 0]];
                      try {
                        points = JSON.parse(event.target.value);
                      } catch {
                        points = schedule.points ?? [[0, draft.value ?? 0]];
                      }
                      updateDraft({
                        ...draft,
                        schedule: { ...schedule, points },
                      });
                    }}
                    className="min-h-16 rounded border border-slate-200 bg-white px-2 py-1.5 font-mono text-[11px] text-slate-700"
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
            {draft.mode !== 'constant' && scopes.length > 0 && (
              <label className="grid gap-1 text-xs text-slate-500">
                <span>Scope</span>
                <select
                  value={draft.sampling_scope ?? descriptor.defaultScope ?? scopes[0]}
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
          </div>
          <div className="rounded border border-slate-200 bg-slate-50/70 p-3">
            <ValueSpecPreview valueSpec={draft} />
            <div className="mt-2 truncate text-[11px] font-medium text-slate-500">
              {valueSpecChipLabel(draft)}
            </div>
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
              onClick={() => onCommit(commitValueSpec(draft, forceValueSpec, descriptor))}
              className="rounded border border-emerald-600 bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-700"
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
  const [localText, setLocalText] = useState(() => displayTextForValue(value));
  useEffect(() => {
    setLocalText(displayTextForValue(value));
  }, [value]);
  const commitLocalText = () => {
    const nextValue = parseInlineText(localText, value);
    if (nextValue !== value) onChange(nextValue);
    setLocalText(displayTextForValue(nextValue));
  };
  return (
    <>
      <div
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
          onClick={() => setOpen(true)}
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
        ['snapshot', 'run', 'trial', 'epoch', 'timestep', 'sweep'].includes(String(scope))
      )
    : ['run', 'sweep', 'trial', 'epoch', 'timestep'];
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
    allowedScopes: scopes.filter((scope) => scope !== 'replicate'),
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
