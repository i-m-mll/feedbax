import { useEffect, useMemo, useState } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import { useComponents } from '@/hooks/useComponents';
import type { ParamSchema, ParamValue } from '@/types/graph';
import clsx from 'clsx';

export function PropertiesPanel() {
  const nodes = useGraphStore((state) => state.nodes);
  const graph = useGraphStore((state) => state.graph);
  const updateNodeParams = useGraphStore((state) => state.updateNodeParams);
  const { components } = useComponents();

  const selectedNode = useMemo(() => nodes.find((node) => node.selected), [nodes]);

  if (!selectedNode) {
    return (
      <div className="p-6 text-sm text-slate-500">
        Select a node on the canvas to view properties.
      </div>
    );
  }

  const nodeSpec = graph.nodes[selectedNode.id];
  const component = nodeSpec ? components.find((item) => item.name === nodeSpec.type) : undefined;

  if (!nodeSpec) {
    return (
      <div className="p-6 text-sm text-slate-500">
        Node data is missing.
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Node</div>
        <div className="text-lg font-semibold text-slate-800">{selectedNode.id}</div>
        <div className="text-sm text-slate-500">{nodeSpec.type}</div>
      </div>

      <div className="space-y-3">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Parameters</div>
        {(component?.param_schema ?? []).map((param) => (
          <ParamInput
            key={param.name}
            schema={param}
            value={nodeSpec.params[param.name] ?? param.default ?? null}
            onChange={(value) => updateNodeParams(selectedNode.id, param.name, value)}
          />
        ))}
        {!component && (
          <div className="text-sm text-slate-400">
            No schema for this component yet.
          </div>
        )}
      </div>

      <div className="border-t border-slate-100 pt-4">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Ports</div>
        <div className="grid grid-cols-2 gap-4 text-xs text-slate-600 break-words">
          <div>
            <div className="font-semibold text-slate-500 mb-1">Inputs</div>
            <ul className="space-y-1">
              {nodeSpec.input_ports.map((port) => (
                <li key={port}>{port}</li>
              ))}
            </ul>
          </div>
          <div>
            <div className="font-semibold text-slate-500 mb-1">Outputs</div>
            <ul className="space-y-1">
              {nodeSpec.output_ports.map((port) => (
                <li key={port}>{port}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function ParamInput({
  schema,
  value,
  onChange,
}: {
  schema: ParamSchema;
  value: ParamValue;
  onChange: (value: ParamValue) => void;
}) {
  const [jsonValue, setJsonValue] = useState<string>(
    schema.type === 'array' || schema.type === 'object'
      ? JSON.stringify(value ?? schema.default ?? null, null, 2)
      : ''
  );

  useEffect(() => {
    if (schema.type === 'array' || schema.type === 'object') {
      setJsonValue(JSON.stringify(value ?? schema.default ?? null, null, 2));
    }
  }, [schema.type, schema.default, value]);

  const parseBounds2d = (raw: ParamValue, fallback: ParamValue | undefined) => {
    const fallbackValue: number[][] = Array.isArray(fallback)
      ? (fallback as number[][])
      : [
          [0, 0],
          [1, 1],
        ];
    const source: number[][] = Array.isArray(raw) ? (raw as number[][]) : fallbackValue;
    const minRaw = Array.isArray(source[0]) ? (source[0] as number[]) : fallbackValue[0];
    const maxRaw = Array.isArray(source[1]) ? (source[1] as number[]) : fallbackValue[1];
    const safe = (item: unknown, defaultValue: number) =>
      typeof item === 'number' && Number.isFinite(item) ? item : defaultValue;
    return {
      minX: safe(minRaw?.[0], fallbackValue[0][0]),
      minY: safe(minRaw?.[1], fallbackValue[0][1]),
      maxX: safe(maxRaw?.[0], fallbackValue[1][0]),
      maxY: safe(maxRaw?.[1], fallbackValue[1][1]),
    };
  };

  if (schema.type === 'int' || schema.type === 'float') {
    const numericValue =
      typeof value === 'number'
        ? value
        : typeof schema.default === 'number'
          ? schema.default
          : 0;
    return (
      <label className="flex flex-col gap-1 text-xs text-slate-500">
        {schema.name}
        <input
          type="number"
          value={numericValue}
          min={schema.min}
          max={schema.max}
          step={schema.step ?? (schema.type === 'int' ? 1 : 0.01)}
          onChange={(event) => onChange(Number(event.target.value))}
          className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800"
        />
      </label>
    );
  }

  if (schema.type === 'bool') {
    return (
      <label className="flex items-center gap-2 text-sm text-slate-600">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(event) => onChange(event.target.checked)}
          className="h-4 w-4 rounded border-slate-300 text-brand-500 focus:ring-brand-500"
        />
        {schema.name}
      </label>
    );
  }

  if (schema.type === 'enum') {
    return (
      <label className="flex flex-col gap-1 text-xs text-slate-500">
        {schema.name}
        <select
          value={String(value ?? schema.default ?? '')}
          onChange={(event) => onChange(event.target.value)}
          className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800"
        >
          {(schema.options ?? []).map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
    );
  }

  if (schema.type === 'bounds2d') {
    const bounds = parseBounds2d(value, schema.default);
    const update = (next: Partial<typeof bounds>) => {
      const merged = { ...bounds, ...next };
      onChange([
        [merged.minX, merged.minY],
        [merged.maxX, merged.maxY],
      ]);
    };
    return (
      <div className="flex flex-col gap-2 text-xs text-slate-500">
        <div>{schema.name}</div>
        <div className="grid grid-cols-2 gap-2">
          <label className="flex flex-col gap-1">
            Min X
            <input
              type="number"
              value={bounds.minX}
              step={schema.step ?? 0.1}
              onChange={(event) => update({ minX: Number(event.target.value) })}
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800"
            />
          </label>
          <label className="flex flex-col gap-1">
            Min Y
            <input
              type="number"
              value={bounds.minY}
              step={schema.step ?? 0.1}
              onChange={(event) => update({ minY: Number(event.target.value) })}
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800"
            />
          </label>
          <label className="flex flex-col gap-1">
            Max X
            <input
              type="number"
              value={bounds.maxX}
              step={schema.step ?? 0.1}
              onChange={(event) => update({ maxX: Number(event.target.value) })}
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800"
            />
          </label>
          <label className="flex flex-col gap-1">
            Max Y
            <input
              type="number"
              value={bounds.maxY}
              step={schema.step ?? 0.1}
              onChange={(event) => update({ maxY: Number(event.target.value) })}
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800"
            />
          </label>
        </div>
      </div>
    );
  }

  if (schema.type === 'array' || schema.type === 'object') {
    return (
      <label className="flex flex-col gap-1 text-xs text-slate-500">
        {schema.name}
        <textarea
          rows={3}
          value={jsonValue}
          onChange={(event) => setJsonValue(event.target.value)}
          onBlur={() => {
            try {
              const parsed = JSON.parse(jsonValue);
              onChange(parsed);
            } catch {
              // leave value unchanged on parse error
            }
          }}
          className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800 font-mono"
        />
      </label>
    );
  }

  return (
    <label className="flex flex-col gap-1 text-xs text-slate-500">
      {schema.name}
      <input
        type="text"
        value={String(value ?? '')}
        onChange={(event) => onChange(event.target.value)}
        className={clsx(
          'rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800'
        )}
      />
    </label>
  );
}
