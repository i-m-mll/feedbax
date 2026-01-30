import { useEffect, useMemo, useState } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import { useComponents } from '@/hooks/useComponents';
import type { ParamSchema, ParamValue, TapSpec } from '@/types/graph';
import clsx from 'clsx';

export function PropertiesPanel() {
  const nodes = useGraphStore((state) => state.nodes);
  const graph = useGraphStore((state) => state.graph);
  const updateNodeParams = useGraphStore((state) => state.updateNodeParams);
  const renameNode = useGraphStore((state) => state.renameNode);
  const addTap = useGraphStore((state) => state.addTap);
  const updateTap = useGraphStore((state) => state.updateTap);
  const removeTap = useGraphStore((state) => state.removeTap);
  const setSelectedTap = useGraphStore((state) => state.setSelectedTap);
  const selectedTapId = useGraphStore((state) => state.selectedTapId);
  const { components } = useComponents();

  const selectedNode = useMemo(
    () => nodes.find((node) => node.selected && node.type !== 'tap'),
    [nodes]
  );
  const taps = graph.taps ?? [];
  const selectedTap = selectedTapId
    ? taps.find((tap) => tap.id === selectedTapId)
    : undefined;

  const [nameValue, setNameValue] = useState('');

  useEffect(() => {
    if (selectedNode) {
      setNameValue(selectedNode.id);
    }
  }, [selectedNode?.id]);

  if (selectedTap) {
    return (
      <TapEditor
        tap={selectedTap}
        nodeIds={Object.keys(graph.nodes)}
        onUpdate={(updates) => updateTap(selectedTap.id, updates)}
        onRemove={() => removeTap(selectedTap.id)}
      />
    );
  }

  if (!selectedNode) {
    return (
      <div className="p-6 text-sm text-slate-500">
        Select a node or tap on the canvas to view properties.
      </div>
    );
  }

  const nodeSpec = graph.nodes[selectedNode.id];
  const component = nodeSpec
    ? components.find((item) => item.name === nodeSpec.type)
    : undefined;
  const nodeTaps = taps.filter((tap) => tap.position.afterNode === selectedNode.id);

  const commitRename = () => {
    if (nameValue.trim() && nameValue.trim() !== selectedNode.id) {
      renameNode(selectedNode.id, nameValue.trim());
    }
  };

  if (!nodeSpec) {
    return <div className="p-6 text-sm text-slate-500">Node data is missing.</div>;
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Node</div>
        <label className="flex flex-col gap-2 text-xs text-slate-500 mt-3">
          Name
          <input
            value={nameValue}
            onChange={(event) => setNameValue(event.target.value)}
            onBlur={commitRename}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                commitRename();
              }
              if (event.key === 'Escape') {
                setNameValue(selectedNode.id);
              }
            }}
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800"
          />
        </label>
        <div className="text-sm text-slate-500 mt-2">{nodeSpec.type}</div>
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
          <div className="text-sm text-slate-400">No schema for this component yet.</div>
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

      <div className="border-t border-slate-100 pt-4 space-y-3">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Taps</div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600 hover:text-slate-800"
            onClick={() => addTap(selectedNode.id, 'probe')}
          >
            Add Probe
          </button>
          <button
            className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600 hover:text-slate-800"
            onClick={() => addTap(selectedNode.id, 'intervention')}
          >
            Add Intervention
          </button>
        </div>
        {nodeTaps.length === 0 ? (
          <div className="text-sm text-slate-400">No taps on this wire yet.</div>
        ) : (
          <div className="space-y-2">
            {nodeTaps.map((tap) => (
              <button
                key={tap.id}
                className="flex w-full items-center justify-between rounded-lg border border-slate-200 px-3 py-2 text-left text-xs text-slate-600 hover:border-brand-200 hover:text-slate-800"
                onClick={() => setSelectedTap(tap.id)}
              >
                <span className="font-medium capitalize">{tap.type}</span>
                <span className="text-slate-400">{Object.keys(tap.paths ?? {}).length} outputs</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function TapEditor({
  tap,
  nodeIds,
  onUpdate,
  onRemove,
}: {
  tap: TapSpec;
  nodeIds: string[];
  onUpdate: (updates: Partial<TapSpec>) => void;
  onRemove: () => void;
}) {
  const [newOutputName, setNewOutputName] = useState('');
  const [newOutputPath, setNewOutputPath] = useState('');

  const transform = tap.transform ?? { type: 'custom', params: {} };
  const [transformJson, setTransformJson] = useState(
    JSON.stringify(transform.params ?? {}, null, 2)
  );
  const [transformType, setTransformType] = useState(transform.type ?? 'custom');

  useEffect(() => {
    setTransformType(transform.type ?? 'custom');
    setTransformJson(JSON.stringify(transform.params ?? {}, null, 2));
  }, [tap.id, tap.transform]);

  const updatePaths = (next: Record<string, string>) => {
    onUpdate({ paths: next });
  };

  const handleRename = (oldName: string, nextName: string) => {
    const trimmed = nextName.trim();
    if (!trimmed || trimmed === oldName) return;
    if (trimmed in tap.paths) return;
    const next = { ...tap.paths };
    const value = next[oldName];
    delete next[oldName];
    next[trimmed] = value;
    updatePaths(next);
  };

  const handlePathChange = (name: string, nextPath: string) => {
    updatePaths({ ...tap.paths, [name]: nextPath });
  };

  const handleRemovePath = (name: string) => {
    const next = { ...tap.paths };
    delete next[name];
    updatePaths(next);
  };

  const handleAddPath = () => {
    const name = newOutputName.trim();
    const path = newOutputPath.trim();
    if (!name || name in tap.paths) return;
    updatePaths({ ...tap.paths, [name]: path });
    setNewOutputName('');
    setNewOutputPath('');
  };

  const handleTypeChange = (nextType: TapSpec['type']) => {
    if (nextType === 'probe') {
      onUpdate({ type: nextType, transform: undefined });
    } else {
      onUpdate({
        type: nextType,
        transform: tap.transform ?? { type: 'custom', params: {} },
      });
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Tap</div>
        <div className="mt-2 flex items-center justify-between">
          <div className="text-sm font-medium text-slate-700 capitalize">{tap.type} tap</div>
          <button className="text-xs text-slate-400 hover:text-rose-500" onClick={onRemove}>
            Remove
          </button>
        </div>
      </div>

      <div className="grid gap-3 text-xs text-slate-500">
        <label className="flex flex-col gap-1">
          Type
          <select
            value={tap.type}
            onChange={(event) => handleTypeChange(event.target.value as TapSpec['type'])}
            className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700"
          >
            <option value="probe">Probe</option>
            <option value="intervention">Intervention</option>
          </select>
        </label>
        <label className="flex flex-col gap-1">
          After node
          <select
            value={tap.position.afterNode}
            onChange={(event) =>
              onUpdate({
                position: {
                  ...tap.position,
                  afterNode: event.target.value,
                  targetNode: undefined,
                },
              })
            }
            className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700"
          >
            {nodeIds.map((nodeId) => (
              <option key={nodeId} value={nodeId}>
                {nodeId}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          Target node
          <select
            value={tap.position.targetNode ?? ''}
            onChange={(event) =>
              onUpdate({
                position: {
                  ...tap.position,
                  targetNode: event.target.value || undefined,
                },
              })
            }
            className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700"
          >
            <option value="">Auto</option>
            {nodeIds.map((nodeId) => (
              <option key={nodeId} value={nodeId}>
                {nodeId}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="border-t border-slate-100 pt-4 space-y-3">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Outputs</div>
        {Object.entries(tap.paths ?? {}).length === 0 ? (
          <div className="text-sm text-slate-400">No outputs defined yet.</div>
        ) : (
          <div className="space-y-2">
            {Object.entries(tap.paths).map(([name, path]) => (
              <TapPathRow
                key={name}
                name={name}
                path={path}
                onRename={(nextName) => handleRename(name, nextName)}
                onPathChange={(nextPath) => handlePathChange(name, nextPath)}
                onRemove={() => handleRemovePath(name)}
              />
            ))}
          </div>
        )}
        <div className="grid grid-cols-[1fr_1.5fr_auto] gap-2">
          <input
            value={newOutputName}
            onChange={(event) => setNewOutputName(event.target.value)}
            className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700"
            placeholder="output name"
          />
          <input
            value={newOutputPath}
            onChange={(event) => setNewOutputPath(event.target.value)}
            className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700"
            placeholder="state.path"
          />
          <button
            className="rounded-lg border border-slate-200 px-2 py-1 text-xs text-slate-600 hover:text-slate-800"
            onClick={handleAddPath}
          >
            Add
          </button>
        </div>
      </div>

      {tap.type === 'intervention' && (
        <div className="border-t border-slate-100 pt-4 space-y-3">
          <div className="text-xs uppercase tracking-[0.3em] text-slate-400">
            Transform
          </div>
          <label className="flex flex-col gap-1 text-xs text-slate-500">
            Type
            <input
              value={transformType}
              onChange={(event) => setTransformType(event.target.value)}
              onBlur={() =>
                onUpdate({
                  transform: {
                    type: transformType || 'custom',
                    params: transform.params ?? {},
                  },
                })
              }
              className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-500">
            Params (JSON)
            <textarea
              rows={4}
              value={transformJson}
              onChange={(event) => setTransformJson(event.target.value)}
              onBlur={() => {
                try {
                  const parsed = JSON.parse(transformJson);
                  onUpdate({
                    transform: {
                      type: transformType || 'custom',
                      params: parsed,
                    },
                  });
                } catch {
                  // ignore invalid JSON
                }
              }}
              className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700 font-mono"
            />
          </label>
        </div>
      )}
    </div>
  );
}

function TapPathRow({
  name,
  path,
  onRename,
  onPathChange,
  onRemove,
}: {
  name: string;
  path: string;
  onRename: (nextName: string) => void;
  onPathChange: (nextPath: string) => void;
  onRemove: () => void;
}) {
  const [localName, setLocalName] = useState(name);
  const [localPath, setLocalPath] = useState(path);

  useEffect(() => {
    setLocalName(name);
  }, [name]);

  useEffect(() => {
    setLocalPath(path);
  }, [path]);

  return (
    <div className="grid grid-cols-[1fr_1.5fr_auto] gap-2">
      <input
        value={localName}
        onChange={(event) => setLocalName(event.target.value)}
        onBlur={() => onRename(localName)}
        className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700"
      />
      <input
        value={localPath}
        onChange={(event) => setLocalPath(event.target.value)}
        onBlur={() => onPathChange(localPath)}
        className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700"
      />
      <button
        className="rounded-lg border border-slate-200 px-2 py-1 text-xs text-slate-600 hover:text-rose-500"
        onClick={onRemove}
      >
        Remove
      </button>
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
        className={clsx('rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800')}
      />
    </label>
  );
}
