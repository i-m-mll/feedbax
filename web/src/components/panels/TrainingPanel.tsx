import { useTrainingStore } from '@/stores/trainingStore';
import { useTraining } from '@/hooks/useTraining';
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import type { LossTermSpec, TimeAggregationSpec } from '@/types/training';
import { LossTermDetail } from './LossTermDetail';
import { AddLossTermModal } from '@/components/modals/AddLossTermModal';
import { fetchProbes, validateLossSpec } from '@/api/client';
import clsx from 'clsx';
import { Plus, Trash2, AlertCircle } from 'lucide-react';

export function TrainingPanel() {
  const { trainingSpec, setTrainingSpec, progress, status } = useTrainingStore();
  const setAvailableProbes = useTrainingStore((state) => state.setAvailableProbes);
  const setLossValidationErrors = useTrainingStore((state) => state.setLossValidationErrors);
  const lossValidationErrors = useTrainingStore((state) => state.lossValidationErrors);
  const removeLossTerm = useTrainingStore((state) => state.removeLossTerm);
  const setHighlightedProbeSelector = useTrainingStore((state) => state.setHighlightedProbeSelector);
  const { start, stop } = useTraining();
  const graphId = useGraphStore((state) => state.graphId);
  const inSubgraph = useGraphStore((state) => state.graphStack.length > 0);
  const [expanded, setExpanded] = useState<Record<string, boolean>>(() =>
    buildDefaultExpanded(trainingSpec.loss)
  );
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [addModalParentPath, setAddModalParentPath] = useState<string[]>([]);
  const [showDetailPanel, setShowDetailPanel] = useState(false);
  const nodeRefs = useRef<Record<string, HTMLDivElement | null>>({});

  // Fetch available probes when graph changes
  useEffect(() => {
    if (graphId) {
      fetchProbes(graphId)
        .then(setAvailableProbes)
        .catch((err) => console.warn('Failed to fetch probes:', err));
    }
  }, [graphId, setAvailableProbes]);

  // Validate loss spec when it changes
  useEffect(() => {
    if (graphId) {
      validateLossSpec(graphId, trainingSpec.loss)
        .then((result) => setLossValidationErrors(result.errors))
        .catch((err) => console.warn('Failed to validate loss:', err));
    }
  }, [graphId, trainingSpec.loss, setLossValidationErrors]);

  const percent = useMemo(() => {
    if (!progress) return 0;
    return Math.round((progress.batch / progress.total_batches) * 100);
  }, [progress]);

  const equationTerms = useMemo(
    () => collectEquationTerms(trainingSpec.loss),
    [trainingSpec.loss]
  );

  useEffect(() => {
    setExpanded((prev) => ({
      ...buildDefaultExpanded(trainingSpec.loss),
      ...prev,
    }));
  }, [trainingSpec.loss]);

  const registerNode = useCallback((pathKey: string, node: HTMLDivElement | null) => {
    nodeRefs.current[pathKey] = node;
  }, []);

  const handleToggle = useCallback((path: string[]) => {
    const key = toPathKey(path);
    setExpanded((prev) => ({
      ...prev,
      [key]: !(prev[key] ?? true),
    }));
  }, []);

  const handleSelect = useCallback((path: string[]) => {
    setSelectedPath(toPathKey(path));
  }, []);

  const handleWeightChange = useCallback(
    (path: string[], weight: number) => {
      if (!Number.isFinite(weight)) return;
      const updated = updateLossWeight(trainingSpec.loss, path, weight);
      setTrainingSpec({ loss: updated });
    },
    [setTrainingSpec, trainingSpec.loss]
  );

  const handleJumpToTerm = useCallback((path: string[]) => {
    const key = toPathKey(path);
    setSelectedPath(key);
    setExpanded((prev) => {
      const next = { ...prev, [ROOT_PATH]: true };
      for (let i = 1; i <= path.length; i += 1) {
        next[path.slice(0, i).join('/')] = true;
      }
      return next;
    });
    nodeRefs.current[key]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, []);

  const handleAddTerm = useCallback((parentPath: string[]) => {
    setAddModalParentPath(parentPath);
    setShowAddModal(true);
  }, []);

  const handleRemoveTerm = useCallback(
    (path: string[]) => {
      if (path.length === 0) return; // Cannot remove root
      if (confirm('Remove this loss term?')) {
        removeLossTerm(path);
        if (selectedPath === toPathKey(path)) {
          setSelectedPath(null);
          setShowDetailPanel(false);
        }
      }
    },
    [removeLossTerm, selectedPath]
  );

  const handleOpenDetail = useCallback((path: string[]) => {
    setSelectedPath(toPathKey(path));
    setShowDetailPanel(true);
  }, []);

  const handleCloseDetail = useCallback(() => {
    setShowDetailPanel(false);
  }, []);

  const handleTermHover = useCallback(
    (term: LossTermSpec | null) => {
      if (term?.selector) {
        setHighlightedProbeSelector(term.selector);
      } else {
        setHighlightedProbeSelector(null);
      }
    },
    [setHighlightedProbeSelector]
  );

  const selectedPathArray = useMemo(() => {
    if (!selectedPath || selectedPath === ROOT_PATH) return [];
    return selectedPath.split('/');
  }, [selectedPath]);

  return (
    <div className="p-6 space-y-4 text-sm text-slate-600 overflow-x-hidden">
      <div>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Training</div>
        <div className="text-base font-semibold text-slate-800">Configuration</div>
      </div>
      <div className="space-y-3">
        <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-4 space-y-2">
          <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Optimizer</div>
          <div className="flex items-center gap-2">
            <select
              value={trainingSpec.optimizer.type}
              onChange={(event) =>
                setTrainingSpec({
                  optimizer: { ...trainingSpec.optimizer, type: event.target.value },
                })
              }
              className="rounded-lg border border-slate-200 px-2 py-1 text-sm"
            >
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
              <option value="adamw">AdamW</option>
            </select>
            <input
              type="number"
              step="0.0001"
              value={Number(trainingSpec.optimizer.params.learning_rate ?? 0.001)}
              onChange={(event) =>
                setTrainingSpec({
                  optimizer: {
                    ...trainingSpec.optimizer,
                    params: {
                      ...trainingSpec.optimizer.params,
                      learning_rate: Number(event.target.value),
                    },
                  },
                })
              }
              className="w-24 rounded-lg border border-slate-200 px-2 py-1 text-sm"
            />
            <span className="text-xs text-slate-400">lr</span>
          </div>
        </div>
        <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Loss Function</div>
              {lossValidationErrors.length > 0 && (
                <div className="flex items-center gap-1 text-amber-500" title={`${lossValidationErrors.length} validation error(s)`}>
                  <AlertCircle className="w-3.5 h-3.5" />
                  <span className="text-xs">{lossValidationErrors.length}</span>
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={() => handleAddTerm([])}
              className="h-7 w-7 rounded-full border border-slate-200 text-slate-500 hover:bg-white hover:text-brand-500 hover:border-brand-200 transition-colors"
              title="Add loss term"
            >
              <Plus className="w-4 h-4 mx-auto" />
            </button>
          </div>
          <LossEquation terms={equationTerms} onSelect={handleJumpToTerm} />
          <LossTree
            term={trainingSpec.loss}
            path={[]}
            depth={0}
            expanded={expanded}
            selectedPath={selectedPath}
            onToggle={handleToggle}
            onSelect={handleSelect}
            onWeightChange={handleWeightChange}
            onAddChild={handleAddTerm}
            onRemove={handleRemoveTerm}
            onOpenDetail={handleOpenDetail}
            onHover={handleTermHover}
            registerNode={registerNode}
          />
        </div>

        {/* Detail panel */}
        {showDetailPanel && selectedPath && (
          <div className="rounded-xl border border-brand-100 bg-white p-4">
            <LossTermDetail
              path={selectedPathArray}
              onClose={handleCloseDetail}
            />
          </div>
        )}

        {/* Add term modal */}
        {showAddModal && (
          <AddLossTermModal
            parentPath={addModalParentPath}
            onClose={() => setShowAddModal(false)}
          />
        )}
        <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-4 space-y-2">
          <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Batches</div>
          <div className="flex items-center gap-2">
            <input
              type="number"
              min={1}
              value={trainingSpec.n_batches}
              onChange={(event) => setTrainingSpec({ n_batches: Number(event.target.value) })}
              className="w-24 rounded-lg border border-slate-200 px-2 py-1 text-sm"
            />
            <span className="text-xs text-slate-400">batches</span>
            <input
              type="number"
              min={1}
              value={trainingSpec.batch_size}
              onChange={(event) => setTrainingSpec({ batch_size: Number(event.target.value) })}
              className="w-20 rounded-lg border border-slate-200 px-2 py-1 text-sm"
            />
            <span className="text-xs text-slate-400">batch size</span>
          </div>
        </div>
      </div>

      {progress && (
        <div className="rounded-xl border border-slate-100 bg-white p-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>Progress</span>
            <span>
              {progress.batch}/{progress.total_batches}
            </span>
          </div>
          <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
            <div
              className="h-full bg-brand-500 transition-all"
              style={{ width: `${percent}%` }}
            />
          </div>
          <div className="text-xs text-slate-500">Loss: {progress.loss.toFixed(4)}</div>
        </div>
      )}
      {status === 'completed' && (
        <div className="text-xs text-mint-500">Training completed.</div>
      )}
      {status === 'error' && (
        <div className="text-xs text-amber-600">Training failed. Check console.</div>
      )}

      {inSubgraph && (
        <div className="text-xs text-amber-600">
          Return to the model root to start training.
        </div>
      )}
      {!graphId && !inSubgraph && (
        <div className="text-xs text-amber-600">Save the project before starting training.</div>
      )}
      <button
        className="w-full rounded-full bg-brand-500 text-white py-2 text-sm font-semibold shadow-soft hover:bg-brand-600"
        onClick={status === 'running' ? stop : start}
        disabled={!graphId || inSubgraph}
      >
        {status === 'running' ? 'Stop Training' : 'Start Training'}
      </button>
    </div>
  );
}

function LossEquation({
  terms,
  onSelect,
}: {
  terms: EquationTerm[];
  onSelect: (path: string[]) => void;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
      <div className="flex flex-wrap items-center gap-1">
        <span className="font-semibold text-slate-700">L =</span>
        {terms.length === 0 && <span className="text-slate-400">0</span>}
        {terms.map((term, index) => (
          <Fragment key={term.pathKey}>
            {index > 0 && <span className="text-slate-400">+</span>}
            <button
              type="button"
              onClick={() => onSelect(term.path)}
              className="rounded px-1 text-slate-600 hover:bg-slate-100 hover:text-slate-800"
              title={term.label}
            >
              {formatWeight(term.weight)}·L_{term.symbol}
            </button>
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function LossTree({
  term,
  path,
  depth = 0,
  expanded,
  selectedPath,
  onToggle,
  onSelect,
  onWeightChange,
  onAddChild,
  onRemove,
  onOpenDetail,
  onHover,
  registerNode,
}: {
  term: LossTermSpec;
  path: string[];
  depth?: number;
  expanded: Record<string, boolean>;
  selectedPath: string | null;
  onToggle: (path: string[]) => void;
  onSelect: (path: string[]) => void;
  onWeightChange: (path: string[], weight: number) => void;
  onAddChild: (parentPath: string[]) => void;
  onRemove: (path: string[]) => void;
  onOpenDetail: (path: string[]) => void;
  onHover: (term: LossTermSpec | null) => void;
  registerNode: (pathKey: string, node: HTMLDivElement | null) => void;
}) {
  const entries = term.children ? Object.entries(term.children) : [];
  const hasChildren = entries.length > 0;
  const pathKey = toPathKey(path);
  const isExpanded = expanded[pathKey] ?? hasChildren;
  const isSelected = selectedPath === pathKey;
  const detailLines = buildDetailLines(term);
  const showDetails = detailLines.length > 0 && !hasChildren;
  const isRoot = path.length === 0;
  const isComposite = term.type === 'Composite' || hasChildren;

  return (
    <div className="space-y-2">
      <div
        ref={(node) => registerNode(pathKey, node)}
        className={clsx(
          'group flex items-center justify-between rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm',
          isSelected && 'border-brand-200 ring-1 ring-brand-200'
        )}
        style={{ marginLeft: depth * 12 }}
        onClick={() => onSelect(path)}
        onDoubleClick={() => onOpenDetail(path)}
        onMouseEnter={() => onHover(term)}
        onMouseLeave={() => onHover(null)}
      >
        <div className="flex items-center gap-2">
          {hasChildren ? (
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                onToggle(path);
              }}
              className="text-slate-400 hover:text-slate-600"
              aria-label={isExpanded ? 'Collapse' : 'Expand'}
            >
              {isExpanded ? '▼' : '▶'}
            </button>
          ) : (
            <span className="text-slate-300">•</span>
          )}
          <div className="flex flex-col">
            <span className="font-semibold text-slate-700">{term.label}</span>
            <span className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
              {term.type}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1 text-xs text-slate-500">
          {/* Action buttons - show on hover */}
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity mr-2">
            {isComposite && (
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  onAddChild(path);
                }}
                className="p-1 rounded text-slate-400 hover:text-brand-500 hover:bg-brand-50"
                title="Add child term"
              >
                <Plus className="w-3.5 h-3.5" />
              </button>
            )}
            {!isRoot && (
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  onRemove(path);
                }}
                className="p-1 rounded text-slate-400 hover:text-red-500 hover:bg-red-50"
                title="Remove term"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
          <span className="text-slate-400">×</span>
          <input
            type="number"
            min={0}
            step={0.001}
            value={Number.isFinite(term.weight) ? term.weight : 0}
            onChange={(event) => onWeightChange(path, Number(event.target.value))}
            onClick={(event) => event.stopPropagation()}
            className="w-20 rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-700"
          />
        </div>
      </div>
      {showDetails && (
        <div
          className="space-y-1 text-xs text-slate-500"
          style={{ marginLeft: depth * 12 + 24 }}
        >
          {detailLines.map((line) => (
            <div key={line}>{line}</div>
          ))}
        </div>
      )}
      {hasChildren && isExpanded && (
        <div className="space-y-2">
          {entries.map(([childLabel, child]) => (
            <LossTree
              key={childLabel}
              term={child}
              path={[...path, childLabel]}
              depth={depth + 1}
              expanded={expanded}
              selectedPath={selectedPath}
              onToggle={onToggle}
              onSelect={onSelect}
              onWeightChange={onWeightChange}
              onAddChild={onAddChild}
              onRemove={onRemove}
              onOpenDetail={onOpenDetail}
              onHover={onHover}
              registerNode={registerNode}
            />
          ))}
        </div>
      )}
    </div>
  );
}

const ROOT_PATH = 'root';

interface EquationTerm {
  path: string[];
  pathKey: string;
  label: string;
  symbol: string;
  weight: number;
}

function toPathKey(path: string[]) {
  return path.length === 0 ? ROOT_PATH : path.join('/');
}

function buildDefaultExpanded(term: LossTermSpec, path: string[] = []): Record<string, boolean> {
  const entries = term.children ? Object.entries(term.children) : [];
  const next: Record<string, boolean> = {
    [toPathKey(path)]: true,
  };
  entries.forEach(([childLabel, child]) => {
    Object.assign(next, buildDefaultExpanded(child, [...path, childLabel]));
  });
  return next;
}

function collectEquationTerms(term: LossTermSpec): EquationTerm[] {
  const terms: EquationTerm[] = [];
  const walk = (node: LossTermSpec, path: string[], weightScale: number) => {
    const nextWeight = weightScale * node.weight;
    const children = node.children ? Object.entries(node.children) : [];
    if (children.length === 0) {
      const key = path[path.length - 1] ?? node.label ?? node.type;
      const label = node.label || key;
      const symbol = sanitizeSymbol(key);
      terms.push({
        path,
        pathKey: toPathKey(path),
        label,
        symbol,
        weight: nextWeight,
      });
      return;
    }
    children.forEach(([childLabel, child]) => {
      walk(child, [...path, childLabel], nextWeight);
    });
  };
  walk(term, [], 1);
  return terms;
}

function updateLossWeight(term: LossTermSpec, path: string[], weight: number): LossTermSpec {
  if (path.length === 0) {
    return { ...term, weight };
  }
  if (!term.children) return term;
  const [head, ...rest] = path;
  const child = term.children[head];
  if (!child) return term;
  return {
    ...term,
    children: {
      ...term.children,
      [head]: updateLossWeight(child, rest, weight),
    },
  };
}

function buildDetailLines(term: LossTermSpec): string[] {
  const selector = formatSelector(term.selector);
  const norm = formatNorm(term.norm);
  const time = formatTimeAggregation(term.time_agg);
  const details: string[] = [];
  const primary = [selector, norm ? `Norm: ${norm}` : null].filter(Boolean).join(' • ');
  if (primary) details.push(primary);
  if (time) details.push(`Time: ${time}`);
  return details;
}

function formatSelector(selector?: string): string | null {
  if (!selector) return null;
  if (selector.startsWith('probe:')) {
    return `Probe: ${selector.slice('probe:'.length)}`;
  }
  return `Path: ${selector}`;
}

function formatNorm(norm?: LossTermSpec['norm']): string | null {
  if (!norm) return null;
  const labels: Record<NonNullable<LossTermSpec['norm']>, string> = {
    squared_l2: 'Squared L2',
    l2: 'L2',
    l1: 'L1',
    huber: 'Huber',
  };
  return labels[norm];
}

function formatTimeAggregation(timeAgg?: TimeAggregationSpec): string | null {
  if (!timeAgg) return null;
  let base: string;
  switch (timeAgg.mode) {
    case 'all':
      base = 'All steps';
      break;
    case 'final':
      base = 'Final step';
      break;
    case 'range':
      base = `Range ${timeAgg.start ?? '?'}–${timeAgg.end ?? '?'}`;
      break;
    case 'segment':
      base = `Segment ${timeAgg.segment_name ?? 'unknown'}`;
      break;
    case 'custom':
      base = `Custom steps ${timeAgg.time_idxs?.join(', ') ?? '[]'}`;
      break;
    default:
      base = 'All steps';
      break;
  }
  if (timeAgg.discount && timeAgg.discount !== 'none') {
    if (timeAgg.discount === 'power') {
      return `${base}, power discount (exp=${timeAgg.discount_exp ?? 1})`;
    }
    return `${base}, linear discount`;
  }
  return base;
}

function formatWeight(value: number): string {
  if (!Number.isFinite(value)) return '0.0';
  const fixed = value.toFixed(3);
  const trimmed = fixed.replace(/\.?0+$/, '');
  return trimmed.includes('.') ? trimmed : `${trimmed}.0`;
}

function sanitizeSymbol(label: string): string {
  const cleaned = label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return cleaned.length > 0 ? cleaned : 'term';
}
