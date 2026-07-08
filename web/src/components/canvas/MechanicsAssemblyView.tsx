import { useMemo, useState } from 'react';
import clsx from 'clsx';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { graphNodeEntityId } from '@/features/scenario/entities';
import {
  editLinkLength,
  isAssemblyEditFailure,
  projectMechanicsAssembly,
  rebindJointFrame,
  type AssemblyRow,
  type AssemblyViewMode,
} from '@/features/domains/mechanicsAssembly';
import { useGraphStore } from '@/stores/graphStore';
import { useWorkspaceStore } from '@/stores/workspaceStore';
import type { AcausalGraphSpec, DomainDiagnostic, GraphUIState } from '@/types/graph';

const MODE_OPTIONS: AssemblyViewMode[] = ['graph', 'assembly', 'split'];

interface MechanicsAssemblyViewProps {
  graph: AcausalGraphSpec;
  uiState: GraphUIState;
  diagnostics: DomainDiagnostic[];
  mode: AssemblyViewMode;
  compact?: boolean;
}

function lengthFor(graph: AcausalGraphSpec, nodeId: string, fallback: number) {
  const value = graph.nodes[nodeId]?.params?.length;
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function RestPreview({
  graph,
  selectedRow,
}: {
  graph: AcausalGraphSpec;
  selectedRow: string | null | undefined;
}) {
  const upper = lengthFor(graph, 'upper', 0.33);
  const forearm = lengthFor(graph, 'forearm', 0.27);
  const scale = 270 / Math.max(0.1, upper + forearm);
  const shoulder = { x: 42, y: 92 };
  const elbow = { x: shoulder.x + upper * scale, y: shoulder.y };
  const wrist = { x: elbow.x + forearm * scale, y: shoulder.y };
  const muscles = Object.keys(graph.nodes ?? {}).filter((nodeId) => graph.nodes[nodeId].type === 'MusclePath');

  return (
    <svg
      className="h-40 w-full rounded-md border border-slate-200 bg-white"
      viewBox="0 0 340 160"
      role="img"
      aria-label="Mechanics assembly rest preview"
    >
      <line x1={18} y1={92} x2={shoulder.x} y2={92} stroke="#64748b" strokeWidth={3} />
      <line
        x1={shoulder.x}
        y1={shoulder.y}
        x2={elbow.x}
        y2={elbow.y}
        stroke={selectedRow === 'upper' ? '#2563eb' : '#334155'}
        strokeWidth={8}
        strokeLinecap="round"
      />
      <line
        x1={elbow.x}
        y1={elbow.y}
        x2={wrist.x}
        y2={wrist.y}
        stroke={selectedRow === 'forearm' ? '#2563eb' : '#334155'}
        strokeWidth={8}
        strokeLinecap="round"
      />
      {muscles.map((nodeId, index) => {
        const yOffset = (index - (muscles.length - 1) / 2) * 8;
        const active = selectedRow === nodeId;
        return (
          <polyline
            key={nodeId}
            points={`${shoulder.x},${shoulder.y + yOffset} ${elbow.x},${elbow.y + yOffset * 0.4} ${wrist.x},${wrist.y + yOffset}`}
            fill="none"
            stroke={active ? '#be123c' : '#fb7185'}
            strokeWidth={active ? 3 : 1.8}
            opacity={active ? 0.95 : 0.55}
          />
        );
      })}
      {[
        ['shoulder', shoulder],
        ['elbow', elbow],
        ['effector', wrist],
      ].map(([id, point]) => (
        <circle
          key={id as string}
          cx={(point as typeof shoulder).x}
          cy={(point as typeof shoulder).y}
          r={selectedRow === id ? 7 : 5}
          fill={selectedRow === id ? '#2563eb' : '#0f766e'}
          stroke="#fff"
          strokeWidth={2}
        />
      ))}
    </svg>
  );
}

function RowIcon({ expanded }: { expanded: boolean }) {
  return expanded ? (
    <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
  ) : (
    <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
  );
}

function rowTone(row: AssemblyRow) {
  if (row.diagnostic_codes.length > 0) return 'border-rose-200 bg-rose-50/70';
  if (row.kind === 'joint') return 'border-sky-100 bg-sky-50/60';
  if (row.kind === 'muscle') return 'border-rose-100 bg-white';
  return 'border-slate-200 bg-white';
}

export function MechanicsAssemblyView({
  graph,
  uiState,
  diagnostics,
  mode,
  compact = false,
}: MechanicsAssemblyViewProps) {
  const replaceAcausalGraph = useGraphStore((state) => state.replaceAcausalGraph);
  const setSelectedNode = useGraphStore((state) => state.setSelectedNode);
  const setAssemblyViewState = useGraphStore((state) => state.setAssemblyViewState);
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
  const [localDiagnostic, setLocalDiagnostic] = useState<DomainDiagnostic | null>(null);
  const projection = useMemo(
    () => projectMechanicsAssembly(graph, [...diagnostics, ...(localDiagnostic ? [localDiagnostic] : [])]),
    [diagnostics, graph, localDiagnostic]
  );
  const assemblyState = uiState.assembly_view;
  const expandedRows = assemblyState?.expanded_rows ?? projection.rows.map((row) => row.id);
  const selectedRow = assemblyState?.selected_row ?? null;

  const selectRow = (row: AssemblyRow) => {
    setSelectedNode(row.node_id);
    selectTopPaneEntity(graphNodeEntityId(row.node_id));
    setAssemblyViewState({ selected_row: row.id });
  };

  const toggleExpanded = (rowId: string) => {
    const expanded = new Set(expandedRows);
    if (expanded.has(rowId)) expanded.delete(rowId);
    else expanded.add(rowId);
    setAssemblyViewState({ expanded_rows: Array.from(expanded) });
  };

  return (
    <div
      data-testid="mechanics-assembly-view"
      className={clsx(
        'flex h-full min-h-0 flex-col border-slate-200 bg-slate-50/95 text-slate-700 shadow-soft',
        compact ? 'rounded-md border' : 'border-l'
      )}
    >
      <div className="flex min-h-12 items-center justify-between border-b border-slate-200 bg-white px-3">
        <div className="text-sm font-semibold text-slate-800">Mechanics assembly</div>
        <div className="grid grid-cols-3 overflow-hidden rounded-md border border-slate-200 text-[11px]">
          {MODE_OPTIONS.map((option) => (
            <button
              key={option}
              type="button"
              className={clsx(
                'h-7 min-w-16 px-2 capitalize',
                mode === option ? 'bg-slate-800 text-white' : 'bg-white text-slate-600 hover:bg-slate-100'
              )}
              onClick={() => setAssemblyViewState({ active_view: option })}
            >
              {option}
            </button>
          ))}
        </div>
      </div>
      <div className="grid min-h-0 flex-1 grid-rows-[auto_1fr] gap-3 overflow-hidden p-3">
        <RestPreview graph={graph} selectedRow={selectedRow} />
        <div className="min-h-0 overflow-auto pr-1">
          {projection.diagnostics.length > 0 && (
            <div className="mb-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
              <span className="font-medium">{projection.diagnostics[0].code}</span>
              <span className="ml-1">{projection.diagnostics[0].message}</span>
            </div>
          )}
          <div className="space-y-1.5">
            {projection.rows.map((row) => {
              const expanded = expandedRows.includes(row.id);
              return (
                <div
                  key={row.id}
                  className={clsx(
                    'min-h-10 rounded-md border text-xs transition-colors',
                    rowTone(row),
                    selectedRow === row.id && 'ring-2 ring-brand-300'
                  )}
                  style={{ marginLeft: row.depth * 14 }}
                >
                  <div className="grid min-h-10 grid-cols-[28px_1fr_auto] items-center gap-2 px-2">
                    <button
                      type="button"
                      className="flex h-7 w-7 items-center justify-center rounded text-slate-500 hover:bg-slate-100"
                      onClick={() => toggleExpanded(row.id)}
                      aria-label={expanded ? `Collapse ${row.label}` : `Expand ${row.label}`}
                    >
                      <RowIcon expanded={expanded} />
                    </button>
                    <button
                      type="button"
                      className="min-w-0 text-left"
                      onClick={() => selectRow(row)}
                    >
                      <span className="block truncate font-medium text-slate-800">{row.label}</span>
                      <span className="block truncate text-[10px] uppercase tracking-normal text-slate-400">
                        {row.kind}
                      </span>
                    </button>
                    <div className="flex min-w-16 justify-end gap-1">
                      {row.diagnostic_codes.map((code) => (
                        <span
                          key={code}
                          className="max-w-28 truncate rounded-full bg-rose-100 px-2 py-0.5 text-[10px] font-medium text-rose-700"
                          title={code}
                        >
                          {code}
                        </span>
                      ))}
                      {typeof row.attachment_count === 'number' && row.attachment_count > 0 && (
                        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-600">
                          {row.attachment_count}
                        </span>
                      )}
                    </div>
                  </div>
                  {expanded && (
                    <div className="space-y-2 border-t border-inherit px-3 py-2">
                      {row.kind === 'link' && (
                        <label className="grid grid-cols-[72px_1fr] items-center gap-2 text-[11px]">
                          <span className="text-slate-500">Length</span>
                          <input
                            className="h-8 rounded-md border border-slate-200 bg-white px-2 text-xs tabular-nums outline-none focus:border-brand-300"
                            type="number"
                            min={0.001}
                            step={0.01}
                            value={row.length ?? ''}
                            onChange={(event) => {
                              const result = editLinkLength(graph, row.node_id, Number(event.target.value));
                              if (result.ok) {
                                setLocalDiagnostic(null);
                                replaceAcausalGraph(result.graph);
                              } else if (isAssemblyEditFailure(result)) {
                                setLocalDiagnostic(result.diagnostic);
                              }
                            }}
                          />
                        </label>
                      )}
                      {row.sockets?.map((socket) => (
                        <label
                          key={`${row.id}-${socket.kind}`}
                          className="grid grid-cols-[72px_1fr] items-center gap-2 text-[11px]"
                        >
                          <span className="capitalize text-slate-500">{socket.kind}</span>
                          <select
                            className={clsx(
                              'h-8 rounded-md border bg-white px-2 text-xs outline-none focus:border-brand-300',
                              socket.diagnostic_codes.length > 0 ? 'border-rose-300' : 'border-slate-200'
                            )}
                            value={socket.frame_id ?? ''}
                            onChange={(event) => {
                              const result = rebindJointFrame(
                                graph,
                                row.node_id,
                                socket.kind,
                                event.target.value
                              );
                              if (result.ok) {
                                setLocalDiagnostic(null);
                                replaceAcausalGraph(result.graph);
                              } else if (isAssemblyEditFailure(result)) {
                                setLocalDiagnostic(result.diagnostic);
                              }
                            }}
                          >
                            {socket.options.map((option) => (
                              <option key={option.id} value={option.id}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
