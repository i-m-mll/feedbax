/**
 * DataSourceNode — the implicit data source on the left edge of the analysis DAG.
 *
 * Renders the hierarchical state field tree (states, model, task) with
 * expand/collapse controls. Each tree node is a connectable React Flow Handle,
 * allowing users to wire specific sub-fields (e.g. "states.net.hidden") to
 * analysis nodes. The node auto-resizes when branches are expanded/collapsed.
 *
 * Top-level items are always visible with chevrons. Connecting to a branch
 * node sends the full subtree; connecting to a leaf sends that specific field.
 */

import { useEffect, useMemo } from 'react';
import { useUpdateNodeInternals, type NodeProps } from '@xyflow/react';
import type { DataSourceNodeData } from '@/stores/analysisStore';
import { useAnalysisStore } from '@/stores/analysisStore';
import { STATE_FIELD_TREE, type StateFieldNode } from '@/types/analysis';
import { buildScenarioEntityRegistry } from '@/features/scenario/entities';
import { ensureObjectiveSpec } from '@/features/scenario/objectives';
import {
  selectorGroupLabel,
  selectorOptionsForRegistry,
  type SelectorOptionGroup,
} from '@/features/scenario/selectors';
import { useStudioSchemaRegistry } from '@/hooks/useStudioSchemas';
import { useGraphStore } from '@/stores/graphStore';
import { getStageByKind, getTrainingScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import { StateFieldTree, countVisibleRows, FIELD_ROW_HEIGHT } from './StateFieldTree';
import { Database } from 'lucide-react';
import clsx from 'clsx';

const WIDTH = 200;
const HEADER_HEIGHT = 36;
const BODY_PADDING = 8;

const DATA_SOURCE_SELECTOR_GROUPS: SelectorOptionGroup[] = [
  'ports',
  'observables',
  'state',
  'task',
  'mechanics',
];

function selectorTreeFromScenarioOptions(
  options: ReturnType<typeof selectorOptionsForRegistry>
): StateFieldNode[] {
  return DATA_SOURCE_SELECTOR_GROUPS.flatMap((group) => {
    const groupOptions = options.filter((option) => option.group === group);
    if (groupOptions.length === 0) return [];
    return [
      {
        label: selectorGroupLabel(group),
        path: `selector-group:${group}`,
        connectable: false,
        children: groupOptions.map((option) => ({
          label: option.label,
          detail: option.detail ?? option.selector.compact,
          path: option.selector.compact,
          selector: option.selector,
        })),
      },
    ];
  });
}

export function DataSourceNode({ id, data, selected }: NodeProps) {
  const nodeData = data as DataSourceNodeData;
  void nodeData;
  const workspace = useWorkspaceStore((s) => s.workspace);
  const graph = useGraphStore((s) => s.graph);
  const trainingStage = getStageByKind(workspace, 'train');
  const trainingScenario = getTrainingScenario(workspace);
  const schemaQuery = useStudioSchemaRegistry(workspace, trainingStage?.scenario_id ?? null);
  const expandedFieldPaths = useAnalysisStore((s) => s.expandedFieldPaths);
  const toggleFieldExpansion = useAnalysisStore((s) => s.toggleFieldExpansion);
  const expandedPaths = useMemo(() => new Set(expandedFieldPaths), [expandedFieldPaths]);
  const fieldTree = useMemo(() => {
    if (!workspace || !trainingScenario) return STATE_FIELD_TREE;
    const registry = buildScenarioEntityRegistry({ scenario: trainingScenario, graph });
    const objectiveSpec = ensureObjectiveSpec(trainingScenario.objective_spec);
    const selectorOptions = selectorOptionsForRegistry({
      registry,
      schemaRegistry: schemaQuery.data ?? null,
      objectiveSpec,
    });
    const selectorTree = selectorTreeFromScenarioOptions(selectorOptions);
    return selectorTree.length > 0 ? selectorTree : STATE_FIELD_TREE;
  }, [graph, schemaQuery.data, trainingScenario, workspace]);
  const visibleCount = countVisibleRows(fieldTree, expandedPaths);

  const bodyHeight = BODY_PADDING * 2 + visibleCount * FIELD_ROW_HEIGHT;
  const totalHeight = HEADER_HEIGHT + bodyHeight;

  // Notify React Flow that handles changed when expansion state changes
  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    updateNodeInternals(id);
  }, [id, visibleCount, updateNodeInternals]);

  return (
    <div
      className={clsx(
        'relative rounded-lg border bg-slate-50/80 backdrop-blur shadow-soft transition-all duration-150',
        selected
          ? 'border-brand-500 ring-1 ring-brand-500/40'
          : 'border-slate-200/80',
      )}
      style={{ width: WIDTH, height: totalHeight }}
    >
      {/* Header */}
      <div className="px-3 py-2 flex items-center justify-center border-b border-slate-100/80 rounded-t-lg">
        <Database className="w-4 h-4 text-slate-400" />
      </div>

      {/* Hierarchical field tree */}
      <div
        className="relative text-[11px] text-slate-400"
        style={{ height: bodyHeight, padding: BODY_PADDING }}
      >
        <StateFieldTree
          nodes={fieldTree}
          expandedPaths={expandedPaths}
          onToggle={toggleFieldExpansion}
          bodyPadding={BODY_PADDING}
        />
      </div>
    </div>
  );
}
