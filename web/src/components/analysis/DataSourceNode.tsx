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
import { type NodeProps, useUpdateNodeInternals } from '@xyflow/react';
import type { DataSourceNodeData } from '@/stores/analysisStore';
import { useAnalysisStore } from '@/stores/analysisStore';
import { STATE_FIELD_TREE, type StateFieldNode } from '@/types/analysis';
import { buildScenarioEntityRegistry } from '@/features/scenario/entities';
import { ensureObjectiveSpec } from '@/features/scenario/objectives';
import {
  selectorOptionsForRegistry,
  type StudioSelectorOption,
} from '@/features/scenario/selectors';
import { useStudioSchemaRegistry } from '@/hooks/useStudioSchemas';
import { useGraphStore } from '@/stores/graphStore';
import { getStageByKind, getTrainingScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import type { GraphSpec } from '@/types/graph';
import { StateFieldTree, countVisibleRows, FIELD_ROW_HEIGHT } from './StateFieldTree';
import { Database } from 'lucide-react';
import { NodeHeader, NodeShell } from '@/components/ui/NodePrimitives';

const WIDTH = 200;
const HEADER_HEIGHT = 36;
const BODY_PADDING = 8;

const MODEL_VARIABLE_NAMESPACES = new Set([
  'graph_port',
  'graph_output',
  'state_path',
]);

function titleCase(value: string): string {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function compactPathKey(option: StudioSelectorOption): string {
  return option.selector.path ?? option.selector.compact;
}

function taskDataKey(option: StudioSelectorOption): string {
  const metadata = option.selector.metadata ?? {};
  const selectorSource = metadata.selector_source;
  const sourceTaskDataId =
    selectorSource && typeof selectorSource === 'object' && 'task_data_id' in selectorSource
      ? (selectorSource as { task_data_id?: unknown }).task_data_id
      : null;
  const valueSchema = metadata.value_schema;
  const valueSchemaId =
    valueSchema && typeof valueSchema === 'object' && 'id' in valueSchema
      ? (valueSchema as { id?: unknown }).id
      : null;
  const candidates = [metadata.task_data_id, sourceTaskDataId, valueSchemaId];
  for (const candidate of candidates) {
    if (typeof candidate !== 'string' || candidate.length === 0) continue;
    return candidate
      .replace(/^value:task_data:/, '')
      .replace(/^task_data:/, '');
  }
  return compactPathKey(option).replace(/^task_data:/, '');
}

function metadataString(option: StudioSelectorOption, key: string): string | null {
  const value = option.selector.metadata[key];
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function modelOwnerId(option: StudioSelectorOption): string | null {
  return (
    metadataString(option, 'graph_port_node_id') ??
    metadataString(option, 'node_id') ??
    (option.selector.namespace === 'graph_port' || option.selector.namespace === 'state_path'
      ? option.selector.target_id ?? null
      : null)
  );
}

function modelGroupLabel(nodeId: string): string {
  return titleCase(nodeId);
}

function readableVariableLabel(option: StudioSelectorOption): string {
  const selector = option.selector;
  const path = selector.path ?? selector.compact;
  const portName = metadataString(option, 'graph_port_name') ?? selector.path;
  const direction =
    metadataString(option, 'graph_port_direction') ?? metadataString(option, 'direction');

  if (selector.namespace === 'graph_port') {
    if (direction === 'input') {
      return portName && portName !== 'input' ? titleCase(portName) : 'Input';
    }
    if (direction === 'output') {
      return portName && portName !== 'output' ? titleCase(portName) : 'Output';
    }
    return portName ? titleCase(portName) : option.label;
  }

  if (selector.namespace === 'graph_output') {
    return 'Output';
  }

  if (selector.namespace === 'state_path') {
    if (path.endsWith('.hidden')) return 'Hidden state';
    if (path.endsWith('.input')) return 'Input';
    if (path.endsWith('.output')) return 'Output';
    if (path.endsWith('.pos') || path.endsWith('.position')) return 'Effector position';
    if (path.endsWith('.vel') || path.endsWith('.velocity')) return 'Effector velocity';
    if (path.endsWith('.noise')) return 'Feedback noise';
  }

  return option.label;
}

function preferTaskOption(
  current: StudioSelectorOption,
  candidate: StudioSelectorOption,
): StudioSelectorOption {
  if (current.selector.namespace !== candidate.selector.namespace) {
    if (candidate.selector.namespace === 'task_data') return candidate;
    if (current.selector.namespace === 'task_data') return current;
  }
  if (current.origin !== candidate.origin) {
    if (candidate.origin !== 'entity_registry') return candidate;
    if (current.origin !== 'entity_registry') return current;
  }
  return candidate.selector.compact.length < current.selector.compact.length ? candidate : current;
}

function selectorNode(option: StudioSelectorOption, label = option.label): StateFieldNode {
  return {
    label,
    detail: option.detail ?? option.selector.compact,
    path: option.selector.compact,
    selector: option.selector,
  };
}

function graphNodeIds(graph: GraphSpec): Set<string> {
  return new Set(Object.keys(graph.nodes));
}

export function selectorTreeFromScenarioOptions(
  options: ReturnType<typeof selectorOptionsForRegistry>,
  validModelOwnerIds?: ReadonlySet<string>,
): StateFieldNode[] {
  const modelOptionsByOwner = new Map<string, Map<string, StudioSelectorOption>>();
  const taskOptionsByKey = new Map<string, StudioSelectorOption>();

  for (const option of options) {
    if (option.group === 'task') {
      if (option.selector.namespace !== 'task_data') continue;
      const key = taskDataKey(option);
      const existing = taskOptionsByKey.get(key);
      taskOptionsByKey.set(key, existing ? preferTaskOption(existing, option) : option);
      continue;
    }

    if (!MODEL_VARIABLE_NAMESPACES.has(option.selector.namespace)) continue;
    if (option.selector.metadata.source === 'state_flow_edge') continue;
    const ownerId = modelOwnerId(option);
    if (!ownerId) continue;
    if (validModelOwnerIds && !validModelOwnerIds.has(ownerId)) continue;

    const ownerOptions = modelOptionsByOwner.get(ownerId) ?? new Map<string, StudioSelectorOption>();
    const key = compactPathKey(option);
    if (!ownerOptions.has(key)) {
      ownerOptions.set(key, option);
    }
    modelOptionsByOwner.set(ownerId, ownerOptions);
  }

  const modelGroups = [...modelOptionsByOwner.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([ownerId, ownerOptions]) => ({
      label: modelGroupLabel(ownerId),
      path: `selector-model:${ownerId}`,
      connectable: false,
      children: [...ownerOptions.values()]
        .sort((left, right) =>
          readableVariableLabel(left).localeCompare(readableVariableLabel(right))
        )
        .map((option) => selectorNode(option, readableVariableLabel(option))),
    }));

  const taskRows = [...taskOptionsByKey.values()]
    .sort((left, right) => left.label.localeCompare(right.label))
    .map((option) => selectorNode(option));

  const tree: StateFieldNode[] = [];
  if (modelGroups.length > 0) {
    tree.push({
      label: 'Model variables',
      path: 'selector-group:model-variables',
      connectable: false,
      children: modelGroups,
    });
  }
  if (taskRows.length > 0) {
    tree.push({
      label: 'Task data',
      path: 'selector-group:task-data',
      connectable: false,
      children: taskRows,
    });
  }
  return tree;
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
  const selectedDataSourceField = useAnalysisStore((s) => s.selectedDataSourceField);
  const toggleFieldExpansion = useAnalysisStore((s) => s.toggleFieldExpansion);
  const setSelectedDataSourceField = useAnalysisStore((s) => s.setSelectedDataSourceField);
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
    const selectorTree = selectorTreeFromScenarioOptions(selectorOptions, graphNodeIds(graph));
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
    <NodeShell tone="dataSource" selected={selected} style={{ width: WIDTH, height: totalHeight }}>
      {/* Header */}
      <NodeHeader tone="dataSource" className="justify-center">
        <Database className="w-4 h-4 text-slate-400" />
      </NodeHeader>

      {/* Hierarchical field tree */}
      <div
        className="relative text-[11px] text-slate-400"
        style={{ height: bodyHeight, padding: BODY_PADDING }}
      >
        <StateFieldTree
          nodes={fieldTree}
          expandedPaths={expandedPaths}
          onToggle={toggleFieldExpansion}
          onSelect={setSelectedDataSourceField}
          selectedPath={selectedDataSourceField?.path ?? null}
          bodyPadding={BODY_PADDING}
        />
      </div>
    </NodeShell>
  );
}
