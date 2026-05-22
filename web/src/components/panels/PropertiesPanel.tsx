import { useEffect, useMemo, useState } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import {
  getActiveStage,
  getScenario,
  getTopPaneState,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import {
  buildScenarioEntityRegistry,
  graphPortEntityId,
  parseGraphPortEntityId,
  probeEntityId,
} from '@/features/scenario/entities';
import {
  selectorDetail,
  selectorDisplayLabel,
  selectorOptionsForRegistry,
  type StudioSelectorOption,
} from '@/features/scenario/selectors';
import { useStudioSchemaRegistry } from '@/hooks/useStudioSchemas';
import { useComponents } from '@/hooks/useComponents';
import { ensureTaskBindingSpec } from '@/features/scenario/taskBindings';
import {
  applyBoundaryOverrides,
  deriveDimensionConstraints,
  deriveSubgraphBoundaryOverrides,
} from '@/features/schema/dimensions';
import { projectStudioSchema } from '@/features/schema/project';
import type { GraphNodeData, GraphSpec, ParamSchema, ParamValue, TapSpec } from '@/types/graph';
import type { AnalysisNodeMeta } from '@/types/analysis';
import type {
  StudioInterventionOperation,
  StudioInterventionTransformSpec,
  StudioSelectorRef,
  StudioValueSpec,
} from '@/types/workspace';
import { FigOpsSection, DependencyPortsSection } from '@/components/analysis/FigOpsSection';
import clsx from 'clsx';

export function PropertiesPanel() {
  const nodes = useGraphStore((state) => state.nodes);
  const graph = useGraphStore((state) => state.graph);
  const graphStack = useGraphStore((state) => state.graphStack);
  const updateNodeParams = useGraphStore((state) => state.updateNodeParams);
  const renameNode = useGraphStore((state) => state.renameNode);
  const renameSubgraphBoundaryPort = useGraphStore(
    (state) => state.renameSubgraphBoundaryPort
  );
  const addTap = useGraphStore((state) => state.addTap);
  const addTapForEdge = useGraphStore((state) => state.addTapForEdge);
  const updateTap = useGraphStore((state) => state.updateTap);
  const removeTap = useGraphStore((state) => state.removeTap);
  const setSelectedTap = useGraphStore((state) => state.setSelectedTap);
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
  const retargetTaskBindingsForNodeRename = useWorkspaceStore(
    (state) => state.retargetActiveScenarioTaskBindingsForNodeRename
  );
  const retargetTaskBindingsForNodePortRename = useWorkspaceStore(
    (state) => state.retargetActiveScenarioTaskBindingsForNodePortRename
  );
  const workspace = useWorkspaceStore((state) => state.workspace);
  const selectedTapId = useGraphStore((state) => state.selectedTapId);
  const selectedEdgeId = useGraphStore((state) => state.selectedEdgeId);
  const edges = useGraphStore((state) => state.edges);
  const { components } = useComponents();
  const activeStage = getActiveStage(workspace);
  const topPane = getTopPaneState(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);
  const trainingScenario = getTrainingScenario(workspace);
  const schemaQuery = useStudioSchemaRegistry(
    workspace,
    activeStage?.scenario_id ?? activeScenario?.id ?? null
  );
  const scenarioRegistry = useMemo(
    () => buildScenarioEntityRegistry({ scenario: activeScenario, graph }),
    [activeScenario, graph]
  );
  const selectorOptions = useMemo(
    () =>
      selectorOptionsForRegistry({
        registry: scenarioRegistry,
        schemaRegistry: schemaQuery.data ?? null,
      }),
    [scenarioRegistry, schemaQuery.data]
  );

  const selectedNode = useMemo(
    () => nodes.find((node) => node.selected && node.type !== 'tap'),
    [nodes]
  );
  const taskBindingSpec = useMemo(
    () =>
      ensureTaskBindingSpec(
        trainingScenario?.task_binding_spec,
        graph,
        trainingScenario?.task_spec
      ),
    [graph, trainingScenario?.task_binding_spec, trainingScenario?.task_spec]
  );
  const parentBoundaryOverrides = useMemo(() => {
    const parentLayer = graphStack[graphStack.length - 1];
    if (!parentLayer?.childNodeId) return new Map();
    const parentTaskBindingSpec = ensureTaskBindingSpec(
      trainingScenario?.task_binding_spec,
      parentLayer.graph,
      trainingScenario?.task_spec
    );
    const parentRegistry = projectStudioSchema(
      parentLayer.graph,
      components,
      parentTaskBindingSpec
    );
    return deriveSubgraphBoundaryOverrides(
      parentLayer.graph,
      parentLayer.childNodeId,
      parentRegistry
    );
  }, [components, graphStack, trainingScenario?.task_binding_spec, trainingScenario?.task_spec]);
  const localSchemaRegistry = useMemo(
    () =>
      applyBoundaryOverrides(
        projectStudioSchema(graph, components, taskBindingSpec),
        parentBoundaryOverrides
      ),
    [components, graph, parentBoundaryOverrides, taskBindingSpec]
  );
  const nodeDimensionConstraints = useMemo(
    () =>
      selectedNode
        ? deriveDimensionConstraints(graph, localSchemaRegistry).filter(
            (constraint) => constraint.node_id === selectedNode.id
          )
        : [],
    [graph, localSchemaRegistry, selectedNode]
  );
  const constraintsByParam = useMemo(
    () => new Map(nodeDimensionConstraints.map((constraint) => [constraint.param, constraint])),
    [nodeDimensionConstraints]
  );
  const taps = graph.taps ?? [];
  const selectedTap = selectedTapId
    ? taps.find((tap) => tap.id === selectedTapId)
    : undefined;
  const selectedEdge = selectedEdgeId
    ? edges.find((edge) => edge.id === selectedEdgeId)
    : undefined;
  const selectedPort = parseGraphPortEntityId(topPane.selected_entity_id);

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
        selectorOptions={selectorOptions}
        onUpdate={(updates) => updateTap(selectedTap.id, updates)}
        onRemove={() => removeTap(selectedTap.id)}
      />
    );
  }

  if (selectedEdge && selectedEdge.type === 'state-flow') {
    return (
      <div className="space-y-5 p-6">
        <div>
          <div className="text-sm font-medium text-slate-800">
            {selectedEdge.source} → {selectedEdge.target}
          </div>
          <div className="mt-1 text-xs text-slate-500">Full state flow</div>
        </div>
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <button
              className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600 hover:text-slate-800"
              onClick={() => addTapForEdge(selectedEdge.id, 'probe')}
            >
              Add Probe Tap
            </button>
            <button
              className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600 hover:text-slate-800"
              onClick={() => addTapForEdge(selectedEdge.id, 'intervention')}
            >
              Add Intervention Tap
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (selectedEdge && selectedEdge.type !== 'state-flow') {
    const edgeData = selectedEdge.data;
    const temporality = edgeData?.temporality ?? 'instant';
    const init = edgeData?.recurrent_initializer as Record<string, unknown> | null | undefined;
    return (
      <div className="space-y-5 p-6">
        <div>
          <div className="text-sm font-medium text-slate-800">
            {selectedEdge.source}.{selectedEdge.sourceHandle} → {selectedEdge.target}.
            {selectedEdge.targetHandle}
          </div>
          <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
            <span
              className={clsx(
                'rounded-full px-2 py-0.5 font-medium',
                temporality === 'recurrent'
                  ? 'bg-sky-50 text-sky-700'
                  : 'bg-slate-100 text-slate-600'
              )}
            >
              {temporality === 'recurrent' ? 'Recurrent t+1' : 'Instant'}
            </span>
            {edgeData?.schema_status && (
              <span className="rounded-full bg-amber-50 px-2 py-0.5 font-medium text-amber-700">
                {edgeData.schema_status}
              </span>
            )}
          </div>
        </div>
        {temporality === 'recurrent' ? (
          <div className="space-y-2 border-t border-slate-100 pt-4">
            <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Initializer</div>
            <div className="text-sm text-slate-600">
              {init?.kind === 'zeros'
                ? `Zeros${Array.isArray(init.shape) ? ` ${JSON.stringify(init.shape)}` : ''}`
                : init
                  ? String(init.kind ?? 'Custom initializer')
                  : 'Missing recurrent initial value'}
            </div>
          </div>
        ) : (
          <div className="text-xs text-slate-400">
            Same-step dataflow. Same-step cycles must be cut by a recurrent edge.
          </div>
        )}
        {edgeData?.schema_message && (
          <div className="rounded-md border border-amber-100 bg-amber-50 px-3 py-2 text-xs text-amber-800">
            {edgeData.schema_message}
          </div>
        )}
        <div className="text-xs text-slate-400">
          Port wires carry data; component-owned state stays in node state slots.
        </div>
      </div>
    );
  }

  if (selectedPort) {
    const nodeSpec = graph.nodes[selectedPort.nodeId];
    const subgraph = graph.subgraphs?.[selectedPort.nodeId];
    const boundaryPorts =
      selectedPort.direction === 'input' ? subgraph?.input_ports : subgraph?.output_ports;
    const isBoundaryAlias = Boolean(boundaryPorts?.includes(selectedPort.port));
    const binding =
      selectedPort.direction === 'input'
        ? subgraph?.input_bindings[selectedPort.port]
        : subgraph?.output_bindings[selectedPort.port];
    const ports =
      selectedPort.direction === 'input' ? nodeSpec?.input_ports : nodeSpec?.output_ports;
    if (nodeSpec && ports?.includes(selectedPort.port)) {
      return (
        <PortPropertiesPanel
          nodeId={selectedPort.nodeId}
          nodeType={nodeSpec.type}
          direction={selectedPort.direction}
          port={selectedPort.port}
          binding={binding}
          isBoundaryAlias={isBoundaryAlias}
          onRename={(nextPort) => {
            const trimmed = nextPort.trim();
            if (!trimmed || trimmed === selectedPort.port || !isBoundaryAlias) return;
            const existingPorts = selectedPort.direction === 'input'
              ? subgraph?.input_ports ?? []
              : subgraph?.output_ports ?? [];
            if (existingPorts.includes(trimmed)) return;
            if (selectedPort.direction === 'input') {
              retargetTaskBindingsForNodePortRename(
                selectedPort.nodeId,
                selectedPort.port,
                trimmed
              );
            }
            renameSubgraphBoundaryPort(
              selectedPort.nodeId,
              selectedPort.direction,
              selectedPort.port,
              trimmed
            );
            selectTopPaneEntity(
              graphPortEntityId(selectedPort.nodeId, selectedPort.direction, trimmed),
              'graph_port_alias_renamed'
            );
          }}
        />
      );
    }
  }

  if (!selectedNode) {
    return (
      <div className="p-6 text-sm text-slate-500">
        Select a node or tap on the canvas to view properties.
      </div>
    );
  }

  const nodeSpec = graph.nodes[selectedNode.id];
  const selectedSubgraph = graph.subgraphs?.[selectedNode.id];
  const component = nodeSpec
    ? components.find((item) => item.name === nodeSpec.type)
    : undefined;
  const nodeTaps = taps.filter((tap) => tap.position.afterNode === selectedNode.id);
  // Check for analysis-specific metadata on the node spec
  const analysisMeta = nodeSpec?.params?._analysis_meta as unknown as AnalysisNodeMeta | undefined;

  const commitRename = () => {
    const nextNodeId = nameValue.trim();
    if (nextNodeId && nextNodeId !== selectedNode.id && !graph.nodes[nextNodeId]) {
      retargetTaskBindingsForNodeRename(selectedNode.id, nextNodeId);
      renameNode(selectedNode.id, nextNodeId);
    }
  };

  const commitBoundaryPortRename = (
    direction: 'input' | 'output',
    previousPort: string,
    nextPort: string
  ) => {
    const trimmed = nextPort.trim();
    if (!selectedSubgraph || !trimmed || trimmed === previousPort) return;
    const existingPorts = direction === 'input'
      ? selectedSubgraph.input_ports
      : selectedSubgraph.output_ports;
    if (existingPorts.includes(trimmed)) return;
    if (direction === 'input') {
      retargetTaskBindingsForNodePortRename(selectedNode.id, previousPort, trimmed);
    }
    renameSubgraphBoundaryPort(selectedNode.id, direction, previousPort, trimmed);
  };

  if (!nodeSpec) {
    return <div className="p-6 text-sm text-slate-500">Node data is missing.</div>;
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <input
          aria-label="Node name"
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
          className="-mx-1 w-full rounded-md border border-transparent bg-transparent px-1 py-1 text-sm font-semibold text-slate-800 outline-none hover:border-slate-200 focus:border-brand-300 focus:bg-white"
        />
        <div className="mt-1 text-sm text-slate-500">{nodeSpec.type}</div>
      </div>

      {component?.is_composite ? (
        <div className="space-y-3">
          <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Parameters</div>
          <div className="text-sm text-slate-400">
            Enter this component to edit its internal structure.
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Parameters</div>
          {(component?.param_schema ?? []).map((param) => (
            <div key={param.name} className="space-y-1">
              <ParamInput
                schema={param}
                value={nodeSpec.params[param.name] ?? param.default ?? null}
                onChange={(value) =>
                  updateNodeParams(
                    selectedNode.id,
                    param.name,
                    value,
                    taskBindingSpec
                  )
                }
              />
              {constraintsByParam.has(param.name) && (
                <DimensionConstraintHint
                  status={constraintsByParam.get(param.name)!.status}
                  value={constraintsByParam.get(param.name)!.inferred_value}
                  message={constraintsByParam.get(param.name)!.message}
                />
              )}
            </div>
          ))}
          {!component && (
            <div className="text-sm text-slate-400">No schema for this component yet.</div>
          )}
        </div>
      )}

      <div className="border-t border-slate-100 pt-4">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">Ports</div>
        {selectedSubgraph ? (
          <BoundaryPortsSection
            graph={selectedSubgraph}
            onRename={commitBoundaryPortRename}
          />
        ) : (
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
        )}
      </div>

      {((selectedNode.data as { state_slots?: GraphNodeData['state_slots'] }).state_slots ?? []).length > 0 && (
        <div className="border-t border-slate-100 pt-4">
          <div className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">
            Recurrence
          </div>
          <div className="space-y-2">
            {((selectedNode.data as { state_slots?: GraphNodeData['state_slots'] }).state_slots ?? []).map((slot) => (
              <div
                key={slot.id}
                className="rounded-md border border-slate-100 px-3 py-2 text-xs text-slate-600"
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="font-medium text-slate-700">{slot.label}</span>
                  <span className="text-slate-400">
                    {Array.isArray(slot.shape) ? JSON.stringify(slot.shape) : 'shape unknown'}
                  </span>
                </div>
                <div className="mt-1 text-slate-400">
                  {slot.initializer?.kind === 'zeros'
                    ? 'Zeros before first timestep'
                    : 'Custom initializer'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

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
                onClick={() => {
                  setSelectedTap(tap.id);
                  selectTopPaneEntity(probeEntityId(tap.id));
                }}
              >
                <span className="font-medium capitalize">{tap.type}</span>
                <span className="text-slate-400">{Object.keys(tap.paths ?? {}).length} outputs</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Analysis-specific sections */}
      {analysisMeta && (
        <>
          <div className="border-t border-slate-100 pt-4">
            <div className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-2">
              Analysis
            </div>
            <div className="text-sm text-slate-600">{analysisMeta.analysis_class}</div>
            {analysisMeta.has_make_figs && (
              <div className="mt-1 text-xs text-emerald-600">Produces figures</div>
            )}
          </div>
          <FigOpsSection meta={analysisMeta} />
          <DependencyPortsSection
            ports={analysisMeta.dependency_ports}
            nodeId={selectedNode.id}
          />
        </>
      )}
    </div>
  );
}

function BoundaryPortsSection({
  graph,
  onRename,
}: {
  graph: GraphSpec;
  onRename: (direction: 'input' | 'output', previousPort: string, nextPort: string) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-4 text-xs text-slate-600">
      <div>
        <div className="font-semibold text-slate-500 mb-2">Inputs</div>
        <div className="space-y-2">
          {graph.input_ports.length === 0 ? (
            <div className="text-slate-400">None</div>
          ) : (
            graph.input_ports.map((port) => (
              <BoundaryPortRow
                key={port}
                port={port}
                binding={graph.input_bindings[port]}
                direction="input"
                onRename={onRename}
              />
            ))
          )}
        </div>
      </div>
      <div>
        <div className="font-semibold text-slate-500 mb-2">Outputs</div>
        <div className="space-y-2">
          {graph.output_ports.length === 0 ? (
            <div className="text-slate-400">None</div>
          ) : (
            graph.output_ports.map((port) => (
              <BoundaryPortRow
                key={port}
                port={port}
                binding={graph.output_bindings[port]}
                direction="output"
                onRename={onRename}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function PortPropertiesPanel({
  nodeId,
  nodeType,
  direction,
  port,
  binding,
  isBoundaryAlias,
  onRename,
}: {
  nodeId: string;
  nodeType: string;
  direction: 'input' | 'output';
  port: string;
  binding?: [string, string];
  isBoundaryAlias: boolean;
  onRename: (nextPort: string) => void;
}) {
  const [value, setValue] = useState(port);

  useEffect(() => {
    setValue(port);
  }, [port]);

  const commit = () => {
    const trimmed = value.trim();
    if (!trimmed || trimmed === port) {
      setValue(port);
      return;
    }
    onRename(trimmed);
  };

  return (
    <div className="space-y-5 p-6">
      <div>
        <div className="text-sm font-medium text-slate-800">
          {nodeId}.{port}
        </div>
        <div className="mt-1 text-xs text-slate-500">
          {direction === 'input' ? 'Input' : 'Output'} port on {nodeType}
        </div>
      </div>
      <div className="space-y-2 border-t border-slate-100 pt-4">
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Alias</div>
        {isBoundaryAlias ? (
          <label className="block space-y-1">
            <input
              aria-label={`${direction} port alias ${port}`}
              value={value}
              onChange={(event) => setValue(event.target.value)}
              onBlur={commit}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.currentTarget.blur();
                }
                if (event.key === 'Escape') {
                  setValue(port);
                  event.currentTarget.blur();
                }
              }}
              className="w-full rounded-md border border-slate-200 px-2 py-1.5 text-sm text-slate-700 outline-none focus:border-brand-300"
            />
            <div
              className="truncate text-[10px] text-slate-400"
              title={binding?.join('.') ?? 'Unbound'}
            >
              {binding ? `Internal: ${binding[0]}.${binding[1]}` : 'Internal: unbound'}
            </div>
          </label>
        ) : (
          <div className="rounded-md border border-slate-100 bg-slate-50 px-2 py-1.5 text-sm text-slate-600">
            {port}
          </div>
        )}
      </div>
    </div>
  );
}

function BoundaryPortRow({
  port,
  binding,
  direction,
  onRename,
}: {
  port: string;
  binding?: [string, string];
  direction: 'input' | 'output';
  onRename: (direction: 'input' | 'output', previousPort: string, nextPort: string) => void;
}) {
  const [value, setValue] = useState(port);

  useEffect(() => {
    setValue(port);
  }, [port]);

  const commit = () => {
    const trimmed = value.trim();
    if (!trimmed || trimmed === port) {
      setValue(port);
      return;
    }
    onRename(direction, port, trimmed);
  };

  return (
    <label className="block space-y-1">
      <input
        aria-label={`${direction} port ${port}`}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            event.currentTarget.blur();
          }
          if (event.key === 'Escape') {
            setValue(port);
            event.currentTarget.blur();
          }
        }}
        className="w-full rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-700 outline-none focus:border-brand-300"
      />
      <div className="truncate text-[10px] text-slate-400" title={binding?.join('.') ?? 'Unbound'}>
        {binding ? `${binding[0]}.${binding[1]}` : 'Unbound'}
      </div>
    </label>
  );
}

function parseNumericDraft(value: string): number | null {
  if (!value.trim()) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function constantInterventionValue(
  value: unknown,
  intervention: StudioInterventionTransformSpec
): StudioValueSpec {
  return {
    schema_version: 'feedbax.studio.value.v1',
    mode: 'constant',
    value,
    dtype: intervention.target_selector?.dtype ?? null,
    shape: intervention.target_selector?.expected_shape ?? null,
    units: intervention.target_selector?.units ?? null,
    frame: intervention.target_selector?.frame ?? null,
    metadata: {},
  };
}

function defaultInterventionSpec(
  selector: StudioSelectorRef | null | undefined
): StudioInterventionTransformSpec {
  return {
    operation: 'clamp',
    target_selector: selector ?? null,
    bounds: null,
    value: null,
    parameters: null,
    metadata: {},
  };
}

function TapEditor({
  tap,
  nodeIds,
  selectorOptions,
  onUpdate,
  onRemove,
}: {
  tap: TapSpec;
  nodeIds: string[];
  selectorOptions: StudioSelectorOption[];
  onUpdate: (updates: Partial<TapSpec>) => void;
  onRemove: () => void;
}) {
  const [newOutputName, setNewOutputName] = useState('');
  const [newOutputPath, setNewOutputPath] = useState('');

  const transform = tap.transform ?? { type: 'custom', params: {} };
  const intervention =
    transform.intervention ?? defaultInterventionSpec(selectorOptions[0]?.selector);
  const [transformJson, setTransformJson] = useState(
    JSON.stringify(transform.params ?? {}, null, 2)
  );
  const [transformType, setTransformType] = useState(transform.type ?? 'custom');
  const [interventionValueJson, setInterventionValueJson] = useState(
    JSON.stringify(intervention.value?.value ?? null, null, 2)
  );

  useEffect(() => {
    setTransformType(transform.type ?? 'custom');
    setTransformJson(JSON.stringify(transform.params ?? {}, null, 2));
    setInterventionValueJson(
      JSON.stringify(transform.intervention?.value?.value ?? null, null, 2)
    );
  }, [tap.id, tap.transform]);

  const updateIntervention = (updates: Partial<StudioInterventionTransformSpec>) => {
    onUpdate({
      transform: {
        type: transformType || transform.type || 'intervention',
        params: transform.params ?? {},
        intervention: {
          ...intervention,
          ...updates,
          metadata: {
            ...(intervention.metadata ?? {}),
            ...(updates.metadata ?? {}),
          },
        },
      },
    });
  };

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
        transform: tap.transform ?? {
          type: 'intervention',
          params: {},
          intervention: defaultInterventionSpec(selectorOptions[0]?.selector),
        },
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
            Targeting
          </div>
          <label className="flex flex-col gap-1 text-xs text-slate-500">
            Operation
            <select
              value={intervention.operation}
              onChange={(event) =>
                updateIntervention({
                  operation: event.target.value as StudioInterventionOperation,
                })
              }
              className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700"
            >
              <option value="clamp">Clamp</option>
              <option value="noise">Noise</option>
              <option value="constant">Constant</option>
              <option value="offset">Offset</option>
              <option value="scale">Scale</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-500">
            Target
            <select
              value={intervention.target_selector?.compact ?? ''}
              onChange={(event) => {
                const option = selectorOptions.find(
                  (candidate) => candidate.selector.compact === event.target.value
                );
                updateIntervention({ target_selector: option?.selector ?? null });
              }}
              className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700"
            >
              <option value="">Select target</option>
              {selectorOptions.map((option) => (
                <option key={option.id} value={option.selector.compact}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          {intervention.target_selector && (
            <div className="rounded-md border border-slate-100 bg-slate-50 px-2 py-1.5 text-xs text-slate-500">
              <div className="font-medium text-slate-700">
                {selectorDisplayLabel(intervention.target_selector)}
              </div>
              <div className="mt-0.5">{selectorDetail(intervention.target_selector)}</div>
            </div>
          )}
          {intervention.operation === 'clamp' ? (
            <div className="grid grid-cols-2 gap-2">
              <label className="flex flex-col gap-1 text-xs text-slate-500">
                Min
                <input
                  value={String(intervention.bounds?.min ?? '')}
                  onChange={(event) =>
                    updateIntervention({
                      bounds: {
                        ...(intervention.bounds ?? {}),
                        min: parseNumericDraft(event.target.value),
                      },
                    })
                  }
                  className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-slate-500">
                Max
                <input
                  value={String(intervention.bounds?.max ?? '')}
                  onChange={(event) =>
                    updateIntervention({
                      bounds: {
                        ...(intervention.bounds ?? {}),
                        max: parseNumericDraft(event.target.value),
                      },
                    })
                  }
                  className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700"
                />
              </label>
            </div>
          ) : (
            <label className="flex flex-col gap-1 text-xs text-slate-500">
              Value (JSON)
              <textarea
                rows={3}
                value={interventionValueJson}
                onChange={(event) => setInterventionValueJson(event.target.value)}
                onBlur={() => {
                  try {
                    updateIntervention({
                      value: constantInterventionValue(
                        JSON.parse(interventionValueJson),
                        intervention
                      ),
                    });
                  } catch {
                    // ignore invalid JSON
                  }
                }}
                className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700 font-mono"
              />
            </label>
          )}
          <div className="text-xs uppercase tracking-[0.3em] text-slate-400">
            Runtime Transform
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
                    intervention: transform.intervention ?? null,
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
                      intervention: transform.intervention ?? null,
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

function DimensionConstraintHint({
  status,
  value,
  message,
}: {
  status: 'inferred' | 'conflict' | 'unknown';
  value: number | null;
  message: string;
}) {
  return (
    <div
      className={clsx(
        'rounded border px-2 py-1 text-[11px]',
        status === 'inferred' && 'border-emerald-200 bg-emerald-50 text-emerald-700',
        status === 'conflict' && 'border-amber-200 bg-amber-50 text-amber-800',
        status === 'unknown' && 'border-slate-200 bg-slate-50 text-slate-500'
      )}
      title={message}
    >
      {status === 'inferred' && value !== null ? `Synced: ${value}` : message}
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
