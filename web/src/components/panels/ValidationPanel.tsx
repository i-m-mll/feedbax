import { useCallback, useMemo, useState } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import {
  getActiveScenario,
  getTopPaneState,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { validateGraph } from '@/features/graph/validation';
import { buildScenarioEntityRegistry } from '@/features/scenario/entities';
import { buildResolvedScene } from '@/features/scenario/projections';
import { ensureTaskBindingSpec, scopedTaskBindingSpec } from '@/features/scenario/taskBindings';
import { projectStudioSchema } from '@/features/schema/project';
import { useComponents } from '@/hooks/useComponents';
import { PanelSectionHeader } from '@/components/ui/PanelPrimitives';
import { semanticTokens } from '@/components/ui/semanticTokens';
import type { DomainDiagnostic } from '@/generated/studioContracts';

type PanelDiagnostic = Pick<
  DomainDiagnostic,
  'severity' | 'code' | 'message' | 'node_ids'
>;

function diagnosticNodeIds(location: Record<string, unknown> | undefined): string[] {
  if (!location) return [];
  const direct = location.node ?? location.entity;
  return typeof direct === 'string' ? [direct] : [];
}

function severityTextClass(severity: DomainDiagnostic['severity']): string {
  if (severity === 'error') return semanticTokens.error.text;
  if (severity === 'warning') return 'text-amber-600';
  return 'text-slate-500';
}

export function ValidationPanel() {
  const [isExpanded, setIsExpanded] = useState(true);

  // state.graph is always the currently active layer (root or nested subgraph),
  // because the store updates it whenever the user enters/exits a subgraph.
  const graph = useGraphStore((state) => state.graph);
  const graphStack = useGraphStore((state) => state.graphStack);
  const currentGraphLabel = useGraphStore((state) => state.currentGraphLabel);
  const setSelectedNode = useGraphStore((state) => state.setSelectedNode);
  const isInSubgraph = graphStack.length > 0;
  const workspace = useWorkspaceStore((state) => state.workspace);
  const { components } = useComponents();
  const selectedEntityId = getTopPaneState(workspace).selected_entity_id;
  const taskBindingSpec = useMemo(() => {
    const scenario = getTrainingScenario(workspace);
    const currentGraphPath = graphStack
      .map((layer) => layer.childNodeId)
      .filter((item): item is string => Boolean(item));
    const rootGraph = graphStack.length > 0 ? graphStack[0].graph : graph;
    return scopedTaskBindingSpec(
      ensureTaskBindingSpec(scenario?.task_binding_spec, rootGraph, scenario?.task_spec),
      currentGraphPath
    );
  }, [graph, graphStack, workspace]);
  const schemaRegistry = useMemo(
    () => projectStudioSchema(graph, components, taskBindingSpec),
    [components, graph, taskBindingSpec]
  );
  const validation = useMemo(() => validateGraph(graph, schemaRegistry), [graph, schemaRegistry]);
  const sceneWarnings = useMemo(() => {
    const scenario = getActiveScenario(workspace);
    const registry = buildScenarioEntityRegistry({ scenario, graph });
    return buildResolvedScene({ scenario, graph, registry, components }).validation.map(
      (message) => ({
        type: message.type,
        message: message.message,
        location: {
          entity: message.entity_id ?? undefined,
          path: message.path ?? undefined,
        },
      })
    );
  }, [components, graph, workspace]);
  const warnings = [...validation.warnings, ...sceneWarnings];
  const diagnostics = useMemo<PanelDiagnostic[]>(() => {
    const rows: PanelDiagnostic[] = [
      ...validation.errors.map((issue) => ({
        severity: 'error' as const,
        code: `graph.${issue.type}`,
        message: issue.message,
        node_ids: diagnosticNodeIds(issue.location),
      })),
      ...warnings.map((issue) => ({
        severity: 'warning' as const,
        code: `graph.${issue.type}`,
        message: issue.message,
        node_ids: diagnosticNodeIds(issue.location),
      })),
      ...validation.cycles.map((cycle) => ({
        severity: 'error' as const,
        code: 'graph.same_step_cycle',
        message: 'Instant wires contain a same-step cycle; mark one cycle edge recurrent',
        node_ids: cycle,
      })),
    ];
    return rows;
  }, [validation.errors, validation.cycles, warnings]);

  const toggleExpanded = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  const hasIssues = diagnostics.length > 0;
  const errorCount = diagnostics.filter((diagnostic) => diagnostic.severity === 'error').length;
  const warningCount = diagnostics.filter((diagnostic) => diagnostic.severity === 'warning').length;

  if (selectedEntityId) return null;

  return (
    <div className="border-t border-slate-100">
      {/* Header */}
      <PanelSectionHeader
        title="Validation"
        expanded={isExpanded}
        onToggle={toggleExpanded}
        badges={
          <>
          {errorCount > 0 && (
            <span className={`rounded-full ${semanticTokens.error.badgeBackground} px-2 py-0.5 text-[10px] font-medium ${semanticTokens.error.badgeText} normal-case tracking-normal`}>
              {errorCount} error{errorCount !== 1 ? 's' : ''}
            </span>
          )}
          {errorCount === 0 && warningCount > 0 && (
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-500 normal-case tracking-normal">
              {warningCount} warning{warningCount !== 1 ? 's' : ''}
            </span>
          )}
          {!hasIssues && (
            <span className={`rounded-full ${semanticTokens.success.background} px-2 py-0.5 text-[10px] font-medium ${semanticTokens.success.text} normal-case tracking-normal`}>
              valid
            </span>
          )}
          </>
        }
      />

      {/* Content */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-2">
          {/* Scope indicator */}
          {isInSubgraph && (
            <div className="text-[10px] text-slate-400">
              Showing: {currentGraphLabel}
            </div>
          )}

          {diagnostics.length === 0 ? (
            <div className="text-xs text-mint-500">Graph is valid.</div>
          ) : (
            <div className="space-y-1.5">
              {diagnostics.map((diagnostic, index) => (
                <div
                  key={`${diagnostic.code}-${index}`}
                  className={`text-xs ${severityTextClass(diagnostic.severity)}`}
                >
                  <span className="font-medium">{diagnostic.code}</span>
                  {diagnostic.node_ids.map((nodeId) => {
                    const canSelect = Boolean(graph.nodes[nodeId]);
                    return canSelect ? (
                      <button
                        key={nodeId}
                        type="button"
                        className="ml-1 rounded border border-current px-1 text-[10px]"
                        onClick={() => setSelectedNode(nodeId)}
                      >
                        {nodeId}
                      </button>
                    ) : (
                      <span key={nodeId} className="ml-1 text-[10px]">
                        {nodeId}
                      </span>
                    );
                  })}
                  <span className="ml-1">{diagnostic.message}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
