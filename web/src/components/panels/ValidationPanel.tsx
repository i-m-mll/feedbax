import { useCallback, useMemo, useState } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import {
  getTopPaneState,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { validateGraph } from '@/features/graph/validation';
import { ensureTaskBindingSpec, scopedTaskBindingSpec } from '@/features/scenario/taskBindings';
import { projectStudioSchema } from '@/features/schema/project';
import { useComponents } from '@/hooks/useComponents';
import { PanelSectionHeader } from '@/components/ui/PanelPrimitives';
import { semanticTokens } from '@/components/ui/semanticTokens';

export function ValidationPanel() {
  const [isExpanded, setIsExpanded] = useState(true);

  // state.graph is always the currently active layer (root or nested subgraph),
  // because the store updates it whenever the user enters/exits a subgraph.
  const graph = useGraphStore((state) => state.graph);
  const graphStack = useGraphStore((state) => state.graphStack);
  const currentGraphLabel = useGraphStore((state) => state.currentGraphLabel);
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

  const toggleExpanded = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  const hasIssues = !validation.valid || validation.warnings.length > 0 || validation.cycles.length > 0;
  const errorCount = validation.errors.length;
  const warningCount = validation.warnings.length + validation.cycles.length;

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

          {validation.valid && validation.warnings.length === 0 && validation.cycles.length === 0 ? (
            <div className="text-xs text-mint-500">Graph is valid.</div>
          ) : (
            <div className="space-y-1.5">
              {validation.errors.map((error, index) => (
                <div key={`error-${index}`} className={`text-xs ${semanticTokens.error.text}`}>
                  {error.message}
                </div>
              ))}
              {validation.warnings.map((warning, index) => (
                <div key={`warning-${index}`} className="text-xs text-slate-500">
                  {warning.message}
                </div>
              ))}
              {validation.cycles.length > 0 && (
                <div className="text-xs text-purple-500">
                  Cycles: {validation.cycles.map((cycle) => cycle.join(' → ')).join(', ')}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
