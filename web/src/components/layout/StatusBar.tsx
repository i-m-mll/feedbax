import { CheckCircle2, Circle, AlertTriangle } from 'lucide-react';
import { useMemo } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import { validateGraph } from '@/features/graph/validation';
import { ensureTaskBindingSpec, scopedTaskBindingSpec } from '@/features/scenario/taskBindings';
import { projectStudioSchema } from '@/features/schema/project';
import { useTrainingStore } from '@/stores/trainingStore';
import { getTrainingScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import { useComponents } from '@/hooks/useComponents';
import { isAcausalGraphSpec } from '@/types/graph';
import { compileStatusForReport, graphPathKey, stableGraphHash } from '@/features/domains/acausal';
import { useCompileStatusStore } from '@/stores/compileStatusStore';

export function StatusBar() {
  const graph = useGraphStore((state) => state.graph);
  const graphStack = useGraphStore((state) => state.graphStack);
  const workspace = useWorkspaceStore((state) => state.workspace);
  const { components } = useComponents();
  const reports = useCompileStatusStore((state) => state.reports);
  const compilingPaths = useCompileStatusStore((state) => state.compilingPaths);
  const trainingScenario = getTrainingScenario(workspace);
  const taskBindingSpec = useMemo(() => {
    const currentGraphPath = graphStack
      .map((layer) => layer.childNodeId)
      .filter((item): item is string => Boolean(item));
    const rootGraph = graphStack.length > 0 ? graphStack[0].graph : graph;
    return scopedTaskBindingSpec(
      ensureTaskBindingSpec(
        trainingScenario?.task_binding_spec,
        rootGraph,
        trainingScenario?.task_spec
      ),
      currentGraphPath
    );
  }, [graph, graphStack, trainingScenario?.task_binding_spec, trainingScenario?.task_spec]);
  const isAcausalLayer = isAcausalGraphSpec(graph);
  const schemaRegistry = useMemo(
    () => (isAcausalLayer ? null : projectStudioSchema(graph, components, taskBindingSpec)),
    [components, graph, isAcausalLayer, taskBindingSpec]
  );
  const validation = useMemo(
    () => (schemaRegistry ? validateGraph(graph, schemaRegistry) : { valid: true, errors: [], warnings: [], cycles: [] }),
    [graph, schemaRegistry]
  );
  const acausalBalance = useMemo(() => {
    if (!isAcausalLayer) return null;
    const path = graphPathKey(
      graphStack.map((layer) => layer.childNodeId).filter((item): item is string => Boolean(item))
    );
    const report = reports[path];
    const status = compileStatusForReport(
      report,
      stableGraphHash(graph),
      compilingPaths.has(path)
    );
    const hasError = report?.diagnostics?.some((diagnostic) => diagnostic.severity === 'error') ?? false;
    return {
      status,
      hasError,
      equations: report?.summary?.equations ?? report?.summary?.eq ?? 0,
      unknowns: report?.summary?.unknowns ?? report?.summary?.unk ?? 0,
    };
  }, [compilingPaths, graph, graphStack, isAcausalLayer, reports]);
  const status = useTrainingStore((state) => state.status);

  return (
    <footer className="h-6 px-4 border-t border-slate-100 bg-white/80 text-xs text-slate-500 flex items-center gap-4">
      <div className="flex items-center gap-1">
        <Circle className="w-2.5 h-2.5 text-mint-500" fill="currentColor" />
        Connected
      </div>
      <div className="flex items-center gap-1">
        {validation.valid ? (
          <CheckCircle2 className="w-3 h-3 text-mint-500" />
        ) : (
          <AlertTriangle className="w-3 h-3 text-amber-500" />
        )}
        {validation.valid ? 'Valid graph' : `${validation.errors.length} issue(s)`}
      </div>
      {acausalBalance && (
        <div className="flex items-center gap-1">
          {acausalBalance.hasError || acausalBalance.status === 'error' ? (
            <AlertTriangle className="w-3 h-3 text-red-500" />
          ) : acausalBalance.status === 'stale' ? (
            <AlertTriangle className="w-3 h-3 text-amber-500" />
          ) : (
            <CheckCircle2 className="w-3 h-3 text-mint-500" />
          )}
          eq {acausalBalance.equations} / unk {acausalBalance.unknowns}
          {acausalBalance.status === 'stale' && <span className="text-amber-600">stale</span>}
        </div>
      )}
      <div className="ml-auto capitalize">{status}</div>
    </footer>
  );
}
