import { useRootGraph } from '@/hooks/useRootGraph';
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
import {
  compileActiveLayer,
  compileLayerStatus,
  compileTarget,
  reportForLayer,
  useDocumentCompileLayers,
} from '@/stores/compileStatusStore';
import { useProjectsStore } from '@/stores/projectsStore';

export function StatusBar() {
  const graphId = useGraphStore((state) => state.graphId);
  const graph = useGraphStore((state) => state.graph);
  const rootGraph = useRootGraph();
  const graphStack = useGraphStore((state) => state.graphStack);
  const workspace = useWorkspaceStore((state) => state.workspace);
  const { components } = useComponents();
  const documentId = useProjectsStore((state) => state.activeTabId);
  const compileLayers = useDocumentCompileLayers(documentId);
  const target = useMemo(
    () => compileTarget(documentId, { graphId, graph, graphStack }),
    [documentId, graphId, graph, graphStack],
  );
  const trainingScenario = getTrainingScenario(workspace);
  const taskBindingSpec = useMemo(() => {
    const currentGraphPath = graphStack
      .map((layer) => layer.childNodeId)
      .filter((item): item is string => Boolean(item));
    return scopedTaskBindingSpec(
      ensureTaskBindingSpec(
        trainingScenario?.task_binding_spec,
        rootGraph,
        trainingScenario?.task_spec
      ),
      currentGraphPath
    );
  }, [rootGraph, graphStack, trainingScenario?.task_binding_spec, trainingScenario?.task_spec]);
  const isAcausalLayer = isAcausalGraphSpec(graph);
  const schemaRegistry = useMemo(
    () => (isAcausalLayer ? null : projectStudioSchema(graph, components, taskBindingSpec)),
    [components, graph, isAcausalLayer, taskBindingSpec]
  );
  const validation = useMemo(
    () => (!isAcausalGraphSpec(graph) && schemaRegistry ? validateGraph(graph, schemaRegistry) : { valid: true, errors: [], warnings: [], cycles: [] }),
    [graph, schemaRegistry]
  );
  const compileLayer = target ? compileLayers.get(target.pathKey) : undefined;
  const report = reportForLayer(compileLayer, target?.kind === 'ready' ? target.hash : undefined);
  const compileStatus = target?.kind === 'ready' ? compileLayerStatus(compileLayer, target.hash)
    : target?.kind === 'invalid' ? 'error'
    : 'never_compiled';
  const failure = compileLayer?.failed;
  const hasError = report?.diagnostics?.some((diagnostic) => diagnostic.severity === 'error') ?? false;
  const status = useTrainingStore((state) => state.status);

  return (
    <footer className="shell-frame__footer h-6 px-4 border-t border-shell-divider bg-shell-chrome text-xs text-shell-muted flex items-center gap-4">
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
      {isAcausalLayer && (
        <div className="flex items-center gap-1">
          {hasError || compileStatus === 'error' ? (
            <AlertTriangle className="w-3 h-3 text-red-500" />
          ) : compileStatus === 'stale' ? (
            <AlertTriangle className="w-3 h-3 text-amber-500" />
          ) : (
            <CheckCircle2 className="w-3 h-3 text-mint-500" />
          )}
          eq {report?.summary?.equations ?? report?.summary?.eq ?? 0}
          {' / unk '}{report?.summary?.unknowns ?? report?.summary?.unk ?? 0}
          {compileStatus === 'stale' && <span className="text-amber-600">stale</span>}
          {compileStatus === 'compiling' && <span>Compiling…</span>}
          {target?.kind === 'invalid' && <span className="text-red-600">{target.message}</span>}
          {target?.kind === 'ready' && compileStatus === 'error' && (
            <button
              type="button"
              className="text-red-600 underline"
              title={failure && failure.hash === target.hash ? failure.message : undefined}
              onClick={() => void compileActiveLayer(target.key, 'retry')}
            >
              Retry compile
            </button>
          )}
        </div>
      )}
      <div className="ml-auto capitalize">{status}</div>
    </footer>
  );
}
