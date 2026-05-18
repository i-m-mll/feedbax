import { useTrainingStore } from '@/stores/trainingStore';
import {
  getActiveStage,
  getScenario,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { useGraphStore } from '@/stores/graphStore';
import { buildScenarioEntityRegistry } from '@/features/scenario/entities';
import {
  addObjectiveTerm,
  createObjectiveTerm,
  ensureObjectiveSpec,
  targetSelectorForEntity,
} from '@/features/scenario/objectives';
import type { LossTermSpec } from '@/types/training';
import type { StudioSelectorRef } from '@/types/workspace';
import { useCallback, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Crosshair, ListPlus } from 'lucide-react';

interface PortContextMenuProps {
  x: number;
  y: number;
  nodeName: string;
  portName: string;
  portType: 'input' | 'output';
  onClose: () => void;
}

export function PortContextMenu({
  x,
  y,
  nodeName,
  portName,
  portType,
  onClose,
}: PortContextMenuProps) {
  const addLossTerm = useTrainingStore((state) => state.addLossTerm);
  const trainingStoreSpec = useTrainingStore((state) => state.trainingSpec);
  const workspace = useWorkspaceStore((state) => state.workspace);
  const trainingScenario = getTrainingScenario(workspace);
  const trainingSpec = trainingScenario?.training_spec ?? trainingStoreSpec;
  const updateActiveScenarioObjectiveSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioObjectiveSpec
  );
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
  const setTopPaneProjection = useWorkspaceStore((state) => state.setTopPaneProjection);
  const graph = useGraphStore((state) => state.graph);
  const setSelectedNode = useGraphStore((state) => state.setSelectedNode);
  const setSelectedTap = useGraphStore((state) => state.setSelectedTap);
  const setSelectedEdge = useGraphStore((state) => state.setSelectedEdge);
  const activeStage = getActiveStage(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);
  const objectiveSpec = ensureObjectiveSpec(activeScenario?.objective_spec);
  const registry = buildScenarioEntityRegistry({ scenario: activeScenario, graph });
  const taskEntity =
    Object.values(registry.entities).find((entity) => entity.kind === 'task_object') ?? null;
  const menuRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [onClose]);

  // Close on escape
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose]);

  const handleAddProbe = useCallback(() => {
    // Generate a unique key for the new loss term
    const baseKey = `${nodeName}_${portName}`.toLowerCase().replace(/[^a-z0-9]/g, '_');
    let key = baseKey;
    let counter = 1;

    // Find existing keys to avoid collision
    const existingKeys = new Set<string>();
    const collectKeys = (term: LossTermSpec) => {
      if (term.children) {
        Object.keys(term.children).forEach((k) => {
          existingKeys.add(k);
          collectKeys(term.children![k]);
        });
      }
    };
    collectKeys(trainingSpec.loss);

    while (existingKeys.has(key)) {
      key = `${baseKey}_${counter}`;
      counter++;
    }

    // Create the new loss term
    const newTerm: LossTermSpec = {
      type: 'TargetStateLoss',
      label: `${nodeName} ${portName}`,
      weight: 1.0,
      selector: `port:${nodeName}.${portName}`,
      norm: 'squared_l2',
      time_agg: {
        mode: 'all',
      },
    };

    // Add to the root loss term (as a child of the composite)
    addLossTerm([], key, newTerm);
    onClose();
  }, [nodeName, portName, trainingSpec.loss, addLossTerm, onClose]);

  const handleAddObjective = useCallback(() => {
    if (!activeScenario) return;
    const sourceSelector: StudioSelectorRef = {
      namespace: 'graph_port',
      compact: `port:${nodeName}.${portName}`,
      target_id: nodeName,
      path: portName,
      role: 'observed',
      metadata: { direction: portType },
    };
    const term = createObjectiveTerm({
      spec: objectiveSpec,
      label: `Objective: ${nodeName}.${portName}`,
      sourceSelector,
      targetSelector: targetSelectorForEntity(taskEntity),
    });
    updateActiveScenarioObjectiveSpec(addObjectiveTerm(objectiveSpec, term));
    setSelectedNode(null);
    setSelectedTap(null);
    setSelectedEdge(null);
    setTopPaneProjection('objectives');
    selectTopPaneEntity(`objective_term:${term.id}`);
    onClose();
  }, [
    activeScenario,
    nodeName,
    objectiveSpec,
    onClose,
    portName,
    portType,
    selectTopPaneEntity,
    setSelectedEdge,
    setSelectedNode,
    setSelectedTap,
    setTopPaneProjection,
    taskEntity,
    updateActiveScenarioObjectiveSpec,
  ]);

  return createPortal(
    <div
      ref={menuRef}
      className="nodrag nopan fixed z-[1000] min-w-40 rounded-lg border border-slate-200 bg-white py-1 shadow-lg"
      style={{ left: x, top: y }}
      onPointerDown={(event) => event.stopPropagation()}
      onMouseDown={(event) => event.stopPropagation()}
      onClick={(event) => event.stopPropagation()}
      onContextMenu={(event) => {
        event.preventDefault();
        event.stopPropagation();
      }}
    >
      <button
        type="button"
        disabled={!activeScenario}
        onClick={(event) => {
          event.stopPropagation();
          handleAddObjective();
        }}
        className="w-full px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 flex items-center gap-2"
      >
        <ListPlus className="w-4 h-4 text-brand-500" />
        Add objective
      </button>
      {portType === 'output' && (
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            handleAddProbe();
          }}
          className="w-full px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 flex items-center gap-2"
        >
          <Crosshair className="w-4 h-4 text-brand-500" />
          Add probe here
        </button>
      )}
      <div className="border-t border-slate-100 my-1" />
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          onClose();
        }}
        className="w-full px-3 py-2 text-left text-sm text-slate-500 hover:bg-slate-50"
      >
        Cancel
      </button>
    </div>,
    document.body
  );
}
