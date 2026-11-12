import { useState } from 'react';
import {
  emptyPipelineSelection, selectStep, type PipelineSelectionState,
} from '@/features/pipeline/selection';

interface SelectionEntry {
  readonly pipeline: PipelineSelectionState;
  readonly stepId: string | null;
  readonly variantId: string | null;
}

function subject(pipeline: PipelineSelectionState, current: string | null) {
  const selection = pipeline.selection;
  if (!selection) return null;
  if (selection.kind !== 'steps') return selection.stepId;
  return current && selection.stepIds.includes(current) ? current : selection.stepIds.at(-1) ?? null;
}

export function useProjectWindowSelection(initialStepId: string | null) {
  const [history, setHistory] = useState<{ entries: readonly SelectionEntry[]; index: number }>(() => ({
    entries: [{
      stepId: initialStepId,
      variantId: null,
      pipeline: initialStepId ? selectStep(emptyPipelineSelection, initialStepId) : emptyPipelineSelection,
    }],
    index: 0,
  }));
  const current = history.entries[history.index];

  const change = (update: (entry: SelectionEntry) => SelectionEntry) => setHistory(({ entries, index }) => {
    const previous = entries[index];
    const next = update(previous);
    if (JSON.stringify(previous) === JSON.stringify(next)) return { entries, index };
    return { entries: [...entries.slice(0, index + 1), next], index: index + 1 };
  });

  return {
    ...current,
    select: (pipeline: PipelineSelectionState, variantId?: string | null) => change((entry) => ({
      variantId: variantId === undefined ? entry.variantId : variantId,
      pipeline, stepId: subject(pipeline, entry.stepId),
    })),
    setVariant: (variantId: string | null) => change((entry) => ({
      ...entry,
      variantId,
      pipeline: entry.stepId ? selectStep(emptyPipelineSelection, entry.stepId) : emptyPipelineSelection,
    })),
    back: history.index > 0
      ? () => setHistory((state) => ({ ...state, index: Math.max(0, state.index - 1) }))
      : undefined,
    forward: history.index < history.entries.length - 1
      ? () => setHistory((state) => ({ ...state, index: Math.min(state.entries.length - 1, state.index + 1) }))
      : undefined,
  };
}
