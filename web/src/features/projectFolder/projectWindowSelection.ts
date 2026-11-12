import { useState } from 'react';
import type { PlanView } from '@/features/pipeline/planView';
import {
  emptyPipelineSelection, selectStep, selectSweepGroup, type PipelineSelectionState,
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

export function settledWindowSelection(plan: PlanView, state: PipelineSelectionState): PipelineSelectionState {
  const selection = state.selection;
  if (!selection || selection.kind === 'steps') return state;
  const step = plan.steps.find(({ id }) => id === selection.stepId);
  if (!step) return state;
  if (selection.kind === 'rows') {
    const available = new Set(step.runs.map(({ id }) => id));
    const runIds = selection.runIds.filter((id) => available.has(id));
    if (runIds.length === selection.runIds.length) return state;
    if (!runIds.length) return selectStep(state, step.id);
    const focus = state.focus && runIds.includes(state.focus.runId) ? state.focus
      : { stepId: step.id, runId: runIds[runIds.length - 1] };
    return { selection: { ...selection, runIds }, focus };
  }
  const terms = selection.terms.flatMap((term) => {
    const sweep = step.sweeps.find(({ id }) => id === term.sweepId);
    const available = new Set(sweep?.values.map(({ id }) => id));
    const valueIds = term.valueIds.filter((id) => available.has(id));
    if (valueIds.length === term.valueIds.length) return [term];
    return valueIds.length ? [{ ...term, valueIds }] : [];
  });
  if (terms.length === selection.terms.length && terms.every((term, index) => term === selection.terms[index])) return state;
  return terms.length ? selectSweepGroup(step.id, terms) : selectStep(state, step.id);
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
