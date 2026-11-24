import { useState } from 'react';
import type { PlanView } from '@/features/pipeline/planView';
import { renamedSavedRunId } from './projectWindowPlan';
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

function renamedSelection({ selection, focus }: PipelineSelectionState, from: string, to: string): PipelineSelectionState {
  const step = (id: string) => id === from ? to : id;
  const run = (stepId: string, id: string) => stepId === from ? renamedSavedRunId(id, to) : id;
  return {
    selection: !selection ? null
      : selection.kind === 'steps' ? { ...selection, stepIds: selection.stepIds.map(step) }
        : selection.kind === 'rows'
          ? { ...selection, stepId: step(selection.stepId), runIds: selection.runIds.map((id) => run(selection.stepId, id)) }
          : { ...selection, stepId: step(selection.stepId) },
    focus: focus && { stepId: step(focus.stepId), runId: run(focus.stepId, focus.runId) },
  };
}

export function settledWindowSelection(plan: PlanView, state: PipelineSelectionState): PipelineSelectionState {
  const selection = state.selection;
  if (!selection) return state;
  if (selection.kind === 'steps') {
    const stepIds = selection.stepIds.filter((id) => plan.steps.some((step) => step.id === id));
    if (stepIds.length === selection.stepIds.length) return state;
    return stepIds.length ? { selection: { ...selection, stepIds }, focus: null } : emptyPipelineSelection;
  }
  const step = plan.steps.find(({ id }) => id === selection.stepId);
  if (!step) return emptyPipelineSelection;
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

/**
 * The window's selection and its history. A change of selection goes through `inTurn`, which holds it behind the
 * gestures still waiting to run, so that each acts on the selection it was made on.
 */
export function useProjectWindowSelection(initialStepId: string | null, inTurn: (change: () => void) => void) {
  const [history, setHistory] = useState<{ entries: readonly SelectionEntry[]; index: number }>(() => ({
    entries: [{
      stepId: initialStepId,
      variantId: null,
      pipeline: initialStepId ? selectStep(emptyPipelineSelection, initialStepId) : emptyPipelineSelection,
    }],
    index: 0,
  }));
  const current = history.entries[history.index];

  const change = (update: (entry: SelectionEntry) => SelectionEntry) => inTurn(() => setHistory(({ entries, index }) => {
    const previous = entries[index];
    const next = update(previous);
    if (JSON.stringify(previous) === JSON.stringify(next)) return { entries, index };
    return { entries: [...entries.slice(0, index + 1), next], index: index + 1 };
  }));

  return {
    ...current,
    /** Whether anything has been selected since the window opened. */
    edited: history.entries.length > 1,
    select: (pipeline: PipelineSelectionState, variantId?: string | null) => change((entry) => ({
      variantId: variantId === undefined ? entry.variantId : variantId,
      pipeline, stepId: subject(pipeline, entry.stepId),
    })),
    /** Follow a step to its new name through the whole history. */
    renameStep: (from: string, to: string) => setHistory(({ entries, index }) => ({
      index,
      entries: entries.map((entry) => ({
        ...entry,
        stepId: entry.stepId === from ? to : entry.stepId,
        pipeline: renamedSelection(entry.pipeline, from, to),
      })),
    })),
    setVariant: (variantId: string) => change((entry) => ({
      ...entry,
      variantId,
      pipeline: entry.stepId ? selectStep(emptyPipelineSelection, entry.stepId) : emptyPipelineSelection,
    })),
    back: history.index > 0
      ? () => inTurn(() => setHistory((state) => ({ ...state, index: Math.max(0, state.index - 1) })))
      : undefined,
    forward: history.index < history.entries.length - 1
      ? () => inTurn(() => setHistory((state) => ({ ...state, index: Math.min(state.entries.length - 1, state.index + 1) })))
      : undefined,
  };
}
