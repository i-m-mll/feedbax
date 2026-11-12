// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { useState } from 'react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import {
  studioAccessibleName,
  studioCountOf,
  studioName,
  studioRunStatus,
} from '@/components/activity/studioVocabulary';
import type { PlanRunView, PlanStepView, PlanView } from '@/features/pipeline/planView';
import { emptyPipelineSelection, type PipelineSelectionState } from '@/features/pipeline/selection';
import { bindStudioCommand } from '@/features/projectWindow/studioCommands';
import { StudioPipelineMap, type StudioPipelineMapProps } from './StudioPipelineMap';
import { fullMapFits, type PipelineCardPresentation } from './pipelineMapLayout';

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const rate = {
  id: 'rate',
  label: 'rate',
  values: [
    { id: 'low', label: '0.001' },
    { id: 'mid', label: '0.003' },
    { id: 'high', label: '0.009' },
  ],
};
const seed = {
  id: 'seed',
  label: 'seed',
  values: [{ id: 'first', label: '1' }, { id: 'second', label: '2' }],
};
const curl = {
  id: 'curl',
  label: 'curl',
  values: [{ id: 'plain', label: '0' }, { id: 'perturbed', label: '1' }],
};

const train: PlanStepView = {
  id: 'train',
  label: 'reach',
  kind: 'train',
  sweeps: [rate, seed],
  runs: rate.values.flatMap((r) => seed.values.map((s): PlanRunView => ({
    id: `${r.id}-${s.id}`,
    label: `${r.label}/${s.label}`,
    values: { rate: r.id, seed: s.id },
    status: 'not_run',
  }))),
  inputs: [],
  statusCounts: { not_run: 6 },
};

function mapWithState(
  plan: PlanView,
  props: Pick<StudioPipelineMapProps, 'mode' | 'drawerOpen' | 'newStepCommand' | 'newSweepCommand'> = {},
) {
  const selectionChanged = vi.fn();
  const presentationChanged = vi.fn();
  function View() {
    const [state, setState] = useState(emptyPipelineSelection);
    const [presentation, setPresentation] = useState<Record<string, PipelineCardPresentation>>({});
    return (
      <StudioPipelineMap
        {...props}
        plan={plan}
        state={state}
        onSelectionChange={(next) => {
          setState(next);
          selectionChanged(next);
        }}
        presentation={presentation}
        onPresentationChange={(stepId, next) => {
          setPresentation((prior) => ({ ...prior, [stepId]: next }));
          presentationChanged(stepId, next);
        }}
      />
    );
  }
  return { ...render(<View />), selectionChanged, presentationChanged };
}

const select = (label: string) => studioAccessibleName('select', label);
const mark = (coordinates: string) => select(`${coordinates} · ${studioRunStatus('not_run')}`);

it('keeps the step, sweep-value, and individual-run handles distinct', () => {
  const onNewStep = vi.fn();
  const onNewSweep = vi.fn();
  const { selectionChanged } = mapWithState({ steps: [train], edges: [] }, {
    newStepCommand: (source, subject) => bindStudioCommand('newStep', () => onNewStep(source), { subject }),
    newSweepCommand: (id, subject) => bindStudioCommand('newSweep', () => onNewSweep(id), { subject }),
  });
  const card = within(screen.getByRole('article'));
  const header = card.getByRole('button', { name: select(train.label) });
  const first = card.getByRole('button', { name: mark('rate 0.003 · seed 1') });
  const second = card.getByRole('button', { name: mark('rate 0.003 · seed 2') });

  fireEvent.click(header);
  expect(header.getAttribute('aria-pressed')).toBe('true');
  expect(first.getAttribute('aria-pressed')).toBe('true');

  fireEvent.click(card.getByRole('button', { name: select('rate 0.003') }));
  expect(header.getAttribute('aria-pressed')).toBe('false');
  expect(card.getByText('rate 0.003')).toBeTruthy();
  expect(card.getByText(studioCountOf(2, 6))).toBeTruthy();
  expect(first.getAttribute('aria-pressed')).toBe('true');
  expect(second.getAttribute('aria-pressed')).toBe('true');

  fireEvent.click(card.getByRole('button', { name: select('seed 1') }), { metaKey: true });
  expect(first.getAttribute('aria-pressed')).toBe('true');
  expect(second.getAttribute('aria-pressed')).toBe('false');

  fireEvent.click(first);
  fireEvent.click(second, { metaKey: true });
  const scope = { kind: 'rows', stepId: train.id, runIds: ['mid-first', 'mid-second'] };
  expect(selectionChanged).toHaveBeenLastCalledWith({
    selection: scope,
    focus: { stepId: train.id, runId: 'mid-second' },
  });
  expect(card.getByRole('button', { name: select('rate 0.003') }).getAttribute('aria-pressed')).toBe('false');

  fireEvent.click(card.getByRole('button', { name: studioAccessibleName('newStep', train.label) }));
  expect(onNewStep).toHaveBeenCalledWith({ stepId: train.id, selection: scope });
  fireEvent.click(card.getByRole('button', { name: studioAccessibleName('newSweep', train.label) }));
  expect(onNewSweep).toHaveBeenCalledWith(train.id);
});

it('shows partial folded marks and preserves group selection when sweeps move', () => {
  const step: PlanStepView = {
    ...train,
    sweeps: [...train.sweeps, curl],
    runs: train.runs.flatMap((run) => curl.values.map((value) => ({
      ...run,
      id: `${run.id}-${value.id}`,
      values: { ...run.values, curl: value.id },
    }))),
    statusCounts: { not_run: 12 },
  };
  const { selectionChanged, presentationChanged } = mapWithState({ steps: [step], edges: [] });
  const card = within(screen.getByRole('article'));
  fireEvent.click(card.getByLabelText(studioAccessibleName('sweep', curl.label)));
  fireEvent.click(card.getByRole('button', { name: studioAccessibleName('fold', curl.label) }));
  fireEvent.click(card.getByRole('button', { name: select('curl 1') }));
  const aggregate = (coordinates: string) => select(
    `${coordinates} · 2 ${studioName('runs')} · 2 ${studioRunStatus('not_run')} · ${studioCountOf(1, 2)}`,
  );
  const foldedMark = card.getByRole('button', { name: aggregate('rate 0.003 · seed 1') });
  expect(foldedMark.getAttribute('aria-pressed')).toBe('mixed');

  const dataTransfer = { effectAllowed: 'none', setData: vi.fn() };
  fireEvent.dragStart(card.getByLabelText(studioAccessibleName('sweep', seed.label)), { dataTransfer });
  fireEvent.drop(card.getByLabelText(studioAccessibleName('sweep', rate.label)), { dataTransfer });
  expect(presentationChanged).toHaveBeenLastCalledWith(train.id, {
    foldedSweepId: curl.id,
    sweepOrder: [seed.id, rate.id, curl.id],
  });
  const movedMark = card.getByRole('button', { name: aggregate('seed 1 · rate 0.003') });
  fireEvent.click(movedMark);
  expect(selectionChanged).toHaveBeenLastCalledWith({
    selection: {
      kind: 'values',
      stepId: train.id,
      terms: [
        { sweepId: seed.id, valueIds: ['first'] },
        { sweepId: rate.id, valueIds: ['mid'] },
      ],
    },
    focus: null,
  });
});

it('automatically folds a large sweep into a group with explicit value IDs', () => {
  const values = Array.from({ length: 1_100 }, (_, index) => ({ id: `seed-${index}`, label: String(index) }));
  const step: PlanStepView = {
    ...train,
    sweeps: [{ ...seed, values }],
    runs: values.map(({ id, label }) => ({ id, label, values: { seed: id }, status: 'not_run' })),
    statusCounts: { not_run: values.length },
  };
  const { selectionChanged } = mapWithState({ steps: [step], edges: [] });
  const card = within(screen.getByRole('article'));
  expect(card.getByLabelText(studioAccessibleName('sweep', seed.label)).textContent).toBe('(seed 1100)');
  const aggregate = select(`1100 ${studioName('runs')} · 1100 ${studioRunStatus('not_run')}`);
  fireEvent.click(card.getByRole('button', { name: aggregate }));
  expect(selectionChanged).toHaveBeenLastCalledWith({
    selection: {
      kind: 'values',
      stepId: step.id,
      terms: [{ sweepId: seed.id, valueIds: values.map(({ id }) => id) }],
    },
    focus: null,
  });
});

it('draws the input subset and follows only linked runs into the drawer and direct step neighbors', () => {
  const evalStep: PlanStepView = {
    id: 'eval',
    label: 'perturbations',
    kind: 'evaluate',
    sweeps: [],
    runs: [{ id: 'eval-1', label: 'perturbation', values: {}, status: 'not_run' }],
    inputs: [{
      id: 'model',
      label: 'Trained model',
      sourceStepId: train.id,
      subset: { terms: ['rate 0.003'], count: 2, total: 6 },
    }],
    statusCounts: { not_run: 1 },
  };
  const report: PlanStepView = {
    ...evalStep,
    id: 'report',
    label: 'summary',
    kind: 'report',
    runs: [{ id: 'report-1', label: 'summary', values: {}, status: 'not_run' }],
    inputs: [{ id: 'results', label: 'Results', sourceStepId: evalStep.id }],
  };
  const unrelated: PlanStepView = { ...train, id: 'unrelated', label: 'baseline' };
  const plan = {
    steps: [report, unrelated, evalStep, train],
    edges: [
      { from: { stepId: train.id, runId: 'mid-first' }, to: { stepId: evalStep.id, runId: 'eval-1' } },
      { from: { stepId: evalStep.id, runId: 'eval-1' }, to: { stepId: report.id, runId: 'report-1' } },
    ],
  };
  const view = mapWithState(plan, { drawerOpen: true });
  const card = within(screen.getByRole('article', { name: studioAccessibleName('train', train.label) }));
  const evaluation = within(screen.getByRole('article', { name: studioAccessibleName('evaluate', evalStep.label) }));
  expect(evaluation.getByText('rate 0.003')).toBeTruthy();
  expect(evaluation.getByText(studioCountOf(2, 6))).toBeTruthy();
  expect(view.container.querySelectorAll('path[stroke="currentColor"]')).toHaveLength(2);
  const drawer = within(screen.getByRole('navigation', { name: studioName('steps') }));
  const evalRow = within(drawer.getByRole('button', { name: select(evalStep.label) }));
  const reportRow = within(drawer.getByRole('button', { name: select(report.label) }));
  fireEvent.click(card.getByRole('button', { name: mark('rate 0.003 · seed 1') }));
  expect(evaluation.getByRole('button', { name: mark('perturbation') }).getAttribute('data-map-focus')).toBe('downstream');
  expect(view.container.querySelectorAll('[data-map-emphasis="strong"]')).toHaveLength(2);
  expect(evalRow.getByText(studioCountOf(1, 1))).toBeTruthy();
  expect(reportRow.getByText(studioCountOf(1, 1))).toBeTruthy();
  fireEvent.click(card.getByRole('button', { name: mark('rate 0.001 · seed 1') }));
  expect(view.container.querySelectorAll('[data-map-focus], [data-map-emphasis="strong"]')).toHaveLength(0);
  expect(evalRow.getByText(studioCountOf(0, 1))).toBeTruthy();
  expect(reportRow.getByText(studioCountOf(0, 1))).toBeTruthy();
  view.unmount();

  const onSelectionChange = vi.fn();
  const state: PipelineSelectionState = { selection: { kind: 'steps', stepIds: [evalStep.id] }, focus: null };
  render(<StudioPipelineMap mode="line" plan={plan} state={state} onSelectionChange={onSelectionChange} />);
  const buttons = screen.getAllByRole('button');
  expect(buttons.map((button) => button.getAttribute('aria-label'))).toEqual([
    select(train.label), select(evalStep.label), select(report.label),
  ]);
  fireEvent.click(buttons[2]);
  expect(onSelectionChange).toHaveBeenCalledWith({
    selection: { kind: 'steps', stepIds: [report.id] }, focus: null,
  });
});

it('uses the region at natural scale to decide when full cards fit', () => {
  const columns = Array.from({ length: 3 }, (_, depth) =>
    Array.from({ length: 12 }, (_, row) => ({ ...train, id: `${depth}-${row}` })));
  expect(fullMapFits(columns, 1440, 900, new Map())).toBe(false);
  expect(fullMapFits(columns.map((column) => column.slice(0, 2)), 1440, 900, new Map())).toBe(true);
  expect(fullMapFits(columns.map((column) => column.slice(0, 2)), 580, 900, new Map())).toBe(false);
});

it('uses one creation position on an empty map', () => {
  const onNewStep = vi.fn();
  mapWithState({ steps: [], edges: [] }, { drawerOpen: true, newStepCommand: () => bindStudioCommand('newStep', onNewStep) });
  const [create] = screen.getAllByRole('button');
  expect(screen.queryByRole('navigation')).toBeNull();
  fireEvent.click(create);
  expect(onNewStep).toHaveBeenCalledWith();
});
