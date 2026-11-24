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
import { StudioPipelineMap, StudioPipelineMapDrawer, type StudioPipelineMapProps } from './StudioPipelineMap';
import {
  fullMapFits, layoutPipelineMap, mapCardWidth, mapMarkSize, placePipelineMap, type PipelineCardPresentation,
} from './pipelineMapLayout';

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
    state: 'not_run',
    results: [],
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
  const header = card.getByRole('button', { name: 'Select reach Train' });
  const first = card.getByRole('button', { name: mark('seed 1 · rate 0.003') });
  const second = card.getByRole('button', { name: mark('seed 2 · rate 0.003') });

  fireEvent.click(header);
  expect(header.getAttribute('aria-pressed')).toBe('true');
  expect(first.getAttribute('aria-pressed')).toBe('true');

  chooseValue(card, 'rate', '0.003');
  expect(header.getAttribute('aria-pressed')).toBe('false');
  expect(card.queryByText(studioCountOf(2, 6))).toBeNull();
  expect(first.getAttribute('aria-pressed')).toBe('true');
  expect(second.getAttribute('aria-pressed')).toBe('true');

  chooseValue(card, 'seed', '1', true);
  expect(first.getAttribute('aria-pressed')).toBe('true');
  expect(second.getAttribute('aria-pressed')).toBe('false');
  expect(first.closest('[role="gridcell"]')?.hasAttribute('data-map-group-cell')).toBe(true);
  expect(second.closest('[role="gridcell"]')?.hasAttribute('data-map-group-cell')).toBe(false);

  fireEvent.click(screen.getByRole('article'));
  expect(header.getAttribute('aria-pressed')).toBe('true');

  fireEvent.click(first);
  expect(header.getAttribute('aria-pressed')).toBe('false');
  fireEvent.click(second, { metaKey: true });
  const scope = { kind: 'rows', stepId: train.id, runIds: ['mid-first', 'mid-second'] };
  expect(selectionChanged).toHaveBeenLastCalledWith({
    selection: scope,
    focus: { stepId: train.id, runId: 'mid-second' },
  });

  fireEvent.click(card.getByRole('button', { name: studioAccessibleName('newStep', train.label) }));
  expect(onNewStep).toHaveBeenCalledWith({ stepId: train.id, selection: scope });
  fireEvent.click(card.getByRole('button', { name: studioAccessibleName('newSweep', train.label) }));
  expect(onNewSweep).toHaveBeenCalledWith(train.id);
});

function chooseValue(card: ReturnType<typeof within>, sweep: string, value: string, extend = false) {
  fireEvent.click(card.getByRole('button', { name: new RegExp(`^${studioAccessibleName('sweep', sweep)} `) }));
  fireEvent.click(screen.getByRole('menuitem', { name: value }), { metaKey: extend });
}

it('folds the sweeps after the first two into one mark per cell, named after the axis labels', () => {
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
  const labels = card.getAllByRole('button', { name: /^Sweep / });
  expect(labels.map((label) => label.textContent)).toEqual(['↓\u00a0seed', '→\u00a0rate', '(curl\u00a0×2)']);
  expect(card.getAllByRole('gridcell')).toHaveLength(6);
  chooseValue(card, 'curl', '1');
  const aggregate = (coordinates: string) => select(
    `${coordinates} · 2 ${studioName('runs')} · 2 ${studioRunStatus('not_run')} · ${studioCountOf(1, 2)}`,
  );
  expect(card.getByRole('button', { name: aggregate('seed 1 · rate 0.003') }).getAttribute('aria-pressed')).toBe('mixed');

  const dataTransfer = { effectAllowed: 'none', setData: vi.fn() };
  fireEvent.dragStart(labels[2], { dataTransfer });
  fireEvent.drop(labels[1], { dataTransfer });
  expect(presentationChanged).toHaveBeenLastCalledWith(train.id, { sweepOrder: [seed.id, curl.id, rate.id] });
  expect(card.getAllByRole('button', { name: /^Sweep / }).map((label) => label.textContent))
    .toEqual(['↓\u00a0seed', '→\u00a0curl', '(rate\u00a0×3)']);
  expect(card.getAllByRole('gridcell')).toHaveLength(4);
  fireEvent.click(card.getByRole('button', { name: select(`seed 1 · curl 0 · 3 ${studioName('runs')} · 3 ${studioRunStatus('not_run')}`) }));
  expect(selectionChanged).toHaveBeenLastCalledWith({
    selection: {
      kind: 'values',
      stepId: train.id,
      terms: [
        { sweepId: seed.id, valueIds: ['first'] },
        { sweepId: curl.id, valueIds: ['plain'] },
      ],
    },
    focus: null,
  });
});

it('shrinks marks to the largest of three sizes at which both directions fit, and scrolls only past the smallest', () => {
  expect(mapMarkSize(11, 1).id).toBe('full');
  expect(mapMarkSize(12, 1).id).toBe('small');
  expect(mapMarkSize(22, 11).id).toBe('small');
  expect(mapMarkSize(23, 1).id).toBe('smallest');
  expect(mapMarkSize(45, 43).id).toBe('smallest');
  expect(mapMarkSize(60, 60).id).toBe('smallest');
});

it('draws the input subset and follows only linked runs into the drawer and direct step neighbors', () => {
  const evalStep: PlanStepView = {
    id: 'eval',
    label: 'perturbations',
    kind: 'evaluate',
    sweeps: [],
    runs: [{ id: 'eval-1', label: 'perturbation', values: {}, status: 'not_run', state: 'not_run', results: [] }],
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
    runs: [{ id: 'report-1', label: 'summary', values: {}, status: 'not_run', state: 'not_run', results: [] }],
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
  expect(view.container.querySelectorAll('path[data-map-edge]')).toHaveLength(2);
  const drawer = within(screen.getByRole('navigation', { name: studioName('steps') }));
  const row = (label: string, count: number) => drawer.getByRole('treeitem', { name: new RegExp('^' + label) })
    .getAttribute('aria-description') === studioCountOf(count, 1);
  fireEvent.click(card.getByRole('button', { name: mark('seed 1 · rate 0.003') }));
  expect(evaluation.getByRole('button', { name: mark('perturbation') }).getAttribute('data-map-focus')).toBe('downstream');
  expect(view.container.querySelectorAll('[data-map-emphasis="strong"]')).toHaveLength(2);
  expect(row('perturbations', 1)).toBe(true);
  expect(row('summary', 1)).toBe(true);
  fireEvent.click(card.getByRole('button', { name: mark('seed 1 · rate 0.001') }));
  expect(view.container.querySelectorAll('[data-map-focus], [data-map-emphasis="strong"]')).toHaveLength(0);
  expect(row('perturbations', 0)).toBe(true);
  expect(row('summary', 0)).toBe(true);
  view.unmount();

  const onSelectionChange = vi.fn();
  const state: PipelineSelectionState = { selection: { kind: 'steps', stepIds: [evalStep.id] }, focus: null };
  render(<StudioPipelineMap mode="line" plan={plan} state={state} onSelectionChange={onSelectionChange} />);
  const buttons = screen.getAllByRole('button');
  expect(buttons).toEqual([
    screen.getByRole('button', { name: 'Select reach Train' }),
    screen.getByRole('button', { name: `${select(evalStep.label)} Evaluate` }),
    screen.getByRole('button', { name: `${select(report.label)} Report` }),
  ]);
  fireEvent.click(buttons[2]);
  expect(onSelectionChange).toHaveBeenCalledWith({
    selection: { kind: 'steps', stepIds: [report.id] }, focus: null,
  });
});

it('uses the region at natural scale to decide when full cards fit', () => {
  const columns = Array.from({ length: 3 }, (_, depth) =>
    Array.from({ length: 12 }, (_, row) => ({ ...train, id: `${depth}-${row}` })));
  const placed = (rows: number) => placePipelineMap(
    { columns: columns.map((column) => column.slice(0, rows)), upstream: new Map() },
    () => ({ width: mapCardWidth, height: 96 }),
  );
  expect(fullMapFits(placed(12), 1440, 900)).toBe(false);
  expect(fullMapFits(placed(2), 1440, 900)).toBe(true);
  expect(fullMapFits(placed(2), 580, 900)).toBe(false);
});

it('moves the cards downstream of a changed subject together, and none upstream of it', () => {
  const step = (id: string, kind: PlanStepView['kind'], source?: string): PlanStepView => ({
    id, label: id, kind, sweeps: [], runs: [], statusCounts: {},
    inputs: source ? [{ id: 'subject', label: 'subject', sourceStepId: source }] : [],
  });
  const heights: Record<string, number> = { train: 200, 'train-2': 120, evaluate: 80, report: 60 };
  const place = (subject: string) => {
    const plan = { edges: [], steps: [step('evaluate', 'evaluate', subject), step('report', 'report', 'evaluate'),
      step('train-2', 'train'), step('train', 'train')] };
    return placePipelineMap(layoutPipelineMap(plan), ({ id }) => ({ width: mapCardWidth, height: heights[id] })).cards;
  };
  const before = place('train-2');
  const after = place('train');
  expect(after.get('train')).toEqual(before.get('train'));
  expect(after.get('train-2')).toEqual(before.get('train-2'));
  const offset = after.get('evaluate')!.y - before.get('evaluate')!.y;
  expect(offset).not.toBe(0);
  expect(after.get('report')!.y - before.get('report')!.y).toBe(offset);
  expect(after.get('evaluate')!.y).toBe(after.get('train')!.y);
});

it('ends the drawer of an empty project with its add row', () => {
  const onNewStep = vi.fn();
  render(<StudioPipelineMapDrawer plan={{ steps: [], edges: [] }} state={emptyPipelineSelection} onSelectionChange={vi.fn()}
    newStepCommand={() => bindStudioCommand('newStep', onNewStep)} />);
  const drawer = within(screen.getByRole('navigation', { name: studioName('steps') }));
  fireEvent.keyDown(drawer.getByRole('button', { name: studioName('newStep') }), { key: 'Enter' });
  expect(onNewStep).toHaveBeenCalledWith();
});

it('marks a drawer row with its worst state, and choosing the mark selects those runs', () => {
  const runs = train.runs.map((run, index): PlanRunView => index < 2 ? { ...run, status: 'failed', state: 'failed' } : run);
  const step = { ...train, runs, statusCounts: { failed: 2, not_run: 4 } };
  const onSelectionChange = vi.fn();
  render(<StudioPipelineMapDrawer plan={{ steps: [step], edges: [] }} state={emptyPipelineSelection} onSelectionChange={onSelectionChange} />);
  const drawer = within(screen.getByRole('navigation', { name: studioName('steps') }));
  fireEvent.click(drawer.getByRole('button', { name: `2 ${studioRunStatus('failed')}` }));
  expect(onSelectionChange).toHaveBeenCalledTimes(1);
  expect(onSelectionChange).toHaveBeenCalledWith({
    selection: { kind: 'rows', stepId: train.id, runIds: [runs[0].id, runs[1].id] }, focus: null,
  });
});
