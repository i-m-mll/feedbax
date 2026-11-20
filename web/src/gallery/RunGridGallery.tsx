import { useMemo, useState } from 'react';
import { Card, CardHead } from '@/components/ui/Card';
import { PipelineRunMarks } from '@/components/pipeline/PipelineRunMarks';
import { mapCardWidth, orderedMapSweeps } from '@/components/pipeline/pipelineMapLayout';
import type { PlanRunView, PlanStepView, PlanSweepView } from '@/features/pipeline/planView';
import { emptyPipelineSelection, selectedRuns } from '@/features/pipeline/selection';

type Axis = [label: string, values: number | readonly string[]];

function sweep([label, values]: Axis): PlanSweepView {
  const labels = typeof values === 'number' ? Array.from({ length: values }, (_, index) => String(index + 1)) : values;
  return { id: label, label, values: labels.map((value, index) => ({ id: `${label}-${index}`, label: value })) };
}

const states = ['succeeded', 'succeeded', 'succeeded', 'running', 'failed', 'not_run', 'stale'] as const;

function gridStep(id: string, axes: readonly Axis[]): PlanStepView {
  const sweeps = axes.map(sweep);
  const combinations = sweeps.reduce<Record<string, string>[]>((rows, { id: sweepId, values }) =>
    rows.flatMap((row) => values.map((value) => ({ ...row, [sweepId]: value.id }))), [{}]);
  const runs = combinations.map((values, index): PlanRunView => {
    const state = states[(index * 7 + Math.floor(index / 5)) % states.length];
    return { id: `${id}-${index}`, label: `run ${index + 1}`, values, status: state === 'stale' ? 'succeeded' : state, state, results: [] };
  });
  const statusCounts: Record<string, number> = {};
  for (const run of runs) statusCounts[run.state] = (statusCounts[run.state] ?? 0) + 1;
  return { id, label: id, kind: 'train', sweeps, runs, inputs: [], statusCounts };
}

const cases: [name: string, step: PlanStepView][] = [
  ['uneven values', gridStep('uneven', [['learning rate', ['1', '2.0000003', '3']]])],
  ['5 values', gridStep('five', [['learning rate', 5]])],
  ['11 values', gridStep('eleven', [['learning rate', 11]])],
  ['3 × 11', gridStep('two', [['optimizer', ['adam', 'sgd', 'lion']], ['learning rate', 11]])],
  ['3 × 4 × 10', gridStep('three', [['optimizer', ['adam', 'sgd', 'lion']], ['learning rate', 4], ['seed', 10]])],
  ['5 × 20', gridStep('small', [['optimizer', 5], ['learning rate', 20]])],
  ['5 × 40', gridStep('forty', [['optimizer', 5], ['learning rate', 40]])],
  ['60 × 60', gridStep('sixty', [['optimizer', 60], ['learning rate', 60]])],
  ['long names', gridStep('long', [['optimizer-b1', 4], ['learning-rate-peak-value', 11], ['seed', 10]])],
];

function RunGridCard({ step }: { step: PlanStepView }) {
  const [state, setState] = useState(emptyPipelineSelection);
  const [order, setOrder] = useState<readonly string[]>();
  const sweeps = useMemo(() => orderedMapSweeps(step, { sweepOrder: order }), [step, order]);
  const selected = useMemo(() => new Set(selectedRuns(step, state.selection).map(({ id }) => id)), [step, state]);
  return (
    <Card className="text-sm" style={{ width: mapCardWidth }}>
      <CardHead className="px-3 py-2 font-medium">{step.label}</CardHead>
      <div className="px-3 pb-2">
        <PipelineRunMarks step={step} sweeps={sweeps} selection={state.selection} selected={selected}
          applySelection={(change) => setState(change)} arrange={setOrder} />
      </div>
    </Card>
  );
}

/** The map card's run grid at each count its design names. */
export function RunGridGallery() {
  return (
    <div className="flex flex-wrap items-start gap-3">
      {cases.map(([name, step]) => (
        <div key={step.id} className="grid justify-items-start gap-1">
          <RunGridCard step={step} />
          <span className="text-xs text-shell-muted">{name}</span>
        </div>
      ))}
    </div>
  );
}
