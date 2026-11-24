import type { ReactNode } from 'react';
import { studioAccessibleName, studioName, studioUnit } from '@/components/activity/studioVocabulary';
import { epochBoundaries, type Percentiles, type SignalAggregate, type SignalSteps, type TrialShown } from './trialSample';

const width = 240;
const height = 64;
const components = ['x', 'y', 'z'];

/** Each component's line style: the first solid, the next dashed, so identity is not in colour alone. */
const dashes = [undefined, '3 2', '1 2'];

function range(values: Iterable<number>) {
  let low = Infinity;
  let high = -Infinity;
  for (const value of values) { low = Math.min(low, value); high = Math.max(high, value); }
  if (!Number.isFinite(low)) return { low: 0, high: 1 };
  if (high - low < 1e-9) return { low: low - 1, high: high + 1 };
  const pad = (high - low) * 0.08;
  return { low: low - pad, high: high + pad };
}

/** A path through the present steps, broken where the signal is absent. */
function path(values: readonly (number | null)[], y: (value: number) => number) {
  const x = (step: number) => (step + 0.5) * width / values.length;
  return values.map((value, step) => value === null ? '' : `${step && values[step - 1] !== null ? 'L' : 'M'}${x(step).toFixed(2)},${y(value).toFixed(2)}`).join(' ');
}

function band(lower: readonly (number | null)[], upper: readonly (number | null)[], y: (value: number) => number) {
  const x = (step: number) => (step + 0.5) * width / lower.length;
  const pieces: string[] = [];
  let run: number[] = [];
  const flush = () => {
    if (run.length) {
      pieces.push(`M${run.map((step) => `${x(step).toFixed(2)},${y(upper[step]!).toFixed(2)}`).join(' L')} L${
        [...run].reverse().map((step) => `${x(step).toFixed(2)},${y(lower[step]!).toFixed(2)}`).join(' L')} Z`);
    }
    run = [];
  };
  lower.forEach((value, step) => { if (value === null) flush(); else run.push(step); });
  flush();
  return pieces.join(' ');
}

/** Presence over the trial: present steps filled, absent ones empty. Over a batch: heavy where every trial has it, light where some do. */
function Presence({ share }: { share: readonly number[] }) {
  return <svg aria-label={studioName('presence')} role="img" viewBox={`0 0 ${share.length} 1`} preserveAspectRatio="none"
    className="h-1.5 w-full text-shell-foreground">
    {share.map((value, step) => value > 0 && <rect key={step} x={step} y={value >= 1 ? 0 : 0.25} width={1.02} height={value >= 1 ? 1 : 0.5}
      fill="currentColor" fillOpacity={value >= 1 ? 0.6 : 0.2} data-share={value >= 1 ? 'all' : 'some'} />)}
  </svg>;
}

function Frame({ label, low, high, boundaries, steps, children }: {
  label: string; low: number; high: number; boundaries: readonly number[]; steps: number; children: ReactNode;
}) {
  return <div className="grid grid-cols-[minmax(0,1fr)_2.5rem] gap-x-1 text-xs tabular-nums text-shell-muted">
    <svg role="img" aria-label={studioAccessibleName('plot', label)} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none"
      className="h-16 w-full text-shell-foreground">
      {boundaries.map((step) => <line key={step} x1={step * width / steps} x2={step * width / steps} y1={0} y2={height}
        stroke="currentColor" strokeOpacity={0.15} vectorEffect="non-scaling-stroke" />)}
      {children}
    </svg>
    <span className="grid content-between leading-none"><span>{high.toPrecision(2)}</span><span>{low.toPrecision(2)}</span></span>
  </div>;
}

/** The legend of a signal's components, by line style. */
function Legend({ count }: { count: number }) {
  if (count < 2) return null;
  return <div className="flex gap-3 text-xs text-shell-muted">
    {components.slice(0, count).map((name, index) => <span key={name} className="flex items-center gap-1">
      <svg aria-hidden="true" viewBox="0 0 12 2" className="h-0.5 w-3 text-shell-foreground">
        <line x1={0} x2={12} y1={1} y2={1} stroke="currentColor" strokeWidth={2} strokeDasharray={dashes[index]} />
      </svg>{name}
    </span>)}
  </div>;
}

/** A signal's trace over one sampled trial; absent steps are drawn as absent, not as zero. */
export function TrialTrace({ label, steps, boundaries, duration }: {
  label: string; steps: SignalSteps; boundaries: readonly number[]; duration: number;
}) {
  const count = steps.find((value) => value)?.length ?? 1;
  const { low, high } = range(steps.flatMap((value) => value ?? []));
  const y = (value: number) => height - (value - low) / (high - low) * height;
  return <div className="grid gap-1">
    <Frame label={label} low={low} high={high} boundaries={boundaries} steps={steps.length}>
      {Array.from({ length: count }, (_, component) => <path key={component} d={path(steps.map((value) => value?.[component] ?? null), y)}
        fill="none" stroke="currentColor" strokeWidth={1.5} strokeDasharray={dashes[component]} vectorEffect="non-scaling-stroke" />)}
    </Frame>
    <div className="mr-11 grid gap-1">
      <Presence share={steps.map((value) => value ? 1 : 0)} />
      <Axis duration={duration} />
    </div>
    <Legend count={count} />
  </div>;
}

/** A signal over a batch: each component's median with its 25–75 and 5–95 percentile bands; presence heavy where every trial has it. */
export function BatchTrace({ label, aggregate, boundaries, duration }: {
  label: string; aggregate: SignalAggregate; boundaries: readonly number[]; duration: number;
}) {
  const percentiles = aggregate.percentiles;
  const count = percentiles?.find((value) => value)?.length ?? 0;
  const { low, high } = range(percentiles?.flatMap((step) => step?.flatMap((value) => [value[0], value[4]]) ?? []) ?? []);
  const y = (value: number) => height - (value - low) / (high - low) * height;
  const at = (component: number, quantile: number) => percentiles!.map((step) => step?.[component][quantile] ?? null);
  return <div className="grid gap-1">
    {count > 0 && <Frame label={label} low={low} high={high} boundaries={boundaries} steps={aggregate.presence.length}>
      {Array.from({ length: count }, (_, component) => <g key={component}>
        <path d={band(at(component, 0), at(component, 4), y)} fill="currentColor" fillOpacity={0.08} />
        <path d={band(at(component, 1), at(component, 3), y)} fill="currentColor" fillOpacity={0.16} />
        <path d={path(at(component, 2), y)} fill="none" stroke="currentColor" strokeWidth={1.5}
          strokeDasharray={dashes[component]} vectorEffect="non-scaling-stroke" />
      </g>)}
    </Frame>}
    <div className="mr-11 grid gap-1">
      <Presence share={aggregate.presence} />
      <Axis duration={duration} />
    </div>
    <Legend count={count} />
  </div>;
}

function Axis({ duration }: { duration: number }) {
  return <div className="flex justify-between text-xs tabular-nums text-shell-muted" aria-hidden="true">
    <span>{0}</span><span>{duration} {studioUnit('second')}</span>
  </div>;
}

/**
 * A term's value per step over what is shown, where it was recorded: over one trial, absent where the term does not
 * apply; over all of them, the median with its percentile bands, and where the term applies in every trial or some.
 */
export function TermTrace({ label, term, shown, stepSeconds }: {
  label: string; term: string; shown: TrialShown; stepSeconds: number;
}) {
  switch (shown.kind) {
    case 'one': {
      const recorded = shown.record.terms?.[term];
      const values = recorded?.values;
      if (!recorded || !values) return null;
      const steps = values.map((value, step) => recorded.steps[step] ? [value] : null);
      return <TrialTrace label={label} steps={steps} duration={steps.length * stepSeconds} boundaries={epochBoundaries(shown.record)} />;
    }
    case 'all': {
      const recorded = shown.aggregate.terms?.[term];
      const percentiles = recorded?.percentiles;
      if (!recorded || !percentiles) return null;
      return <BatchTrace label={label} duration={recorded.steps.length * stepSeconds} boundaries={[]} aggregate={{
        presence: recorded.steps, percentiles: percentiles.map((step) => step && [step]) as (Percentiles[] | null)[],
      }} />;
    }
  }
}
