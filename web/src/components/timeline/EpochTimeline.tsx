import clsx from 'clsx';
import type { ReactNode } from 'react';
import { studioName, studioUnit } from '@/components/activity/studioVocabulary';

/**
 * An epoch as a definition fixes it, in seconds: its least, nominal and greatest length. A drawn length spans least
 * to greatest around its nominal; a condition ends it somewhere between its least and its limit; Rest is what remains
 * of the trial.
 */
export interface TimelineEpoch {
  readonly id: string;
  readonly name: string;
  readonly ends: 'length' | 'drawn' | 'condition' | 'rest';
  readonly least: number;
  readonly nominal: number;
  readonly greatest: number;
}

/** Adjacent epochs repeated `count` times, drawn once. */
export interface TimelineRepeat {
  readonly first: string;
  readonly last: string;
  readonly count: number;
}

/** An epoch's interval in one trial, in seconds; a repeated epoch has one per repetition. */
export interface TimelineInterval {
  readonly id: string;
  readonly name: string;
  readonly start: number;
  readonly end: number;
}

/** An epoch over a batch of trials: per step, the share of trials that are in it. */
export interface TimelineShare {
  readonly id: string;
  readonly name: string;
  readonly share: readonly number[];
}

export type TimelineContent =
  | { readonly kind: 'definition'; readonly epochs: readonly TimelineEpoch[]; readonly repeats: readonly TimelineRepeat[] }
  | { readonly kind: 'trial'; readonly intervals: readonly TimelineInterval[] }
  | { readonly kind: 'batch'; readonly epochs: readonly TimelineShare[] };

export interface EpochTimelineProps {
  readonly content: TimelineContent;
  /** The trial's length, in seconds. */
  readonly duration: number;
  /** The model's step, in seconds: ticks also carry step counts, and a batch's shares are per step. */
  readonly stepSeconds?: number;
  /** The epoch drawn as selected. */
  readonly selected?: string;
  readonly onSelect?: (epoch: string) => void;
  /** Controls at the strip's trailing end, such as a trial stepper. */
  readonly trailing?: ReactNode;
  readonly className?: string;
}

const percent = (value: number, total: number) => `${(100 * value / total).toFixed(3)}%`;

/** A tick spacing of 1, 2 or 5 times a power of ten giving about `count` ticks over `span`. */
function tickStep(span: number, count = 6) {
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((factor) => factor * power).find((step) => step >= raw) ?? raw;
}

const seconds = (value: number, step: number) => value.toFixed(Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));

/**
 * A definition laid out once along its axis: each epoch at its nominal length, a condition's out to its limit, a repeat
 * once. `real` gives the trial time a layout position stands for, which a repeat drawn once makes differ.
 */
function definitionLayout(epochs: readonly TimelineEpoch[], repeats: readonly TimelineRepeat[]) {
  const widths = epochs.map((epoch) => epoch.ends === 'condition' ? epoch.greatest : epoch.nominal);
  const starts = widths.map((_, index) => widths.slice(0, index).reduce((sum, width) => sum + width, 0));
  const total = widths.reduce((sum, width) => sum + width, 0);
  const spans = repeats.flatMap((repeat) => {
    const first = epochs.findIndex(({ id }) => id === repeat.first);
    const last = epochs.findIndex(({ id }) => id === repeat.last);
    if (first < 0 || last < first) return [];
    const start = starts[first];
    const end = starts[last] + widths[last];
    const nominal = epochs.slice(first, last + 1).reduce((sum, epoch) => sum + epoch.nominal, 0);
    return [{ ...repeat, start, end, extra: (repeat.count - 1) * nominal }];
  });
  const real = (position: number) => position + spans.filter(({ end }) => end <= position).reduce((sum, { extra }) => sum + extra, 0);
  return { widths, starts, total, spans, real };
}

function Ticks({ total, real, stepSeconds }: { total: number; real: (position: number) => number; stepSeconds?: number }) {
  const step = tickStep(total);
  const ticks = Array.from({ length: Math.floor(total / step + 1e-9) + 1 }, (_, index) => index * step);
  return <div aria-hidden="true" className="relative h-4 text-xs tabular-nums text-shell-muted">
    {ticks.map((tick) => {
      const time = real(tick);
      return <span key={tick} className="absolute top-0 flex -translate-x-1/2 flex-col items-center whitespace-nowrap leading-none"
        style={{ left: percent(tick, total) }}>
        <span className="h-1 w-px bg-shell-divider" />
        <span>{seconds(time, step)}{tick === 0 ? ` ${studioUnit('second')}` : ''}
          {stepSeconds && tick > 0 ? <span className="opacity-70"> · {Math.round(time / stepSeconds)}</span> : null}
        </span>
      </span>;
    })}
  </div>;
}

const segment = 'absolute top-0 flex h-5 min-w-0 items-center overflow-hidden rounded-sm border px-1 text-xs';

function EpochSegment({ id, name, left, width, total, selected, onSelect, dashedFrom, children }: {
  id: string; name: string; left: number; width: number; total: number; selected?: string; onSelect?: (epoch: string) => void;
  /** Where a condition's dashed remainder starts, as a share of the segment. */
  dashedFrom?: number;
  children?: ReactNode;
}) {
  const chosen = selected === id;
  const style = { left: percent(left, total), width: percent(width, total) };
  const look = clsx(segment, chosen ? 'border-shell-foreground' : 'border-shell-control',
    dashedFrom === undefined && 'bg-shell-field');
  const body = <>
    {dashedFrom !== undefined && <>
      <span aria-hidden="true" className="absolute inset-y-0 left-0 bg-shell-field" style={{ width: percent(dashedFrom, 1) }} />
      <span aria-hidden="true" className="absolute inset-y-[-1px] right-[-1px] rounded-r-sm border border-dashed border-shell-muted"
        style={{ left: percent(dashedFrom, 1) }} />
    </>}
    <span className="relative truncate">{name}</span>
    {children}
  </>;
  return onSelect
    ? <span role="button" tabIndex={-1} aria-pressed={chosen} aria-label={name} className={clsx(look, 'cursor-default')} style={style}
      onPointerDown={() => onSelect(id)}>{body}</span>
    : <span className={look} style={style}>{body}</span>;
}

function Definition({ epochs, repeats, stepSeconds, selected, onSelect }: {
  epochs: readonly TimelineEpoch[]; repeats: readonly TimelineRepeat[]; stepSeconds?: number;
  selected?: string; onSelect?: (epoch: string) => void;
}) {
  const { widths, starts, total, spans, real } = definitionLayout(epochs, repeats);
  return <>
    <div className="relative h-3" aria-hidden={!spans.length}>
      {spans.map((span) => <span key={span.first} role="img" aria-label={`${studioName('repeat')} × ${span.count}`}
        className="absolute bottom-0.5 h-1.5 border-x border-t border-shell-muted"
        style={{ left: percent(span.start, total), width: percent(span.end - span.start, total) }}>
        <span className="absolute -top-2 right-0 translate-x-full pl-1 text-xs leading-none text-shell-muted">× {span.count}</span>
      </span>)}
    </div>
    <div className="relative h-7">
      {epochs.map((epoch, index) => {
        const width = widths[index];
        const drawn = epoch.ends === 'drawn' && epoch.greatest > epoch.least;
        return <span key={epoch.id}>
          <EpochSegment id={epoch.id} name={epoch.name} left={starts[index]} width={width} total={total}
            selected={selected} onSelect={onSelect}
            dashedFrom={epoch.ends === 'condition' ? epoch.least / width : undefined} />
          {drawn && <span role="img" aria-label={`${epoch.name} ${epoch.least}–${epoch.greatest} ${studioUnit('second')}`}
            className="absolute top-[1.375rem] h-1.5 border-x border-shell-foreground"
            style={{
              left: percent(starts[index] + epoch.least, total),
              width: percent(epoch.greatest - epoch.least, total),
            }}>
            <span className="absolute inset-x-0 top-1/2 h-px bg-shell-foreground" />
          </span>}
        </span>;
      })}
    </div>
    <Ticks total={total} real={real} stepSeconds={stepSeconds} />
  </>;
}

function Trial({ intervals, duration, stepSeconds, selected, onSelect }: {
  intervals: readonly TimelineInterval[]; duration: number; stepSeconds?: number; selected?: string; onSelect?: (epoch: string) => void;
}) {
  return <>
    <div className="h-3" />
    <div className="relative h-7">
      {intervals.map((interval, index) => <EpochSegment key={`${interval.id}:${index}`} id={interval.id} name={interval.name}
        left={interval.start} width={interval.end - interval.start} total={duration} selected={selected} onSelect={onSelect} />)}
    </div>
    <Ticks total={duration} real={(time) => time} stepSeconds={stepSeconds} />
  </>;
}

/** Runs of steps that every trial is in (`all`) or some are in (`some`). */
function shareRuns(share: readonly number[]) {
  const runs: { start: number; end: number; level: 'all' | 'some' }[] = [];
  share.forEach((value, step) => {
    const level = value >= 1 ? 'all' : value > 0 ? 'some' : undefined;
    const last = runs.at(-1);
    if (!level) return;
    if (last && last.level === level && last.end === step) last.end = step + 1;
    else runs.push({ start: step, end: step + 1, level });
  });
  return runs;
}

/** Epochs of different trials overlap in time, so a batch draws one row per epoch: heavy where every trial is in it, light where some are. */
function Batch({ epochs, duration, stepSeconds, selected, onSelect }: {
  epochs: readonly TimelineShare[]; duration: number; stepSeconds: number; selected?: string; onSelect?: (epoch: string) => void;
}) {
  const steps = Math.round(duration / stepSeconds);
  return <>
    <div className="grid gap-px pt-1">
      {epochs.map((epoch) => <div key={epoch.id} role={onSelect ? 'button' : undefined} tabIndex={onSelect ? -1 : undefined}
        aria-label={epoch.name} aria-pressed={onSelect ? selected === epoch.id : undefined}
        className="grid grid-cols-[4.5rem_minmax(0,1fr)] items-center gap-1 text-xs" onPointerDown={() => onSelect?.(epoch.id)}>
        <span className={clsx('truncate', selected === epoch.id ? 'font-medium text-shell-foreground' : 'text-shell-muted')}>{epoch.name}</span>
        <span className="relative h-3">
          {shareRuns(epoch.share).map((run) => <span key={run.start} data-share={run.level}
            className={clsx('absolute bg-shell-foreground', run.level === 'all' ? 'inset-y-0 opacity-60' : 'inset-y-[3px] opacity-20')}
            style={{ left: percent(run.start, steps), width: percent(run.end - run.start, steps) }} />)}
        </span>
      </div>)}
    </div>
    <div className="grid grid-cols-[4.5rem_minmax(0,1fr)] gap-1">
      <span />
      <Ticks total={duration} real={(time) => time} stepSeconds={stepSeconds} />
    </div>
  </>;
}

/**
 * A trial's epochs on a time axis. A definition draws what it fixes exactly and marks what it cannot: a whisker
 * where a length is drawn, a dashed remainder where an epoch ends on a condition, a repeat once with its count. A
 * sampled trial draws its own boundaries; a batch, one row per epoch. Nothing on it is dragged.
 */
export function EpochTimeline({ content, duration, stepSeconds, selected, onSelect, trailing, className }: EpochTimelineProps) {
  return <div role="group" aria-label={studioName('timeline')} className={clsx('flex min-w-0 items-start gap-3', className)}>
    <div className="min-w-0 grow">
      {content.kind === 'definition'
        ? <Definition epochs={content.epochs} repeats={content.repeats} stepSeconds={stepSeconds} selected={selected} onSelect={onSelect} />
        : content.kind === 'trial'
          ? <Trial intervals={content.intervals} duration={duration} stepSeconds={stepSeconds} selected={selected} onSelect={onSelect} />
          : <Batch epochs={content.epochs} duration={duration} stepSeconds={stepSeconds ?? duration / content.epochs[0].share.length}
            selected={selected} onSelect={onSelect} />}
    </div>
    {trailing && <div className="flex shrink-0 items-center pt-3">{trailing}</div>}
  </div>;
}
