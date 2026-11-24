import clsx from 'clsx';
import { useRef, type PointerEvent, type ReactNode } from 'react';
import { studioName, studioUnit } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { elementKey, type ElementAddress } from '@/features/projectWindow/elementSelection';

/**
 * A named span of a row, in seconds: an epoch of one trial. `spread` is where its end may fall over trials, drawn as a
 * whisker; `dashedFrom` is where a remainder the trial may not reach begins, drawn dashed.
 */
export interface ShelfSegment {
  readonly key: string;
  readonly name: string;
  readonly start: number;
  readonly end: number;
  readonly selects?: ElementAddress;
  readonly spread?: { readonly least: number; readonly greatest: number };
  readonly dashedFrom?: number;
}

/**
 * An unnamed span of a row, in seconds: where a signal is present, or where a term applies. Over a batch of trials,
 * `some` marks a span where only some of the trials have it, drawn light.
 */
export interface ShelfBar {
  readonly start: number;
  readonly end: number;
  readonly some?: boolean;
}

/** A row of the shelf: its name at the left, then its segments or its bars on the shelf's time axis. */
export type ShelfRow = {
  readonly key: string;
  readonly name: string;
  readonly glyph?: ReactNode;
} & (
  | { readonly kind: 'segments'; readonly segments: readonly ShelfSegment[] }
  | { readonly kind: 'bars'; readonly bars: readonly ShelfBar[]; readonly selects?: ElementAddress }
);

export interface ShelfProps {
  readonly rows: readonly ShelfRow[];
  /** The trial's length, in seconds: the axis runs from 0 to it, as the scrubber does. */
  readonly duration: number;
  /** The step, in seconds: ticks also carry step counts. */
  readonly stepSeconds?: number;
  readonly time: number;
  readonly onTimeChange: (time: number) => void;
  readonly selected: readonly ElementAddress[];
  readonly onSelect: (address: ElementAddress) => void;
  /** Controls drawn at the shelf's trailing end. */
  readonly trailing?: ReactNode;
}

const percent = (value: number, total: number) => `${(100 * value / total).toFixed(3)}%`;

/** A tick spacing of 1, 2 or 5 times a power of ten giving about `count` ticks over `span`. */
function tickStep(span: number, count = 6) {
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((factor) => factor * power).find((step) => step >= raw) ?? raw;
}

const seconds = (value: number, step: number) => value.toFixed(Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));

function Ticks({ duration, stepSeconds }: { duration: number; stepSeconds?: number }) {
  const step = tickStep(duration);
  const ticks = Array.from({ length: Math.floor(duration / step + 1e-9) + 1 }, (_, index) => index * step);
  return <div aria-hidden="true" className="relative h-4 text-xs tabular-nums text-shell-muted">
    {ticks.map((tick) => <span key={tick} className="absolute top-0 flex -translate-x-1/2 flex-col items-center whitespace-nowrap leading-none"
      style={{ left: percent(tick, duration) }}>
      <span className="h-1 w-px bg-shell-divider" />
      <span>{seconds(tick, step)}{tick === 0 ? ` ${studioUnit('second')}` : ''}
        {stepSeconds && tick > 0 ? <span className="opacity-70"> · {Math.round(tick / stepSeconds)}</span> : null}
      </span>
    </span>)}
  </div>;
}

const markAttribute = 'data-shelf-mark';

function Segment({ segment, duration, chosen }: { segment: ShelfSegment; duration: number; chosen: boolean }) {
  const { key, name, start, end, spread, dashedFrom, selects } = segment;
  const width = end - start;
  return <>
    <span role={selects ? 'button' : undefined} tabIndex={selects ? -1 : undefined} aria-label={name}
      aria-pressed={selects ? chosen : undefined} {...{ [markAttribute]: key }} data-start={start} data-end={end}
      className={clsx('absolute inset-y-0 flex min-w-0 items-center overflow-hidden rounded-sm border px-1 text-xs',
        chosen ? 'border-shell-foreground' : 'border-shell-control', dashedFrom === undefined && 'bg-shell-field')}
      style={{ left: percent(start, duration), width: percent(width, duration) }}>
      {dashedFrom !== undefined && <>
        <span aria-hidden="true" className="absolute inset-y-0 left-0 bg-shell-field" style={{ width: percent(dashedFrom - start, width) }} />
        <span aria-hidden="true" className="absolute inset-y-[-1px] right-[-1px] rounded-r-sm border border-dashed border-shell-muted"
          style={{ left: percent(dashedFrom - start, width) }} />
      </>}
      <span className="relative truncate">{name}</span>
    </span>
    {spread && spread.greatest > spread.least && <span role="img"
      aria-label={`${name} ${seconds(spread.least, 0.01)}–${seconds(spread.greatest, 0.01)} ${studioUnit('second')}`}
      data-least={spread.least} data-greatest={spread.greatest}
      className="pointer-events-none absolute -bottom-1 h-1.5 border-x border-shell-foreground"
      style={{ left: percent(spread.least, duration), width: percent(spread.greatest - spread.least, duration) }}>
      <span className="absolute inset-x-0 top-1/2 h-px bg-shell-foreground" />
    </span>}
  </>;
}

/**
 * Rows on one time axis under the clock, their ends at the scrubber's ends, for one trial: each a name, then named
 * segments or bars. A row, or a segment, selects what it stands for; dragging along the axis moves the playhead,
 * which runs through every row. The shelf knows what its rows select, not what they are.
 */
export function Shelf({ rows, duration, stepSeconds, time, onTimeChange, selected, onSelect, trailing }: ShelfProps) {
  const chosen = new Set(selected.map(elementKey));
  const isChosen = (address?: ElementAddress) => !!address && chosen.has(elementKey(address));
  const selects = new Map(rows.flatMap((row) => row.kind === 'segments'
    ? row.segments.flatMap(({ key, selects }) => selects ? [[`${row.key}:${key}`, selects] as const] : [])
    : row.selects ? [[row.key, row.selects] as const] : []));
  const drag = useRef<{ mark?: string; x: number; moved: boolean }>();
  const timeAt = (event: PointerEvent<HTMLElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    return Math.min(duration, Math.max(0, (event.clientX - box.left) / box.width * duration));
  };
  const axis = (row: ShelfRow) => ({
    onPointerDown: (event: PointerEvent<HTMLElement>) => {
      if (event.button !== 0) return;
      event.currentTarget.setPointerCapture(event.pointerId);
      const mark = (event.target as Element).closest(`[${markAttribute}]`)?.getAttribute(markAttribute);
      drag.current = { mark: mark ? `${row.key}:${mark}` : row.kind === 'bars' ? row.key : undefined, x: event.clientX, moved: false };
    },
    onPointerMove: (event: PointerEvent<HTMLElement>) => {
      const current = drag.current;
      if (!current || !event.currentTarget.hasPointerCapture(event.pointerId)) return;
      if (!current.moved && Math.abs(event.clientX - current.x) < 3) return;
      current.moved = true;
      onTimeChange(timeAt(event));
    },
    onPointerUp: (event: PointerEvent<HTMLElement>) => {
      const current = drag.current;
      drag.current = undefined;
      if (!current) return;
      if (current.moved) return;
      const address = current.mark ? selects.get(current.mark) : undefined;
      if (address) onSelect(address);
      onTimeChange(timeAt(event));
    },
    onPointerCancel: () => { drag.current = undefined; },
  });
  const head = Math.min(duration, Math.max(0, time));
  return <section aria-label={studioName('shelf')} className="flex min-w-0 items-start gap-3 px-3 py-2">
    <div className="grid min-w-0 flex-1 grid-cols-[7rem_minmax(0,1fr)] gap-x-2 gap-y-1">
      {rows.map((row) => {
        const rowChosen = row.kind === 'bars' && isChosen(row.selects);
        const name = <>{row.glyph}<span className="truncate font-mono">{row.name}</span></>;
        return <div key={row.key} role="group" aria-label={row.name} className="col-span-2 grid grid-cols-subgrid items-center">
          {row.kind === 'bars' && row.selects
            ? <Button variant="bare" size="content" align="start" pressed={rowChosen} onClick={() => onSelect(row.selects!)}
              className={clsx('flex h-5 min-w-0 items-center gap-1 text-left text-xs', rowChosen ? 'text-shell-foreground' : 'text-shell-muted')}>
              {name}
            </Button>
            : <span className="flex h-5 min-w-0 items-center gap-1 text-xs text-shell-muted">{name}</span>}
          <div className="relative h-5 touch-none" {...axis(row)}>
            {row.kind === 'segments'
              ? row.segments.map((segment) => <Segment key={segment.key} segment={segment} duration={duration}
                chosen={isChosen(segment.selects)} />)
              : row.bars.map((bar) => <span key={bar.start} data-start={bar.start} data-end={bar.end} data-share={bar.some ? 'some' : 'all'}
                className={clsx('absolute inset-y-1 rounded-sm bg-shell-foreground',
                  bar.some ? rowChosen ? 'opacity-30' : 'opacity-[0.15]' : rowChosen ? 'opacity-70' : 'opacity-40')}
                style={{ left: percent(bar.start, duration), width: percent(bar.end - bar.start, duration) }} />)}
            <span aria-hidden="true" className="pointer-events-none absolute -inset-y-0.5 w-px bg-shell-accent"
              style={{ left: percent(head, duration) }} />
          </div>
        </div>;
      })}
      <span />
      <Ticks duration={duration} stepSeconds={stepSeconds} />
    </div>
    {trailing}
  </section>;
}
