import { useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { ClickPoint, FullAxis, PlotElement } from 'plotly.js-dist-min';
import { useQuery } from '@tanstack/react-query';
import { requestJson } from '@/api/request';
import { StudioRegion, studioRegionIssue, type StudioRegionState } from '@/components/activity/StudioRegion';
import { useRootAttribute } from '@/appearance/useRootAttribute';
import { studioAccessibleName, studioName, studioRunStatus } from '@/components/activity/studioVocabulary';
import { StatusMark } from '@/components/ui/StatusMark';
import { parseContract, type FigureProduct, type StudioSavedLightHistory } from '@/generated/studioContracts';
import { axisAttribute, FigureClockContext, marksTrials, timePanels, traceTrial, trialRestyle, type TimePanel } from './figureClock';
import { Button } from '@/components/ui/Button';
import { Section } from '@/components/ui/Section';

type Fields = Record<string, unknown>;

function themedLayout(layout: Fields | undefined): Fields {
  const style = getComputedStyle(document.documentElement);
  const token = (name: string) => style.getPropertyValue(name).trim();
  const [text, muted, divider] = [token('--text-primary'), token('--text-muted'), token('--app-divider')];
  const template = (layout?.template ?? {}) as { layout?: Fields };
  const base = template.layout ?? {};
  const axis = (key: string) => ({
    ...(base[key] as Fields | undefined), color: muted, gridcolor: divider, linecolor: divider, zerolinecolor: divider,
  });
  const { width: _width, height: _height, title: _title, annotations: _annotations, ...sized } = layout ?? {};
  return {
    ...sized,
    autosize: true,
    legend: { ...(sized.legend as Fields | undefined), orientation: 'h', x: 0, xanchor: 'left', y: -0.2, yanchor: 'top' },
    template: {
      ...template,
      layout: {
        ...base, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { ...(base.font as Fields | undefined), color: text }, xaxis: axis('xaxis'), yaxis: axis('yaxis'),
      },
    },
  };
}

/** Where the playhead is drawn in each time panel, in the plot's pixels; null where the time is outside the panel's range. */
function playheadPlaces(node: PlotElement, panels: readonly TimePanel[], time: number) {
  return panels.map(({ x, y }) => {
    const xa = node._fullLayout[axisAttribute(x)] as FullAxis | undefined;
    const ya = node._fullLayout[axisAttribute(y)] as FullAxis | undefined;
    if (!xa || !ya) return null;
    const left = xa.l2p(time);
    return left < 0 || left > xa._length ? null : { left: xa._offset + left, top: ya._offset, height: ya._length };
  });
}

/** The time panel at a point of the plot, and the time there. */
function timeAt(node: PlotElement, panels: readonly TimePanel[], clientX: number, clientY: number) {
  const box = node.getBoundingClientRect();
  const [px, py] = [clientX - box.left, clientY - box.top];
  for (const { x, y } of panels) {
    const xa = node._fullLayout[axisAttribute(x)] as FullAxis | undefined;
    const ya = node._fullLayout[axisAttribute(y)] as FullAxis | undefined;
    if (xa && ya && px >= xa._offset && px <= xa._offset + xa._length && py >= ya._offset && py <= ya._offset + ya._length) {
      return xa.p2l(px - xa._offset);
    }
  }
  return undefined;
}

/** Of the points a click reports, the trace drawn nearest the pointer, since a unified hover reports every trace at its x. */
function clickedTrace(node: PlotElement, figure: FigureProduct, points: readonly ClickPoint[], clientY: number) {
  const top = node.getBoundingClientRect().top;
  let best: { trace: number; distance: number } | undefined;
  for (const { curveNumber, y } of points) {
    const ya = node._fullLayout[axisAttribute(String(figure.data[curveNumber]?.yaxis ?? 'y'))] as FullAxis | undefined;
    if (!ya || typeof y !== 'number') continue;
    const distance = Math.abs(top + ya._offset + ya.l2p(y) - clientY);
    if (!best || distance < best.distance) best = { trace: curveNumber, distance };
  }
  return best?.trace;
}

function SavedPlot({ url, name }: { url: string; name: string }) {
  const target = useRef<HTMLDivElement>(null);
  const lines = useRef<(HTMLDivElement | null)[]>([]);
  const [drawn, setDrawn] = useState<PlotElement>();
  const [plot, setPlot] = useState<{ attempt: number; error?: unknown }>({ attempt: 0 });
  const theme = useRootAttribute('data-theme');
  const { clock, declare } = useContext(FigureClockContext);
  const latest = useRef(clock);
  latest.current = clock;
  const figure = useQuery({
    queryKey: ['saved-plotly-figure', url],
    queryFn: async ({ signal }) => parseContract('FigureProduct', await requestJson(url, { signal })),
  });
  const runs = figure.data?.layout.meta.runs;
  useEffect(() => {
    if (!declare || !runs) return;
    declare(runs);
    return () => declare(undefined);
  }, [declare, runs]);
  const epoch = clock?.epoch ?? null;
  const panels = useMemo(() => figure.data ? timePanels(figure.data, epoch) : [], [figure.data, epoch]);
  useEffect(() => {
    const node = target.current;
    if (!node || !figure.data) return;
    let active = true;
    let purge: (() => void) | undefined;
    const product = figure.data;
    const { data, layout, frames } = product;
    const timed = timePanels(product, latest.current?.epoch ?? null).length > 0;
    void (async () => {
      try {
        const Plotly = await import('plotly.js-dist-min');
        if (!active) return;
        purge = () => Plotly.purge(node);
        const plotted = await Plotly.newPlot(node, {
          ...structuredClone({ data, layout: { ...themedLayout(layout), ...timed && { dragmode: false } }, frames: frames ?? undefined }),
          config: { responsive: true },
        }) as PlotElement;
        plotted.on('plotly_click', ({ points, event }) => {
          const trace = clickedTrace(plotted, product, points, event.clientY);
          const trial = trace === undefined ? undefined : traceTrial(product, trace);
          if (trial !== undefined && marksTrials(product, latest.current)) latest.current!.pickTrial(trial);
        });
        if (active) setDrawn(plotted);
      } catch (error) {
        if (active) setPlot((current) => ({ ...current, error }));
      }
    })();
    return () => {
      active = false;
      setDrawn(undefined);
      purge?.();
    };
  }, [figure.data, theme, plot.attempt]);
  const restyle = figure.data && trialRestyle(figure.data, clock);
  const restyled = JSON.stringify(restyle);
  useEffect(() => {
    if (!drawn || !restyle) return;
    void import('plotly.js-dist-min').then((Plotly) => Plotly.restyle(drawn, restyle.update, restyle.traces));
    // The restyle is keyed by its value, which a new clock object with the same pick leaves unchanged.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawn, restyled]);
  const time = clock?.time ?? 0;
  const place = useCallback(() => {
    if (!drawn) return;
    const places = playheadPlaces(drawn, panels, latest.current?.time ?? 0);
    places.forEach((at, index) => {
      const line = lines.current[index];
      if (!line) return;
      line.hidden = !at;
      if (at) {
        line.style.transform = `translateX(${at.left}px)`;
        line.style.top = `${at.top}px`;
        line.style.height = `${at.height}px`;
      }
    });
  }, [drawn, panels]);
  useLayoutEffect(place, [place, time]);
  useEffect(() => {
    drawn?.on('plotly_afterplot', place);
  }, [drawn, place]);
  function press(event: React.PointerEvent) {
    if (!drawn || !clock || event.button !== 0) return;
    const at = timeAt(drawn, panels, event.clientX, event.clientY);
    if (at === undefined) return;
    latest.current?.setTime(at);
    const move = (moved: PointerEvent) => {
      const next = timeAt(drawn, panels, moved.clientX, moved.clientY);
      if (next !== undefined) latest.current?.setTime(next);
    };
    const release = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', release);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', release);
  }
  const error = figure.error ?? plot.error;
  const state: StudioRegionState = error
    ? { kind: 'failed', issue: studioRegionIssue('figures', error), retry: () => {
      setPlot(({ attempt }) => ({ attempt: attempt + 1 }));
      if (figure.error) void figure.refetch();
    } }
    : figure.data ? { kind: 'result', missing: [] } : { kind: 'loading' };
  return <StudioRegion state={state} skeleton={<div className="h-[min(70vh,36rem)] min-h-80 rounded bg-shell-subtle motion-safe:animate-pulse" />}>
    <div className="relative" onPointerDown={clock && panels.length ? press : undefined}>
      <div ref={target} role="img" aria-label={studioAccessibleName('figure', name)} className="h-[min(70vh,36rem)] min-h-80 w-full" aria-busy={!drawn || undefined} />
      {drawn && panels.map((_, index) => <div key={index} ref={(line) => { lines.current[index] = line; }} data-figure-playhead=""
        hidden className="pointer-events-none absolute left-0 w-px bg-[var(--text-primary)]" />)}
    </div>
  </StudioRegion>;
}

export function SavedFigureResult({ history, name }: { history: StudioSavedLightHistory; name: string }) {
  const [previousSelected, setPreviousSelected] = useState(false);
  const current = history.result?.products.figure;
  const previous = history.previous?.products.figure;
  const showPrevious = Boolean(previous && (!current || previousSelected));
  const url = showPrevious ? previous : current;
  if (!url) return null;
  return <section aria-label={studioName(showPrevious ? 'previousResult' : 'currentResult')} className="space-y-2 p-2">
    {(showPrevious || current && previous) && <div className="flex items-center gap-2">
      {showPrevious && <StatusMark status="stale" label={studioRunStatus('stale')} />}
      {current && previous && <>
        <Button pressed={!showPrevious} onClick={() => setPreviousSelected(false)}>{studioName('currentResult')}</Button>
        <Button pressed={showPrevious} onClick={() => setPreviousSelected(true)}>{studioName('previousResult')}</Button>
      </>}
    </div>}
    <SavedPlot key={url} url={url} name={name} />
  </section>;
}

export function SavedReportResult({ url }: { url: string }) {
  const [open, setOpen] = useState(false);
  return <Section level="group" name={studioName('result')} open={open} onOpenChange={setOpen}>
    {open && <iframe aria-label={studioName('report')} src={url} sandbox="allow-scripts allow-same-origin"
      className="h-[28rem] w-full rounded-shell-control border border-shell-divider" />}
  </Section>;
}
