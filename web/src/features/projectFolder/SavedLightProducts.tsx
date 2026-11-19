import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { z } from 'zod';
import { requestJson } from '@/api/request';
import { StudioRegion, studioRegionIssue, type StudioRegionState } from '@/components/activity/StudioRegion';
import { useRootAttribute } from '@/appearance/useRootAttribute';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import type { StudioSavedLightHistory } from '@/generated/studioContracts';
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
  return {
    ...layout,
    template: {
      ...template,
      layout: {
        ...base, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { ...(base.font as Fields | undefined), color: text }, xaxis: axis('xaxis'), yaxis: axis('yaxis'),
      },
    },
  };
}

const figureProduct = z.object({
  data: z.array(z.record(z.unknown())),
  layout: z.record(z.unknown()).optional(),
  frames: z.array(z.record(z.unknown())).optional(),
});

function SavedPlot({ url, name }: { url: string; name: string }) {
  const target = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [plot, setPlot] = useState<{ attempt: number; error?: unknown }>({ attempt: 0 });
  const theme = useRootAttribute('data-theme');
  const figure = useQuery({
    queryKey: ['saved-plotly-figure', url],
    queryFn: async ({ signal }) => figureProduct.parse(await requestJson(url, { signal })),
  });
  useEffect(() => {
    const node = target.current;
    if (!node || !figure.data) return;
    let active = true;
    let purge: (() => void) | undefined;
    const { data, layout, frames } = figure.data;
    void (async () => {
      try {
        const Plotly = await import('plotly.js-dist-min');
        if (!active) return;
        purge = () => Plotly.purge(node);
        await Plotly.newPlot(node, {
          ...structuredClone({ data, layout: themedLayout(layout), frames }), config: { responsive: true },
        });
        if (active) setReady(true);
      } catch (error) {
        if (active) setPlot((current) => ({ ...current, error }));
      }
    })();
    return () => {
      active = false;
      purge?.();
    };
  }, [figure.data, theme, plot.attempt]);
  const error = figure.error ?? plot.error;
  const state: StudioRegionState = error
    ? { kind: 'failed', issue: studioRegionIssue('figures', error), retry: () => {
      setPlot(({ attempt }) => ({ attempt: attempt + 1 }));
      if (figure.error) void figure.refetch();
    } }
    : figure.data ? { kind: 'result', missing: [] } : { kind: 'loading' };
  return <StudioRegion state={state} skeleton={<div className="min-h-64 rounded bg-shell-subtle motion-safe:animate-pulse" />}>
    <div ref={target} role="img" aria-label={studioAccessibleName('figure', name)} className="min-h-64 w-full" aria-busy={!ready || undefined} />
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
    {current && previous && <div className="flex gap-2">
      <Button pressed={!showPrevious} onClick={() => setPreviousSelected(false)}>{studioName('currentResult')}</Button>
      <Button pressed={showPrevious} onClick={() => setPreviousSelected(true)}>{studioName('previousResult')}</Button>
    </div>}
    <SavedPlot key={url} url={url} name={name} />
  </section>;
}

export function SavedReportResult({ url }: { url: string }) {
  const [open, setOpen] = useState(false);
  return <Section nested name={studioName('result')} open={open} onOpenChange={setOpen}>
    {open && <iframe aria-label={studioName('report')} src={url} sandbox="allow-scripts allow-same-origin"
      className="h-[28rem] w-full rounded-shell-control border border-shell-divider" />}
  </Section>;
}
