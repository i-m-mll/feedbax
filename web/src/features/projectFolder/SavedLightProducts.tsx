import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { z } from 'zod';
import { apiErrorMessage, requestJson } from '@/api/request';
import { studioName } from '@/components/activity/studioVocabulary';
import type { StudioSavedLightHistory } from '@/generated/studioContracts';

const figureProduct = z.object({
  data: z.array(z.record(z.unknown())),
  layout: z.record(z.unknown()).optional(),
  frames: z.array(z.record(z.unknown())).optional(),
});

function SavedPlot({ url, name }: { url: string; name: string }) {
  const target = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [plotError, setPlotError] = useState<unknown>();
  const figure = useQuery({
    queryKey: ['saved-plotly-figure', url],
    queryFn: async ({ signal }) => figureProduct.parse(await requestJson(url, { signal })),
    retry: false,
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
        await Plotly.newPlot(node, { ...structuredClone({ data, layout, frames }), config: { responsive: true } });
        if (active) setReady(true);
      } catch (error) {
        if (active) setPlotError(error);
      }
    })();
    return () => {
      active = false;
      purge?.();
    };
  }, [figure.data]);
  const error = figure.error ?? plotError;
  return <div>
    {Boolean(error) && <p role="alert">{apiErrorMessage(error)}</p>}
    <div ref={target} role="img" aria-label={name} className="min-h-64 w-full"
      aria-busy={!ready && !error || undefined} />
  </div>;
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
      <button type="button" className="rounded-shell-control border border-shell-divider px-2 py-1"
        aria-pressed={!showPrevious} onClick={() => setPreviousSelected(false)}>{studioName('currentResult')}</button>
      <button type="button" className="rounded-shell-control border border-shell-divider px-2 py-1"
        aria-pressed={showPrevious} onClick={() => setPreviousSelected(true)}>{studioName('previousResult')}</button>
    </div>}
    <SavedPlot key={url} url={url} name={name} />
  </section>;
}

export function SavedReportResult({ url }: { url: string }) {
  const [open, setOpen] = useState(false);
  return <details onToggle={({ currentTarget }) => setOpen(currentTarget.open)}>
    <summary className="cursor-pointer">{studioName('result')}</summary>
    {open && <iframe title={studioName('report')} src={url} sandbox="allow-scripts allow-same-origin"
      className="h-[28rem] w-full rounded-shell-control border border-shell-divider" />}
  </details>;
}
