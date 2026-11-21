import { useState, type ReactNode } from 'react';
import { StudioRegion, useRegionPending, type StudioRegionState } from '@/components/activity/StudioRegion';
import { StudioTrainResults, type TrainMetricResult } from '@/components/activity/StudioTrainResults';
import { studioName, studioReason } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { Section } from '@/components/ui/Section';
import type { StudioSavedTrainRuns } from '@/generated/studioContracts';
import { savedRunId, type SavedTrainRow as SavedRow } from './projectWindowPlan';
import { StudioEvaluationResults } from './StudioEvaluationResults';

function ResultDetail({ name, result }: { name: 'currentResult' | 'previousResult'; result: NonNullable<SavedRow['result']> }) {
  const [copyError, setCopyError] = useState(false);
  return <section aria-label={studioName(name)} className="min-w-0">
    <h3 className="font-medium">{studioName(name)}</h3>
    {result.progress && <dl className="space-y-1">
      <div><dt>{studioName('updates')}</dt><dd>{result.progress.completed_updates}</dd></div>
    </dl>}
    <Button onClick={async () => {
      setCopyError(false);
      try {
        await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      } catch {
        setCopyError(true);
      }
    }}>{studioName('copyDetails')}</Button>
    {copyError && <p role="alert">{studioReason('copyFailed', studioName('details'))}</p>}
  </section>;
}

/**
 * The run the Results show, read from one plan: the focused row of the step, or its only row, with its chosen or
 * latest attempt, the step's evaluations among the targeted runs, and the one of them in focus.
 */
export function focusedResult(source: StudioSavedTrainRuns | undefined, stepId: string | undefined, focusRunId: string | null | undefined,
  runIds: ReadonlySet<string>, chosen: { runId: string; jobId: string } | undefined) {
  const rows = source?.rows.filter(({ key }) => key.step === stepId) ?? [];
  const row = rows.find(({ key }) => savedRunId(key) === focusRunId) ?? (rows.length === 1 ? rows[0] : undefined);
  const selectedAttempt = row && chosen?.runId === savedRunId(row.key)
    ? row.attempts.find(({ job_id }) => job_id === chosen.jobId) : undefined;
  const scene = source?.window_map?.evaluations?.filter(({ key }) => key.step === stepId && runIds.has(savedRunId(key))) ?? [];
  const evaluation = scene.find(({ key }) => savedRunId(key) === focusRunId) ?? (scene.length === 1 ? scene[0] : undefined);
  return { row, attempt: selectedAttempt ?? row?.attempts.at(-1), scene, evaluation };
}

function TrainResults({ row, attempt, ...plot }: Omit<Parameters<typeof StudioTrainResults>[0], 'selection'> & {
  attempt: SavedRow['attempts'][number] | undefined;
}) {
  const [comparing, setComparing] = useState(false);
  const results: TrainMetricResult[] = [
    ...(row.result ? [{ name: 'currentResult', result: row.result } as const] : []),
    ...(row.previous ? [{ name: 'previousResult', result: row.previous } as const] : []),
  ];
  const selection = comparing && results.length ? { kind: 'results', results } as const
    : attempt ? { kind: 'attempt', attempt } as const : undefined;
  return (
    <>
      {selection && <StudioTrainResults {...plot} row={row} selection={selection} />}
      {results.length > 0 && <Section level="group" className="mt-3" bodyClassName="text-sm" open={comparing}
        onOpenChange={setComparing} name={studioName(results.length === 2 ? 'compareResults' : results[0].name)}>
        {row.change && (
          <p className="mt-2 break-all">
            <code>{row.change.path}</code>{' '}
            <code>{JSON.stringify(row.change.before)}</code> → <code>{JSON.stringify(row.change.after)}</code>
          </p>
        )}
        <div className="mt-2 grid gap-3 sm:grid-cols-2">
          {results.map(({ name, result }) => <ResultDetail key={name} name={name} result={result} />)}
        </div>
      </Section>}
    </>
  );
}

/**
 * The Results of the focused run in the plan on screen. Until a plan has been read they are loading; after that they
 * keep the plan on screen, marked pending while the shown revision's plan is read, and a replay waits for that plan.
 */
export function SavedResults({ shown, current, failure, result, project, stopped, onReplay, fallback }: {
  /** The plan on screen, which is of an earlier revision while the shown revision's plan is read. */
  shown: StudioSavedTrainRuns | undefined;
  /** Whether `shown` is the plan of the shown revision. */
  current: boolean;
  failure?: StudioRegionState;
  /** The focused run, read from `shown` with `focusedResult`. */
  result: ReturnType<typeof focusedResult>;
  project: string;
  stopped?: ReactNode;
  onReplay(update: number): void;
  /** Shown when the plan on screen has no Results for the focus. */
  fallback: ReactNode;
}) {
  const region = useRegionPending(!!shown && !current);
  if (failure) return <StudioRegion state={failure} />;
  if (!shown) return <StudioRegion state={{ kind: 'loading' }} />;
  const { row, attempt, evaluation } = result;
  if (evaluation?.checkpoint_series) return <div {...region}>
    <StudioEvaluationResults evaluation={evaluation} replayable={current} onReplay={onReplay} />
  </div>;
  if (!row) return <>{fallback}</>;
  return <div {...region}>
    {stopped}
    <TrainResults key={savedRunId(row.key)} project={project} revision={shown.revision}
      evaluations={shown.window_map?.evaluations ?? []} mapRows={shown.window_map?.rows ?? []}
      rows={shown.rows} row={row} attempt={attempt} />
  </div>;
}
