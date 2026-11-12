import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { fetchSavedTrainRuns, runSavedTrain } from '@/api/jobs';
import { apiErrorMessage } from '@/api/request';
import { StudioRuns } from '@/components/pipeline/StudioRuns';
import type { PlanRunView, PlanStepView, PlanView } from '@/features/pipeline/planView';
import { emptyPipelineSelection, selectedRuns, type PipelineSelection } from '@/features/pipeline/selection';
import type { StudioSavedTrainRuns } from '@/generated/studioContracts';
import { useStudioJobActions, useStudioJobs } from '@/hooks/useStudioJobs';
import { openProjectEditor } from './openProjectEditor';
import type { ProjectEditController } from './projectEditController';
import { StudioTrainAuthoringForm } from './StudioTrainAuthoringForm';

type EditorSession = Awaited<ReturnType<typeof openProjectEditor>>;
type SavedRow = StudioSavedTrainRuns['rows'][number];

function rowId({ key }: SavedRow) {
  return JSON.stringify([key.variant, key.step, key.row]);
}

function runView(row: SavedRow): PlanRunView {
  const attempt = row.attempts.at(-1);
  return {
    id: rowId(row), label: row.label, variant: row.key.variant, values: {},
    status: row.status, reason: row.reason ?? undefined,
    progress: attempt?.progress ?? undefined,
  };
}

function TrainResults({ row }: { row: SavedRow }) {
  const results = [
    ...(row.result ? [{ label: 'Current result', result: row.result }] : []),
    ...(row.previous ? [{ label: 'Previous result', result: row.previous }] : []),
  ];
  if (!results.length) return null;
  return (
    <details className="mt-3 rounded border border-slate-200 p-3 text-sm">
      <summary className="cursor-pointer font-medium">
        {results.length === 2 ? 'Compare receipts' : results[0].label}
      </summary>
      {row.change && (
        <p className="mt-2 break-all">
          <code>{row.change.path}</code>{' '}
          <code>{JSON.stringify(row.change.before)}</code> → <code>{JSON.stringify(row.change.after)}</code>
        </p>
      )}
      <div className="mt-2 grid gap-3 sm:grid-cols-2">
        {results.map(({ label, result }) => (
          <section key={label} aria-label={label} className="min-w-0">
            <h3 className="font-medium">{label}</h3>
            <dl className="space-y-1">
              <div><dt>Request</dt><dd className="break-all"><code>{result.request_id}</code></dd></div>
              <div><dt>Receipt</dt><dd className="break-all"><code>{result.receipt.manifest_id}</code></dd></div>
              <div><dt>SHA-256</dt><dd className="break-all"><code>{result.receipt.manifest_sha256}</code></dd></div>
              {result.progress && <div><dt>Completed updates</dt><dd>{result.progress.completed_updates}</dd></div>}
            </dl>
          </section>
        ))}
      </div>
    </details>
  );
}

export default function SavedTrainWorkspace({ projectId }: { projectId: string }) {
  const [session, setSession] = useState<EditorSession>();
  const [error, setError] = useState<string>();
  useEffect(() => {
    const controller = new AbortController();
    void openProjectEditor(projectId, controller.signal).then(
      (opened) => { if (!controller.signal.aborted) setSession(opened); },
      (cause: unknown) => {
        if (!controller.signal.aborted) setError(apiErrorMessage(cause));
      },
    );
    return () => controller.abort();
  }, [projectId]);
  if (error) return <main className="p-4"><p role="alert">{error}</p></main>;
  if (!session) return <main aria-busy="true" />;
  return <SavedProjectTrain editor={session.editor} />;
}

function SavedProjectTrain({ editor }: { editor: ProjectEditController }) {
  const { saved, configuration, pending, outside } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const jobs = useStudioJobs();
  const steps = Object.entries(saved.project.steps ?? {}).filter(([, step]) => step.kind === 'train');
  const [requestedStep, setRequestedStep] = useState<string>();
  const [commandError, setCommandError] = useState<string>();
  const stepId = steps.find(([id]) => id === requestedStep)?.[0] ?? steps[0]?.[0];
  const projection = useQuery({
    queryKey: ['saved-train-runs', saved.project_id, saved.revision, jobs.data?.cursor],
    enabled: jobs.data !== undefined,
    queryFn: ({ signal }) => fetchSavedTrainRuns(saved.project_id, saved.revision, signal),
    placeholderData: keepPreviousData,
    gcTime: 0,
    retry: false,
  });
  const data = projection.data;
  const rows = data?.rows.filter((row) => row.key.step === stepId) ?? [];
  const view: PlanView = {
    edges: [],
    steps: steps.map(([id]) => ({
      id, label: id, kind: 'train', inputs: [], sweeps: [], statusCounts: {},
      runs: (data?.rows ?? []).filter((row) => row.key.step === id).map(runView),
    })),
  };
  const step = view.steps.find(({ id }) => id === stepId);
  return (
    <main className="mx-auto max-w-4xl space-y-4 p-4">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-lg font-semibold">{configuration.project}</h1>
        {step && (
          <label className="flex gap-2">
            Train step
            <select value={step.id} onChange={(event) => {
              setRequestedStep(event.target.value);
              setCommandError(undefined);
            }}>
              {view.steps.map(({ id, label }) => <option key={id} value={id}>{label}</option>)}
            </select>
          </label>
        )}
      </header>
      {jobs.error && <p role="alert">{apiErrorMessage(jobs.error)}</p>}
      {projection.error && <p role="alert">{apiErrorMessage(projection.error)}</p>}
      {commandError && <p role="alert">{commandError}</p>}
      {step && (
        <SavedTrainStep
          key={`${step.id}:${data?.revision}`} editor={editor}
          step={step} rows={rows} view={view} revision={data?.revision}
          ready={data?.revision === saved.revision && !pending && !outside}
          reportError={setCommandError}
        />
      )}
    </main>
  );
}

function SavedTrainStep({ editor, step, rows, view, revision, ready, reportError }: {
  editor: ProjectEditController;
  step: PlanStepView;
  rows: SavedRow[];
  view: PlanView;
  revision: string | undefined;
  ready: boolean;
  reportError: (message: string | undefined) => void;
}) {
  const [selection, setSelection] = useState(emptyPipelineSelection);
  const [busy, setBusy] = useState(false);
  const submission = useRef<{ intent: string; token: string } | undefined>(undefined);
  const jobs = useStudioJobActions();
  const selected = selectedRuns(step, selection.selection);
  const focusedRow = selected.length === 1 ? rows.find((row) => rowId(row) === selected[0].id) : undefined;

  async function run(selected: PipelineSelection) {
    if (busy || !ready || !revision) return;
    const ids = new Set(selectedRuns(step, selected).map(({ id }) => id));
    const keys = rows.filter((row) => ids.has(rowId(row))).map(({ key }) => key);
    if (!keys.length) return;
    setBusy(true);
    reportError(undefined);
    try {
      if (await editor.flush() !== revision) throw new Error('The project changed. Select a row from the current plan.');
      const project = editor.getSnapshot().saved.project_id;
      const intent = JSON.stringify([revision, keys]);
      const token = submission.current?.intent === intent ? submission.current.token : crypto.randomUUID();
      submission.current = { intent, token };
      const result = await runSavedTrain(project, { revision, run_keys: keys, token });
      submission.current = undefined;
      const refusal = result.items.find(({ disposition }) => disposition === 'rejected');
      if (refusal?.reason) reportError(refusal.reason);
      jobs.refresh();
    } catch (cause) {
      reportError(apiErrorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <StudioRuns
        plan={view} step={step} state={selection} onSelectionChange={setSelection}
        region={{ kind: 'result', missing: [] }}
        definition={<StudioTrainAuthoringForm editor={editor} stepId={step.id} />}
        commands={[{ name: 'run', enabled: ready && !busy, perform: run }]}
      />
      {ready && focusedRow && <TrainResults key={rowId(focusedRow)} row={focusedRow} />}
    </>
  );
}
