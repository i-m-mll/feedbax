import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { fetchSavedTrainRuns, runSavedTrain } from '@/api/jobs';
import { apiErrorMessage } from '@/api/request';
import { StudioTrainResults } from '@/components/activity/StudioTrainResults';
import type { PipelineMapStepSource, StudioPipelineMapProps } from '@/components/pipeline/StudioPipelineMap';
import { StudioRuns } from '@/components/pipeline/StudioRuns';
import type { PlanStepView, PlanView } from '@/features/pipeline/planView';
import {
  emptyPipelineSelection, selectedRuns, selectStep,
  type PipelineSelection, type PipelineSelectionState,
} from '@/features/pipeline/selection';
import type { StudioSavedTrainRuns } from '@/generated/studioContracts';
import { useStudioJobActions, useStudioJobs } from '@/hooks/useStudioJobs';
import { openProjectEditor } from './openProjectEditor';
import type { ProjectEditController } from './projectEditController';
import { StudioTrainAuthoringForm } from './StudioTrainAuthoringForm';
import { createCheckpointTrain, rowId, savedTrainPlan } from './savedTrainPlan';

type EditorSession = Awaited<ReturnType<typeof openProjectEditor>>;
type SavedRow = StudioSavedTrainRuns['rows'][number];

export interface SavedTrainRegions {
  map: StudioPipelineMapProps;
  details: ReactNode;
}

type PlaceRegions = (regions: SavedTrainRegions) => ReactNode;

function TrainReceipts({ row }: { row: SavedRow }) {
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

export default function SavedTrainWorkspace({ projectId, placeRegions }: {
  projectId: string;
  placeRegions?: PlaceRegions;
}) {
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
  return <SavedProjectTrain editor={session.editor} placeRegions={placeRegions} />;
}

function SavedProjectTrain({ editor, placeRegions }: { editor: ProjectEditController; placeRegions?: PlaceRegions }) {
  const { saved, configuration, pending, outside } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const jobs = useStudioJobs();
  const stepIds = useMemo(() => Object.entries(saved.project.steps ?? {})
    .filter(([, step]) => step.kind === 'train').map(([id]) => id), [saved.project.steps]);
  const [selection, setSelection] = useState(emptyPipelineSelection);
  const [focusStep, setFocusStep] = useState<string>();
  const [commandError, setCommandError] = useState<string>();
  const creating = useRef(false);
  const selected = selection.selection;
  const requested = selected?.kind === 'steps' ? selected.stepIds.at(-1) : selected?.stepId;
  const stepId = requested && stepIds.includes(requested) ? requested : stepIds[0];
  const projection = useQuery({
    queryKey: ['saved-train-runs', saved.project_id, saved.revision, jobs.data?.cursor],
    queryFn: ({ signal }) => fetchSavedTrainRuns(saved.project_id, saved.revision, signal),
    placeholderData: keepPreviousData,
    gcTime: 0,
    retry: false,
  });
  const data = projection.data;
  const rows = data?.rows.filter((row) => row.key.step === stepId) ?? [];
  const view = useMemo(() => savedTrainPlan(stepIds, data), [stepIds, data]);
  const step = view.steps.find(({ id }) => id === stepId);
  const ready = data?.revision === saved.revision && !pending && !outside;

  function choose(next: PipelineSelectionState) {
    setSelection(next);
    setFocusStep(undefined);
    setCommandError(undefined);
  }

  async function newStep(source?: PipelineMapStepSource) {
    if (!source || !data || !ready || creating.current) return;
    creating.current = true;
    setCommandError(undefined);
    try {
      const id = await createCheckpointTrain(editor, source, view, data);
      setSelection(selectStep(emptyPipelineSelection, id));
      setFocusStep(id);
    } catch (error) {
      setCommandError(apiErrorMessage(error));
    } finally {
      creating.current = false;
    }
  }

  const details = step && (
    <SavedTrainStep
      key={`${step.id}:${data?.revision}`} editor={editor} project={saved.project_id}
      step={step} rows={rows} view={view} revision={data?.revision}
      planRows={data?.rows ?? []} ready={ready} reportError={setCommandError}
      selection={selection} onSelectionChange={choose} onNewStep={newStep}
      focusFirstField={ready && focusStep === step.id} onFirstFieldFocused={() => setFocusStep(undefined)}
    />
  );
  return (
    <main className="mx-auto max-w-4xl space-y-4 p-4">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-lg font-semibold">{configuration.project}</h1>
        {step && (
          <label className="flex gap-2">
            Train step
            <select value={step.id} onChange={(event) => choose(selectStep(selection, event.target.value))}>
              {view.steps.map(({ id, label }) => <option key={id} value={id}>{label}</option>)}
            </select>
          </label>
        )}
      </header>
      {jobs.error && <p role="alert">{apiErrorMessage(jobs.error)}</p>}
      {projection.error && <p role="alert">{apiErrorMessage(projection.error)}</p>}
      {commandError && <p role="alert">{commandError}</p>}
      {placeRegions ? placeRegions({
        map: { plan: view, state: selection, onSelectionChange: choose, onNewStep: ready && step ? newStep : undefined },
        details,
      }) : details}
    </main>
  );
}

function SavedTrainStep({
  editor, project, step, rows, planRows, view, revision, ready, reportError,
  selection, onSelectionChange, onNewStep, focusFirstField, onFirstFieldFocused,
}: {
  editor: ProjectEditController;
  project: string;
  step: PlanStepView;
  rows: SavedRow[];
  planRows: SavedRow[];
  view: PlanView;
  revision: string | undefined;
  ready: boolean;
  reportError: (message: string | undefined) => void;
  selection: PipelineSelectionState;
  onSelectionChange: (state: PipelineSelectionState) => void;
  onNewStep: (source: PipelineMapStepSource) => void;
  focusFirstField: boolean;
  onFirstFieldFocused: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const submission = useRef<{ intent: string; token: string } | undefined>(undefined);
  const jobs = useStudioJobActions();
  const focus = selection.focus;
  const focusedRow = focus?.stepId === step.id ? rows.find((row) => rowId(row) === focus.runId) : undefined;
  const attempt = focusedRow?.attempts.at(-1);

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
        plan={view} step={step} state={selection} onSelectionChange={onSelectionChange}
        region={{ kind: 'result', missing: [] }}
        definition={<StudioTrainAuthoringForm editor={editor} stepId={step.id}
          focusFirstField={focusFirstField} onFirstFieldFocused={onFirstFieldFocused} />}
        commands={[
          { name: 'run', enabled: ready && !busy, perform: run },
          { name: 'newStep', enabled: ready && !busy, perform: (scope) => {
            onNewStep({ stepId: step.id, selection: scope.kind === 'steps' ? null : scope });
          } },
        ]}
      />
      {ready && focusedRow && (
        <>
          {attempt && <StudioTrainResults project={project} rows={planRows} row={focusedRow} attempt={attempt} />}
          <TrainReceipts key={rowId(focusedRow)} row={focusedRow} />
        </>
      )}
    </>
  );
}
