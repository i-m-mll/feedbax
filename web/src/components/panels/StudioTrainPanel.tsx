import { useMemo, useRef, useState, type FormEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { previewStudioRuns, submitStudioJobs } from '@/api/jobs';
import { apiErrorMessage } from '@/api/request';
import { StudioRunInspector } from '@/components/activity/StudioRunInspector';
import { StudioRunProgress, StudioRunReason, StudioRunStatus } from '@/components/activity/StudioRunStatus';
import { focusForJob, useStudioRunFocus } from '@/components/activity/studioRunFocus';
import {
  studioAccessibleName,
  studioDefinitionStatus,
  studioName,
  studioReason,
} from '@/components/activity/studioVocabulary';
import type { StudioActivityJob } from '@/generated/studioContracts';
import { useRootGraph } from '@/hooks/useRootGraph';
import { useStudioJobActions, useStudioJobConnection, useStudioJobs } from '@/hooks/useStudioJobs';
import { captureActiveStudioDocument, studioPersistence } from '@/services/studioPersistence';
import { useGraphStore } from '@/stores/graphStore';
import { useProjectsStore } from '@/stores/projectsStore';
import { getScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import type { StudioScenarioSpec } from '@/types/workspace';
import {
  formatAxisValue,
  initialMatrixSpec,
  parseAxisValuesInput,
  upsertTrainMatrixAxis,
  validateAxisPath,
  workspaceWithMatrixSelection,
  workspaceWithoutMatrixSelection,
  type TrainMatrixAxisDraft,
  type TrainMatrixSpec,
} from '@/utils/trainMatrix';

type PreviewRun = Awaited<ReturnType<typeof previewStudioRuns>>['rows'][number];
type RunIdentity = Pick<PreviewRun, 'run_key' | 'request_id'>;
type RunRow = { run: PreviewRun | StudioActivityJob; job?: StudioActivityJob };

const button = 'rounded-md border border-slate-300 px-2.5 py-1 font-medium hover:bg-slate-100 disabled:opacity-50';
const input = 'mt-1 block h-8 w-full rounded-md border border-slate-300 px-2';
const initialAxis = {
  label: studioName('learningRate'),
  path: 'training_spec.optimizer.params.learning_rate',
  values: '0.001, 0.0003',
};

function runIdentity(run: RunIdentity): string {
  return JSON.stringify([run.run_key, run.request_id]);
}

function SweepEditor({
  matrix,
  scenario,
  onChange,
}: {
  matrix: TrainMatrixSpec;
  scenario: StudioScenarioSpec | null;
  onChange: (matrix: TrainMatrixSpec) => void;
}) {
  const [draft, setDraft] = useState(initialAxis);
  const [editing, setEditing] = useState<string>();
  const path = draft.path.trim();
  const pathError = Boolean(path && (
    validateAxisPath(path, scenario) ||
    (editing && matrix.axes.some((axis) => axis.path === path && axis.id !== editing))
  ));
  const values = parseAxisValuesInput(draft.values);

  function saveAxis(event: FormEvent) {
    event.preventDefault();
    if (!path || pathError || values.length === 0) return;
    onChange(upsertTrainMatrixAxis(matrix, {
      path,
      label: draft.label.trim() || path,
      values,
    }, editing));
    setEditing(undefined);
  }

  function editAxis(axis: TrainMatrixAxisDraft) {
    setEditing(axis.id);
    setDraft({ label: axis.label, path: axis.path, values: JSON.stringify(axis.values) });
  }

  return (
    <section className="space-y-3 rounded-lg border border-slate-200 bg-white p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="font-semibold">{studioName('sweep')}</h2>
        {matrix.axes.length > 1 && (
          <label className="flex items-center gap-2">
            {studioName('mode')}
            <select
              className={button}
              value={matrix.mode}
              onChange={(event) => onChange({
                ...matrix,
                mode: event.target.value === 'zip' ? 'zip' : 'cross',
                manualCoordinates: undefined,
              })}
            >
              <option value="cross">{studioName('cross')}</option>
              <option value="zip">{studioName('zip')}</option>
            </select>
          </label>
        )}
      </div>
      {matrix.axes.length > 0 && (
        <ul className="divide-y divide-slate-200">
          {matrix.axes.map((axis) => (
            <li key={axis.id} className="flex flex-wrap items-center gap-2 py-1">
              <strong>{axis.label}</strong>
              <span className="mr-auto tabular-nums">{axis.values.map(formatAxisValue).join(' · ')}</span>
              <button type="button" className={button} aria-label={studioAccessibleName('edit', axis.label)} onClick={() => editAxis(axis)}>
                {studioName('edit')}
              </button>
              <button
                type="button"
                className={button}
                aria-label={studioAccessibleName('remove', axis.label)}
                onClick={() => {
                  if (editing === axis.id) setEditing(undefined);
                  onChange({
                    ...matrix,
                    axes: matrix.axes.filter((item) => item.id !== axis.id),
                    manualCoordinates: undefined,
                  });
                }}
              >
                {studioName('remove')}
              </button>
            </li>
          ))}
        </ul>
      )}
      <form aria-label={studioName('axis')} onSubmit={saveAxis} className="flex flex-wrap items-end gap-2">
        <label className="min-w-28 flex-1">
          {studioName('name')}
          <input className={input} value={draft.label} onChange={(event) => setDraft({ ...draft, label: event.target.value })} />
        </label>
        <label className="min-w-48 flex-[2]">
          {studioName('path')}
          <input className={input} value={draft.path} aria-invalid={pathError} onChange={(event) => setDraft({ ...draft, path: event.target.value })} />
        </label>
        <label className="min-w-32 flex-1">
          {studioName('values')}
          <input className={input} value={draft.values} onChange={(event) => setDraft({ ...draft, values: event.target.value })} />
        </label>
        <button type="submit" className={button} disabled={!path || pathError || values.length === 0}>
          {studioName(editing ? 'edit' : 'add')}
        </button>
        {pathError && <p role="alert" className="basis-full text-red-700">{studioReason('invalidPath', path)}</p>}
      </form>
    </section>
  );
}

export function StudioTrainPanel({ stageId }: { stageId: string }) {
  const tabId = useProjectsStore((state) => state.activeTabId);
  const projectId = useGraphStore((state) => state.graphId);
  const revision = useGraphStore((state) => state.localRevision);
  const graph = useRootGraph();
  const workspace = useWorkspaceStore((state) => state.workspace);
  const setWorkspace = useWorkspaceStore((state) => state.setWorkspace);
  const stage = workspace?.stages.find((item) => item.id === stageId);
  const scenario = getScenario(workspace, stage?.scenario_id);
  const matrix = initialMatrixSpec(stage, null);
  const focus = useStudioRunFocus((state) => state.focus);
  const selectFocus = useStudioRunFocus((state) => state.select);
  const jobs = useStudioJobs();
  const connection = useStudioJobConnection();
  const actions = useStudioJobActions();
  const preview = useQuery({
    queryKey: ['studio-train-preview', tabId, projectId, stageId, revision],
    enabled: Boolean(workspace && stage),
    queryFn: ({ signal }) => previewStudioRuns({
      project_id: projectId || 'draft',
      graph,
      workspace: workspace!,
      selections: [{ stage_id: stageId }],
    }, signal),
    retry: false,
    staleTime: Infinity,
  });
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const commandRunning = useRef(false);
  const retryToken = useRef<{ intent: string; token: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rowErrors, setRowErrors] = useState<ReadonlyMap<string, string>>(new Map());

  const currentRuns = preview.data?.rows ?? [];
  const previewIssues = preview.data?.issues ?? [];
  const previewReady = preview.isSuccess && !previewIssues.length;
  const currentKeys = new Set(currentRuns.map(runIdentity));
  const attempts = useMemo(() => {
    const stageJobs = jobs.data?.jobs.filter(
      (job) => job.project_id === projectId && job.stage_id === stageId,
    ) ?? [];
    const continued = new Map<string, StudioActivityJob>();
    for (const job of stageJobs) {
      if (job.continues) continued.set(job.continues, job);
    }
    const latest = stageJobs.filter((job) => !continued.has(job.job_id));
    return {
      continued,
      byId: new Map(stageJobs.map((job) => [job.job_id, job])),
      byRun: new Map(latest.map((job) => [runIdentity(job), job])),
      byRequest: new Map(latest.map((job) => [job.request_id, job])),
    };
  }, [jobs.data, projectId, stageId]);
  const jobForRun = (run: RunIdentity) => attempts.byRun.get(runIdentity(run)) ?? attempts.byRequest.get(run.request_id);
  const currentRows: RunRow[] = currentRuns.map((run) => ({ run, job: jobForRun(run) }));
  const previousRows: RunRow[] = [...attempts.byRun.values()]
    .filter((job) => !currentKeys.has(runIdentity(job)))
    .map((job) => ({ run: job, job }));
  const selectedRows = currentRows.filter(({ run }) => selected.has(runIdentity(run)));
  const runRows = selected.size ? selectedRows : currentRows;
  const matchingFocus = focus?.projectId === projectId && focus.stageId === stageId ? focus : null;
  let focusedJob = matchingFocus?.jobId ? attempts.byId.get(matchingFocus.jobId) : undefined;
  while (focusedJob && attempts.continued.has(focusedJob.job_id)) {
    focusedJob = attempts.continued.get(focusedJob.job_id);
  }
  if (!focusedJob && matchingFocus) {
    focusedJob = jobForRun({ run_key: matchingFocus.runKey, request_id: matchingFocus.requestId });
  }

  function changeMatrix(next: TrainMatrixSpec) {
    const current = useWorkspaceStore.getState().workspace;
    if (!current) return;
    setSelected(new Set());
    setWorkspace(next.axes.length
      ? workspaceWithMatrixSelection(current, stageId, next)
      : workspaceWithoutMatrixSelection(current, stageId));
  }

  function selectRow(run: PreviewRun) {
    const id = runIdentity(run);
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function command(name: Parameters<typeof studioName>[0], perform: () => Promise<void>) {
    if (commandRunning.current) return;
    commandRunning.current = true;
    setBusy(true);
    setError(null);
    setRowErrors(new Map());
    try {
      await perform();
    } catch (cause) {
      setError(apiErrorMessage(cause, studioReason('unavailable', studioName(name))));
    } finally {
      commandRunning.current = false;
      setBusy(false);
    }
  }

  function submit() {
    if (!runRows.length || !previewReady || preview.isFetching) return;
    void command('run', async () => {
      const draft = captureActiveStudioDocument();
      const isCurrent = () =>
        useProjectsStore.getState().activeTabId === draft.documentId &&
        useGraphStore.getState().localRevision === draft.localRevision;
      const changed = () => new Error(studioReason('changed', stage?.label ?? studioName('train')));
      if (draft.documentId !== tabId || draft.localRevision !== revision) throw changed();

      let savedId = draft.graphId;
      if (!savedId || useGraphStore.getState().isDirty) {
        const saved = await studioPersistence.save(draft, 'manual');
        if (saved.ok === false) throw new Error(saved.message);
        savedId = saved.result.graphId;
      }
      if (!isCurrent()) throw changed();

      const input = {
        project_id: savedId,
        graph: draft.envelope.graph!,
        workspace: draft.workspace,
        selections: [{ stage_id: stageId, run_keys: runRows.map(({ run }) => run.run_key) }],
      };
      const confirmed = await previewStudioRuns(input);
      if (!isCurrent()) throw changed();
      const issue = confirmed.issues[0];
      if (issue) throw new Error(issue.message);
      const confirmedKeys = new Set(confirmed.rows.map(runIdentity));
      if (runRows.some(({ run }) => !confirmedKeys.has(runIdentity(run)))) {
        void preview.refetch();
        throw changed();
      }

      const intent = JSON.stringify([savedId, ...runRows.map(({ run }) => runIdentity(run))]);
      const token = retryToken.current?.intent === intent ? retryToken.current.token : crypto.randomUUID();
      retryToken.current = { intent, token };
      const submission = await submitStudioJobs({ ...input, token });
      retryToken.current = null;
      const byKey = new Map(runRows.map(({ run }) => [run.run_key, run]));
      const rejected = new Map<string, string>();
      for (const item of submission.items) {
        const run = byKey.get(item.run_key);
        if (run && item.disposition === 'rejected') {
          rejected.set(runIdentity(run), item.reason ?? studioReason('unavailable', run.label));
        }
      }
      setRowErrors(rejected);
      const accepted = submission.items.find((item) => item.job_id && item.disposition !== 'rejected');
      const run = accepted && byKey.get(accepted.run_key);
      if (run && isCurrent()) {
        selectFocus({ projectId: savedId, stageId, runKey: run.run_key, requestId: run.request_id, jobId: accepted.job_id ?? undefined });
      }
    });
  }

  function stopJobs(selectedJobs: StudioActivityJob[], mode: 'checkpoint' | 'force') {
    void command(mode === 'force' ? 'forceStop' : 'stop', async () => {
      const byId = new Map(selectedJobs.map((job) => [job.job_id, runIdentity(job)]));
      const result = await actions.stop([...byId.keys()], mode);
      const failures = new Map<string, string>();
      for (const item of result) {
        const key = byId.get(item.job_id);
        if (key && item.error) failures.set(key, item.error.message);
      }
      setRowErrors(failures);
    });
  }

  function resumeJobs(selectedJobs: StudioActivityJob[]) {
    void command('resume', async () => {
      const failures = new Map<string, string>();
      for (const job of selectedJobs) {
        try {
          await actions.resume(job.job_id);
        } catch (cause) {
          failures.set(runIdentity(job), apiErrorMessage(cause, studioReason('unavailable', job.label)));
        }
      }
      setRowErrors(failures);
    });
  }

  function showRun(run: RunIdentity, job: StudioActivityJob) {
    selectFocus({ ...focusForJob(job), runKey: run.run_key, requestId: run.request_id });
    if (job.outcome && !job.seen) void command('openRun', async () => { await actions.seen(job.job_id); });
  }

  function jobCommands(targets: StudioActivityJob[], distinguishRun = false) {
    const checkpoint = targets.filter((job) => job.allowed_actions.includes('checkpoint'));
    const force = targets.filter((job) => job.allowed_actions.includes('force'));
    const resumable = targets.filter((job) => job.allowed_actions.includes('resume'));
    const target = distinguishRun ? targets[0].label : undefined;
    return (
      <>
        {checkpoint.length > 0 && (
          <button type="button" className={button} disabled={busy} aria-label={studioAccessibleName('stop', target)} onClick={() => stopJobs(checkpoint, 'checkpoint')}>
            {studioName('stop')}
          </button>
        )}
        {force.length > 0 && (
          <button type="button" className={button} disabled={busy} aria-label={studioAccessibleName('forceStop', target)} onClick={() => stopJobs(force, 'force')}>
            {studioName('forceStop')}
          </button>
        )}
        {resumable.length > 0 && (
          <button type="button" className={button} disabled={busy} aria-label={studioAccessibleName('resume', target)} onClick={() => resumeJobs(resumable)}>
            {studioName('resume')}
          </button>
        )}
      </>
    );
  }

  function table(rows: RunRow[], name: 'runs' | 'previousRuns', selectable: boolean) {
    if (!rows.length) return null;
    return (
      <table aria-label={studioName(name)} className="w-full text-left">
        <thead className="bg-slate-100 text-slate-700">
          <tr>
            {selectable && <th scope="col" className="px-2 py-1">{studioName('select')}</th>}
            <th scope="col" className="px-2 py-1">{studioName('run')}</th>
            <th scope="col" className="px-2 py-1">{studioName('status')}</th>
            <th scope="col" className="px-2 py-1">{studioName('progress')}</th>
            <th scope="col"><span className="sr-only">{studioName('details')}</span></th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-200 bg-white">
          {rows.map(({ run, job }) => {
            const key = runIdentity(run);
            const reason = rowErrors.get(key) ?? (job && rowErrors.get(runIdentity(job)));
            const status = job?.status ?? (reason ? 'cannot_run' : jobs.data ? 'not_run' : null);
            const selectedJob = matchingFocus?.runKey === run.run_key && matchingFocus.requestId === run.request_id;
            return (
              <tr key={key} className={selectedJob ? 'bg-brand-50' : undefined}>
                {selectable && (
                  <td className="px-2 py-1">
                    <input type="checkbox" aria-label={studioAccessibleName('select', run.label)} checked={selected.has(key)} onChange={() => selectRow(run)} />
                  </td>
                )}
                <th scope="row" className="px-2 py-1 font-medium">
                  {job ? (
                    <button type="button" aria-pressed={selectedJob} className="text-brand-700 hover:underline" onClick={() => showRun(run, job)}>
                      {run.label}
                    </button>
                  ) : run.label}
                  {job && !selectable && (job.earlier_definition || previewReady) && (
                    <span className="block text-xs text-slate-600">{studioDefinitionStatus('earlier')}</span>
                  )}
                </th>
                <td className="px-2 py-1">
                  {status && <StudioRunStatus status={status} />}
                  {reason && <p className="text-xs">{studioReason('observed', reason)}</p>}
                  {job && <StudioRunReason job={job} />}
                </td>
                <td className="px-2 py-1">
                  {job?.progress && <StudioRunProgress progress={job.progress} etaSeconds={job.eta_seconds} />}
                </td>
                <td className="px-2 py-1">
                  {job && <div className="flex flex-wrap gap-1">{jobCommands([job], true)}</div>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    );
  }

  const selectedJobs = [...new Map(selectedRows.flatMap(
    ({ job }) => job ? [[job.job_id, job] as const] : [],
  )).values()];
  return (
    <div className="mx-auto grid max-w-7xl gap-4 px-5 py-4 text-sm text-slate-800 xl:grid-cols-[minmax(0,1fr)_22rem]">
      <div className="min-w-0 space-y-3">
        <SweepEditor matrix={matrix} scenario={scenario} onChange={changeMatrix} />
        <section aria-label={studioName('runs')} aria-busy={preview.isFetching || jobs.isPending} className="space-y-2 rounded-lg border border-slate-200 bg-white p-3">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              className="rounded-md bg-brand-600 px-3 py-1.5 font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
              disabled={busy || !previewReady || preview.isFetching || !runRows.length}
              onClick={submit}
            >
              {studioName('run')}
            </button>
            {selected.size > 0 && (
              <button type="button" className={button} onClick={() => setSelected(new Set())}>
                {studioName('clearSelection')}
              </button>
            )}
            <span className="mr-auto tabular-nums">{runRows.length} / {currentRuns.length}</span>
            {jobCommands(selectedJobs)}
          </div>
          {error && <p role="alert" className="text-red-700">{studioReason('observed', error)}</p>}
          {preview.isError && (
            <div role="alert" className="flex items-center gap-2 text-red-700">
              <p>{apiErrorMessage(preview.error, studioReason('unavailable', studioName('runs')))}</p>
              <button type="button" className={button} onClick={() => void preview.refetch()}>{studioName('retry')}</button>
            </div>
          )}
          {(jobs.isError || connection === 'disconnected') && (
            <div role="alert" className="flex items-center gap-2 text-red-700">
              <p>{apiErrorMessage(jobs.error, studioReason('unreachable', studioName('service')))}</p>
              <button
                type="button"
                className={button}
                disabled={busy}
                onClick={() => void command('retry', async () => { await actions.refresh(); })}
              >
                {studioName('retry')}
              </button>
            </div>
          )}
          {previewIssues.length > 0 && (
            <ul role="alert" className="list-disc pl-4 text-red-700">
              {previewIssues.map((issue) => <li key={`${issue.stage_id}:${issue.path}:${issue.message}`}>{studioReason('observed', issue.message)}</li>)}
            </ul>
          )}
          <div className="overflow-x-auto">{table(currentRows, 'runs', true)}</div>
          {previousRows.length > 0 && (
            <div className="space-y-2 overflow-x-auto">
              <h2 className="font-semibold">{studioName(previewReady ? 'previousRuns' : 'runs')}</h2>
              {table(previousRows, previewReady ? 'previousRuns' : 'runs', false)}
            </div>
          )}
        </section>
      </div>
      {focusedJob && <StudioRunInspector job={focusedJob} />}
    </div>
  );
}
