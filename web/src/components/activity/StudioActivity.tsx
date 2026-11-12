import { bindStudioCommand, commandAttributes, jobAllowsCommand } from '@/features/projectWindow/studioCommands';
import { useState } from 'react';
import type { StudioActivityJob } from '@/generated/studioContracts';
import { useStudioJobActions, useStudioJobConnection, useStudioJobs } from '@/hooks/useStudioJobs';
import { StudioRegion, StudioRegionBoundary, studioRegionIssue, type StudioRegionState } from './StudioRegion';
import { StudioRunInspector } from './StudioRunInspector';
import { StudioRunProgress, StudioRunReason, StudioRunStatus } from './StudioRunStatus';
import { studioDefinitionStatus, studioMachineStatus, studioName, studioStep, type StudioRunStatusId } from './studioVocabulary';

const liveStatuses = new Set<StudioRunStatusId>([
  'queued',
  'awaiting_authorization',
  'starting',
  'running',
  'stopping',
  'collecting',
]);

function isLive(job: StudioActivityJob) {
  return liveStatuses.has(job.status);
}

function coordinateValue(value: unknown) {
  return typeof value === 'string' ? value : JSON.stringify(value);
}

function ActivityRun({
  job,
  project,
  selected,
  onOpen,
  onClose,
}: {
  job: StudioActivityJob;
  project?: string;
  selected: boolean;
  onOpen: () => Promise<boolean>;
  onClose: () => void;
}) {
  const actions = useStudioJobActions();
  const [pending, setPending] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const { Icon, tint, name } = studioStep('train');
  const live = isLive(job);
  const coordinates = Object.entries(job.coordinates);

  const execute = async (command: () => Promise<void>) => {
    setPending(true);
    setFailure(null);
    try {
      await command();
    } catch (error) {
      setFailure(studioRegionIssue('run', error).reason);
    } finally {
      setPending(false);
    }
  };

  const acknowledge = async () => {
    if (!live && !job.seen) await actions.seen(job.job_id);
  };

  const open = () => execute(async () => {
    const opened = await onOpen();
    await acknowledge();
    if (opened) onClose();
  });

  const stop = (mode: 'checkpoint' | 'force') => execute(async () => {
    const result = await actions.stop([job.job_id], mode);
    const item = result.find(({ job_id }) => job_id === job.job_id);
    if (item?.error) setFailure(studioRegionIssue('run', new Error(item.error.message)).reason);
  });

  const resume = () => execute(async () => { await actions.resume(job.job_id); });

  return (
    <li
      className="space-y-2 rounded-md border border-l-4 bg-white p-3 shadow-sm"
      style={{ borderLeftColor: tint }}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <button
          type="button"
          aria-current={selected ? 'true' : undefined}
          onClick={() => void open()}
          disabled={pending}
          className="min-w-0 flex-1 rounded-sm text-left hover:text-blue-700 disabled:opacity-60"
        >
          <span className="flex items-center gap-1.5 text-xs font-medium text-slate-700">
            <Icon aria-hidden="true" className="size-3.5 shrink-0" />
            {name}
            {project && <span className="truncate font-normal">{project}</span>}
          </span>
          <span className="block break-words font-medium">{job.label}</span>
        </button>
        <StudioRunStatus status={job.status} />
      </div>
      {coordinates.length > 0 && (
        <dl className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-700">
          {coordinates.map(([axis, value]) => (
            <div key={axis} className="flex gap-1">
              <dt>{axis}</dt>
              <dd className="tabular-nums">{coordinateValue(value)}</dd>
            </div>
          ))}
        </dl>
      )}
      {job.earlier_definition && (
        <p className="text-xs font-medium text-slate-700">{studioDefinitionStatus('earlier')}</p>
      )}
      <StudioRunReason job={job} />
      {job.progress && <StudioRunProgress progress={job.progress} etaSeconds={job.eta_seconds} />}
      {failure && <p role="alert" className="text-xs text-red-700">{failure}</p>}
      <div className="flex flex-wrap gap-2">
        {jobAllowsCommand(job, 'stop') && (
          <ActivityCommand name="stop" pending={pending} onClick={() => void stop('checkpoint')} />
        )}
        {jobAllowsCommand(job, 'forceStop') && (
          <ActivityCommand name="forceStop" pending={pending} onClick={() => void stop('force')} />
        )}
        {jobAllowsCommand(job, 'resume') && (
          <ActivityCommand name="resume" pending={pending} onClick={() => void resume()} />
        )}
        {!live && !job.seen && (
          <ActivityCommand name="dismiss" pending={pending} onClick={() => void execute(acknowledge)} />
        )}
      </div>
    </li>
  );
}

function ActivityCommand({
  name,
  pending,
  onClick,
}: {
  name: 'stop' | 'forceStop' | 'resume' | 'dismiss';
  pending: boolean;
  onClick: () => void;
}) {
  const command = bindStudioCommand(name, onClick, { enabled: !pending });
  return (
    <button
      type="button"
      {...commandAttributes(command)}
      disabled={!command.enabled}
      onClick={command.invoke}
      className="rounded border border-slate-300 bg-white px-2 py-1 text-xs font-medium hover:bg-slate-100 disabled:opacity-60"
    >
      {studioName(name)}
    </button>
  );
}

interface StudioActivityPaneProps {
  projectNames: ReadonlyMap<string, string>;
  onOpenRun: (job: StudioActivityJob) => Promise<boolean>;
  onClose: () => void;
}

export function StudioActivityPane(props: StudioActivityPaneProps) {
  return (
    <StudioRegionBoundary name="activity">
      <ActivityContent {...props} />
    </StudioRegionBoundary>
  );
}

function ActivityContent({ projectNames, onOpenRun, onClose }: StudioActivityPaneProps) {
  const snapshot = useStudioJobs();
  const actions = useStudioJobActions();
  const connection = useStudioJobConnection();
  const [selection, setSelection] = useState<string | null>(null);
  const jobs = snapshot.data?.jobs ?? [];
  const selected = jobs.find((job) => job.job_id === selection);
  const visible = jobs
    .filter((job) => isLive(job) || !job.seen || job === selected)
    .sort((a, b) => (projectNames.get(a.project_id) ?? a.project_id).localeCompare(projectNames.get(b.project_id) ?? b.project_id)
      || a.label.localeCompare(b.label)
      || a.job_id.localeCompare(b.job_id));
  const machine = snapshot.data?.sessions.find((session) => session.id === 'local');
  const machineField = machine && (
    <dl className="flex justify-between rounded-md border border-slate-200 px-3 py-2 text-sm">
      <dt className="font-medium">{studioName('thisMac')}</dt>
      <dd className="text-slate-700">{studioMachineStatus(machine.state)}</dd>
    </dl>
  );
  const state: StudioRegionState = connection === 'disconnected' || snapshot.isError
    ? {
        kind: 'failed',
        issue: studioRegionIssue('activity', snapshot.error, connection === 'disconnected'),
        retry: actions.refresh,
      }
    : snapshot.isPending
      ? { kind: 'loading' }
      : visible.length > 0
        ? { kind: 'result', missing: [] }
        : { kind: 'empty' };

  return (
    <StudioRegion
      state={state}
      skeleton={<div className="h-20 animate-pulse rounded-md bg-slate-200" />}
      empty={(
        <>
          {machineField}
          <ul aria-label={studioName('runs')} className="min-h-10 rounded-md border border-dashed border-slate-200" />
        </>
      )}
    >
      {machineField}
      <ul aria-label={studioName('runs')} className="space-y-2">
        {visible.map((job) => (
          <ActivityRun
            key={job.job_id}
            job={job}
            project={projectNames.get(job.project_id)}
            selected={job === selected}
            onClose={onClose}
            onOpen={async () => {
              const opened = await onOpenRun(job);
              if (!opened) setSelection(job.job_id);
              return opened;
            }}
          />
        ))}
      </ul>
      {selected && <StudioRunInspector job={selected} />}
    </StudioRegion>
  );
}

export function useStudioActivityCount() {
  const snapshot = useStudioJobs();
  const connection = useStudioJobConnection();
  return connection === 'connected' ? snapshot.data?.jobs.filter(isLive).length ?? 0 : 0;
}
