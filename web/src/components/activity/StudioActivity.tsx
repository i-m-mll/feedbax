import { bindStudioCommand, commandAttributes, jobAllowsCommand } from '@/features/projectWindow/studioCommands';
import { holdWhile } from '@/features/projectWindow/heldGestures';
import { useId, useState, type PropsWithChildren } from 'react';
import { closeStudioMachine } from '@/api/jobs';
import type { StudioActivityJob, StudioRemoteSession } from '@/generated/studioContracts';
import { StudioJobsProvider, useStudioJobActions, useStudioJobs, useStudioJobsProject, useStudioUnreadableJobs } from '@/hooks/useStudioJobs';
import { StudioRegion, StudioRegionBoundary, studioRegionIssue, type StudioRegionState } from './StudioRegion';
import { StudioRunInspector } from './StudioRunInspector';
import { StudioCollectionProgress, StudioRunProgress, StudioRunReason, StudioRunStatus } from './StudioRunStatus';
import {
  studioDateTime, studioDefinitionStatus, studioMachinePower, studioMachineStatus, studioMoney, studioName, studioOpeningStep, studioReason,
  studioStep, type StudioResultState, type StudioRunStatusId,
} from './studioVocabulary';

/** The state a finished job's result is in, where the window knows its validity. */
export type ActivityResultState = (job: StudioActivityJob) => StudioResultState | undefined;
import { Button, IconButton } from '@/components/ui/Button';
import { X } from 'lucide-react';
import { Section } from '@/components/ui/Section';

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

const activityTime = (job: StudioActivityJob) => job.ended_at_ms ?? job.submitted_at_ms;

function coordinateValue(value: unknown) {
  return typeof value === 'string' ? value : JSON.stringify(value);
}

function ActivityRun({
  job,
  selected,
  machine,
  resultState,
  onOpen,
  onClose,
}: {
  job: StudioActivityJob;
  selected: boolean;
  machine?: string;
  resultState?: StudioResultState;
  onOpen: () => Promise<boolean>;
  onClose: () => void;
}) {
  const actions = useStudioJobActions();
  const [pending, setPending] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const { Icon, tint, name } = studioStep(job.kind);
  const live = isLive(job);
  const coordinates = Object.entries(job.coordinates);
  /** Who submitted the run, when it was not this Studio: a run from the command line says so. */
  const client = job.actor?.client === 'studio' ? undefined : job.actor ?? undefined;
  /** When the run ended, or while it is live, when it was submitted. */
  const time = activityTime(job);

  /** Runs a row's request; a command made on the row meanwhile is held until it is done. */
  const execute = (command: () => Promise<void>) => holdWhile(async () => {
    setPending(true);
    setFailure(null);
    try {
      await command();
    } catch (error) {
      setFailure(studioRegionIssue('run', error).reason);
    } finally {
      setPending(false);
    }
  });

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
      className="space-y-2 rounded-md border border-l-4 bg-shell-surface p-3 shadow-sm"
      style={{ borderLeftColor: tint }}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <Button
          variant="bare"
          size="content"
          aria-current={selected ? 'true' : undefined}
          onClick={() => void open()}
          disabled={pending}
          className="min-w-0 flex-1 rounded-sm text-left hover:[&:not([data-disabled])]:underline"
        >
          <span className="flex items-center gap-1.5 text-xs font-medium text-shell-muted">
            <Icon aria-hidden="true" className="size-3.5 shrink-0" />
            {name}
          </span>
          <span className="block break-words font-medium">{job.label}</span>
        </Button>
        <div className="flex items-center gap-1">
          <StudioRunStatus status={resultState ?? job.status} />
          {!live && !job.seen && <ActivityDismiss job={job.job_id} onClick={() => void execute(acknowledge)} />}
        </div>
      </div>
      <time className="block text-xs tabular-nums text-shell-muted" dateTime={new Date(time).toISOString()}>
        {studioDateTime(time)}
      </time>
      {(coordinates.length > 0 || client || machine) && (
        <dl className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-shell-muted">
          {machine && <div className="flex gap-1"><dt>{studioName('machine')}</dt><dd>{machine}</dd></div>}
          {coordinates.map(([axis, value]) => (
            <div key={axis} className="flex gap-1">
              <dt>{axis}</dt>
              <dd className="tabular-nums">{coordinateValue(value)}</dd>
            </div>
          ))}
          {client && <div className="flex gap-1">
            <dt>{studioName('client')}</dt>
            <dd>{studioName(client.client)}</dd>
          </div>}
          {client?.label && <div className="flex min-w-0 gap-1">
            <dt>{studioName('label')}</dt>
            <dd className="min-w-0 break-words">{client.label}</dd>
          </div>}
        </dl>
      )}
      {job.earlier_definition && (
        <p className="text-xs font-medium text-shell-muted">{studioDefinitionStatus('earlier')}</p>
      )}
      <StudioRunReason job={job} />
      {job.status === 'running' && job.progress && <StudioRunProgress progress={job.progress} etaSeconds={job.eta_seconds} />}
      {job.status === 'collecting' && <StudioCollectionProgress progress={job.collection_progress} />}
      {failure && <p role="alert" className="text-xs text-shell-status-failure">{failure}</p>}
      <div className="flex flex-wrap gap-2">
        {jobAllowsCommand(job, 'stop') && (
          <ActivityCommand name="stop" job={job.job_id} onClick={() => void stop('checkpoint')} />
        )}
        {jobAllowsCommand(job, 'forceStop') && (
          <ActivityCommand name="forceStop" job={job.job_id} onClick={() => void stop('force')} />
        )}
        {jobAllowsCommand(job, 'resume') && (
          <ActivityCommand name="resume" job={job.job_id} onClick={() => void resume()} />
        )}
      </div>
    </li>
  );
}

function ActivityMachine({ machine }: { machine: StudioRemoteSession }) {
  const heading = useId();
  const actions = useStudioJobActions();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const opening = machine.opening_observations ?? [];
  const currentStep = opening.at(-1);
  async function close() {
    setError(undefined);
    setBusy(true);
    try {
      await closeStudioMachine(machine.id);
      actions.refresh();
    } catch (cause) {
      setError(studioRegionIssue('machine', cause).reason);
    } finally {
      setBusy(false);
    }
  }
  return <section aria-labelledby={heading} className="space-y-2 rounded-md border border-shell-divider px-3 py-2 text-sm">
    <header className="flex items-baseline justify-between gap-2">
      <h3 id={heading} className="font-medium">{machine.name}</h3>
      <span className="text-shell-muted">{studioMachineStatus(machine.state)}</span>
    </header>
    {machine.state === 'opening' && currentStep && <p className="text-shell-muted">{studioOpeningStep(currentStep.step)}</p>}
    {machine.opening_failure && <p className="text-shell-muted">{machine.opening_failure}</p>}
    <dl className="grid grid-cols-[auto_1fr] gap-x-3">
      {machine.region && <><dt>{studioName('region')}</dt><dd>{machine.region}</dd></>}
      <dt>{studioName('hourlyRate')}</dt><dd>{studioMoney(machine.hourly_microusd)}</dd>
      <dt>{studioName(machine.billed_cost_microusd != null ? 'billedCost' : 'estimatedCost')}</dt>
      <dd>{studioMoney(machine.billed_cost_microusd ?? machine.estimated_cost_microusd)}</dd>
      <dt>{studioName('costCeiling')}</dt><dd>{studioMoney(machine.ceiling_microusd)}</dd>
      <dt>{studioName('closesAt')}</dt><dd>{studioDateTime(machine.closes_at_ms)}</dd>
      {machine.idle_since_ms !== null && <><dt>{studioName('idleSince')}</dt><dd>{studioDateTime(machine.idle_since_ms)}</dd></>}
      {machine.state === 'unreachable' && machine.power_state && <>
        <dt>{studioName('power')}</dt><dd>{studioMachinePower(machine.power_state)}</dd>
      </>}
      {machine.provider_observed_at_ms != null && <>
        <dt>{studioName('providerChecked')}</dt><dd>{studioDateTime(machine.provider_observed_at_ms)}</dd>
      </>}
      <dt>{studioName('runs')}</dt><dd>{machine.job_ids.length}</dd>
    </dl>
    {opening.length > 0 && <Section level="group" name={studioName('openingHistory')} defaultOpen={false}>
      <ol className="list-disc ps-5">
        {opening.map(({ step, observed_at_ms }, index) => <li key={index}>
          {studioOpeningStep(step)} · {studioDateTime(observed_at_ms)}
        </li>)}
      </ol>
    </Section>}
    {error && <p role="alert">{error}</p>}
    <Button disabled={!machine.can_close || busy} onClick={() => void close()}>{studioName('close')}</Button>
  </section>;
}

function ActivityCommand({
  name,
  job,
  onClick,
}: {
  name: 'stop' | 'forceStop' | 'resume';
  /** The job the command acts on, which tells apart the commands of each row. */
  job: string;
  onClick: () => void;
}) {
  const command = bindStudioCommand(name, onClick, { keyboard: false, instance: job });
  return (
    <Button {...commandAttributes(command)} disabled={!command.enabled} onClick={command.invoke}>
      {studioName(name)}
    </Button>
  );
}

/** Dismiss: marks a settled run seen, which takes it off the list of runs to look at. */
function ActivityDismiss({ job, onClick }: { job: string; onClick: () => void }) {
  const command = bindStudioCommand('dismiss', onClick, { keyboard: false, instance: job });
  return <IconButton size="mark" label={command.label} icon={<X aria-hidden="true" />} {...commandAttributes(command)}
    disabled={!command.enabled} onClick={command.invoke} />;
}

interface StudioActivityPaneProps {
  projectNames: ReadonlyMap<string, string>;
  onOpenRun: (job: StudioActivityJob) => Promise<boolean>;
  onClose: () => void;
  standalone?: boolean;
  resultState?: ActivityResultState;
}

export function StudioActivityPane(props: StudioActivityPaneProps) {
  return (
    <StudioRegionBoundary name="activity">
      <ActivityContent {...props} />
    </StudioRegionBoundary>
  );
}

/**
 * Activity's runs grouped by project, newest first: the window's own project first, then the others
 * in the order of their most recent activity.
 */
export function activityGroups(visible: readonly StudioActivityJob[], ownProject: string | null) {
  const runs = new Map<string, StudioActivityJob[]>();
  for (const job of [...visible].sort((a, b) => activityTime(b) - activityTime(a) || a.job_id.localeCompare(b.job_id))) {
    runs.set(job.project_id, [...(runs.get(job.project_id) ?? []), job]);
  }
  const rank = (projectId: string) => projectId === ownProject ? Infinity : activityTime(runs.get(projectId)![0]);
  return [...runs]
    .sort(([a], [b]) => rank(b) - rank(a))
    .map(([projectId, group]) => ({ projectId, runs: group }));
}

function ActivityProject({ name, children }: PropsWithChildren<{ name: string }>) {
  const heading = useId();
  return (
    <li>
      <section aria-labelledby={heading} className="space-y-2">
        <h3 id={heading} className="truncate text-xs font-medium text-shell-muted">{name}</h3>
        <ul className="space-y-2">{children}</ul>
      </section>
    </li>
  );
}

function ActivityContent({ projectNames, onOpenRun, onClose, standalone, resultState }: StudioActivityPaneProps) {
  const snapshot = useStudioJobs();
  const unreadable = useStudioUnreadableJobs();
  const actions = useStudioJobActions();
  const [selection, setSelection] = useState<string | null>(null);
  const jobs = snapshot.data?.jobs ?? [];
  const selected = jobs.find((job) => job.job_id === selection);
  const ownProject = useStudioJobsProject();
  const visible = jobs.filter((job) => isLive(job) || !job.seen || job === selected);
  const groups = activityGroups(visible, ownProject);
  const machine = snapshot.data?.sessions.find((session) => session.id === 'local');
  const remoteMachines = snapshot.data?.sessions.filter((session): session is StudioRemoteSession => 'name' in session) ?? [];
  const machineByJob = new Map(remoteMachines.flatMap((session) => session.job_ids.map((id) => [id, session.name] as const)));
  const machineField = machine && (
    <dl className="flex justify-between rounded-md border border-shell-divider px-3 py-2 text-sm">
      <dt className="font-medium">{studioName('thisMac')}</dt>
      <dd className="text-shell-muted">{studioMachineStatus(machine.state ?? 'open')}</dd>
    </dl>
  );
  const state: StudioRegionState = snapshot.isError
    ? { kind: 'failed', issue: studioRegionIssue('activity', snapshot.error), retry: actions.refresh }
    : snapshot.isPending
      ? { kind: 'loading' }
      : unreadable.length > 0
        ? {
          kind: 'partial',
          unread: unreadable.map((record) => ({
            key: record.job_id, reason: studioReason('unreadable', record.label ?? record.job_id),
          })),
        }
        : visible.length > 0
          ? { kind: 'result', missing: [] }
          : { kind: 'empty' };

  return (
    <StudioRegion
      state={state}
      skeleton={<div className="h-20 motion-safe:animate-pulse rounded-md bg-shell-divider" />}
      empty={(
        <>
          {machineField}
          {remoteMachines.map((session) => <ActivityMachine key={session.id} machine={session} />)}
          <ul aria-label={studioName('runs')} className="min-h-10 rounded-md border border-dashed border-shell-divider" />
        </>
      )}
    >
      {machineField}
      {remoteMachines.map((session) => <ActivityMachine key={session.id} machine={session} />)}
      <ul aria-label={studioName('runs')} className="space-y-3">
        {groups.map(({ projectId, runs }) => (
          <ActivityProject key={projectId} name={projectNames.get(projectId) ?? projectId}>
            {runs.map((job) => (
              <ActivityRun
                key={job.job_id}
                job={job}
                selected={job === selected}
                machine={machineByJob.get(job.job_id)}
                resultState={resultState?.(job)}
                onClose={onClose}
                onOpen={async () => {
                  const opened = await onOpenRun(job);
                  if (!opened) setSelection(job.job_id);
                  return opened;
                }}
              />
            ))}
          </ActivityProject>
        ))}
      </ul>
      {selected && (standalone
        ? <StudioJobsProvider projectId={selected.project_id}><StudioRunInspector job={selected} /></StudioJobsProvider>
        : <StudioRunInspector job={selected} />)}
    </StudioRegion>
  );
}

export function useStudioActivityCount() {
  const snapshot = useStudioJobs();
  return snapshot.data?.jobs.filter(isLive).length ?? 0;
}
