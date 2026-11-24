import type { Dispatch, SetStateAction } from 'react';
import { useStore } from 'zustand';
import { Shelf, type ShelfRow } from '@/components/timeline/Shelf';
import { ClockGroup, type ClockTimeline } from '@/components/timeline/ClockGroup';
import { useTimeAxisStore } from '@/components/timeline/timeAxis';
import {
  defaultTrialClock, sampleCount, shelfOpen, shownPick, usePlayhead, type PlayheadStore, type TrialClock, type TrialOffer,
  type TrialPick,
} from '@/components/timeline/trialClock';
import { studioName } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { studioRegionIssue, StudioRegion } from '@/components/activity/StudioRegion';
import { sceneTimeline } from '@/components/scene/sceneView';
import type { Problem, StudioSavedEvaluationRow, TaskDocument, TrialRecord } from '@/generated/studioContracts';
import { selectElement, type ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import { showProjectWindowView, type ProjectWindowState, type ProjectWindowView } from '@/features/projectWindow/projectWindowState';
import { recordedTrials, useSavedEvaluationScene, type StoredModel } from './managedProjectDetails';
import type { TrialView } from './StudioTrialAuthoring';
import { batchRows, nominalRows, trialRows, trialTermRows, type ShelfTask } from './task/shelfRows';
import { shelfTask, taskDocumentOf, useNominalTrial, useTaskBatch, useTaskSamples } from './task/taskTrials';
import type { TrialShown, TrialsUnavailable } from './task/trialSample';

interface TrialClockInputs {
  readonly windowState: ProjectWindowState;
  readonly setWindowState: Dispatch<SetStateAction<ProjectWindowState>>;
  readonly projectId: string;
  readonly revision: string;
  /** The selected step's Trial authoring, whose task the clock samples. */
  readonly view: TrialView | undefined;
  /** The task document the view reads, whose elements the shelf's rows select. */
  readonly taskDocument: string | undefined;
  /** The Objective document the view reads, whose terms the shelf's term rows select. */
  readonly objectiveDocument: string | undefined;
  readonly stored: StoredModel | undefined;
  readonly evaluation: StudioSavedEvaluationRow | undefined;
  readonly sceneVisible: boolean;
  /** The view the shelf's open state is kept for: the main slot's. */
  readonly shelfView: ProjectWindowView | undefined;
  readonly elements: ElementSelectionStore;
  readonly playhead: PlayheadStore;
}

/** The element of the task document a Problem is about, as the Task view names it: `variable:<id>`, `epoch:<id>`, `signal:<id>`. */
function problemElement(task: TaskDocument, problem: Problem): string | undefined {
  if (problem.variable) return `variable:${problem.variable}`;
  const [group, index] = problem.path ?? [];
  if (typeof index !== 'number') return undefined;
  if (group === 'variables') return task.variables[index] && `variable:${task.variables[index].id}`;
  if (group === 'epochs') return task.epochs[index] && `epoch:${task.epochs[index].id}`;
  if (group === 'signals') return task.signals[index] && `signal:${task.signals[index].id}`;
  return undefined;
}

/**
 * What the clock shows as views draw it, of what is picked: one trial's record (the nominal trial, a sample, or a
 * recorded trial) or the samples' aggregate; the Problems that keep it from being drawn; or nothing while it is read.
 */
function useShown(pick: TrialPick | undefined, clock: TrialClock, offer: TrialOffer, view: TrialView | undefined,
  stored: StoredModel | undefined, recorded: TrialRecord | undefined) {
  const count = sampleCount(clock, offer);
  const sampled = pick?.trials === 'samples';
  const nominal = useNominalTrial(view, stored?.value, pick?.trials === 'nominal');
  const samples = useTaskSamples(view, stored?.value, clock.seed, count, sampled && pick.showing.kind === 'one');
  const batch = useTaskBatch(view, stored?.value, clock.seed, count, sampled && pick.showing.kind === 'all');
  const one = (drawn: TrialRecord[] | TrialsUnavailable | undefined, index: number, of: number): TrialShown | TrialsUnavailable | undefined => {
    if (!drawn || !Array.isArray(drawn)) return drawn;
    const record = drawn[index];
    return record && { kind: 'one', index, count: of, record };
  };
  const read = (): TrialShown | TrialsUnavailable | undefined => {
    if (!pick) return undefined;
    const { showing } = pick;
    switch (pick.trials) {
      case 'nominal': return one(nominal.data, 0, 1);
      case 'recorded': return showing.kind === 'one' && recorded
        ? { kind: 'one', index: showing.index, count: offer.recorded, record: recorded } : undefined;
      case 'samples':
        switch (showing.kind) {
          case 'one': return one(samples.data, showing.index, count);
          case 'all': {
            const drawn = batch.data;
            return !drawn || drawn.kind === 'problems' ? drawn : { kind: 'all', count, aggregate: drawn };
          }
        }
    }
  };
  const query = pick?.trials === 'nominal' ? nominal : pick?.trials === 'samples' ? pick.showing.kind === 'all' ? batch : samples
    : undefined;
  return { shown: read(), query };
}

/**
 * The shelf's rows for what is shown: the nominal trial as its definition draws it, with each drawn length's spread, and
 * its terms as its record has them once it is read; one trial as its record has it; the aggregate's shares.
 */
function rowsOf(task: ShelfTask, pick: TrialPick, shown: TrialShown | TrialsUnavailable | undefined, stepSeconds: number): ShelfRow[] | undefined {
  if (pick.trials === 'nominal') {
    return [...nominalRows(task), ...shown?.kind === 'one' ? trialTermRows(task, shown.record, stepSeconds) : []];
  }
  if (!shown) return undefined;
  if (shown.kind === 'problems') return nominalRows(task);
  switch (shown.kind) {
    case 'one': return trialRows(task, shown.record, stepSeconds);
    case 'all': return batchRows(task, shown.aggregate, stepSeconds);
  }
}

const frames = (steps: number, stepSeconds: number) => Array.from({ length: steps + 1 }, (_, step) => step * stepSeconds);

/**
 * The window's clock for the selection: what it shows (P37's value, of the task document's nominal trial, its
 * samples and the trials the selected run recorded, one at a time or all at once), the records every view draws, and
 * the time; with the bottom bar's clock group and the shelf on that time.
 */
export function useTrialClock({
  windowState, setWindowState, projectId, revision, view, taskDocument, objectiveDocument, stored, evaluation, sceneVisible,
  shelfView, elements, playhead,
}: TrialClockInputs) {
  const clock = windowState.clock ?? defaultTrialClock;
  const recorded = recordedTrials(evaluation);
  const period = view?.period_seconds;
  const task = view && taskDocumentOf(view);
  const offer: TrialOffer = {
    nominal: !!task && period != null, samples: task && period != null && stored && !view.reads_model ? view.batch_size ?? 0 : 0, recorded: recorded.ids.length,
  };
  const pick = shownPick(clock, offer);
  const index = pick?.showing.kind === 'one' ? pick.showing.index : undefined;
  const replay = useSavedEvaluationScene(projectId, revision, evaluation, sceneVisible,
    pick?.trials === 'recorded' ? index : undefined, playhead);
  const recordedTrial = replay.view.kind === 'replay' ? replay.view.trial : undefined;
  const { shown, query } = useShown(pick, clock, offer, view, stored, recordedTrial);
  const drawn = view && taskDocument ? shelfTask(view, taskDocument, objectiveDocument) : undefined;
  const [taskTime, setTaskTime] = usePlayhead(playhead, JSON.stringify(['task', taskDocument, pick, clock.seed]));
  const [time, setTime] = pick?.trials === 'recorded' ? [replay.time, replay.setTime] : [taskTime, setTaskTime];
  const replayTimeline = pick?.trials === 'recorded' ? sceneTimeline(replay.view) : null;
  /** A recorded trial's step is its run's; any other trial's is the step's Model's. */
  const step = replayTimeline ? replayTimeline.frameTimes[1] : period;
  const rows = pick && (drawn && step != null ? rowsOf(drawn, pick, shown, step) : shown && []);
  const epochs = rows?.find((row) => row.kind === 'segments');
  const timeline: ClockTimeline = replayTimeline
    ? { duration: replayTimeline.duration, frameTimes: replayTimeline.frameTimes, bands: [] }
    : {
      duration: drawn?.duration ?? 0,
      frameTimes: view?.n_steps && period != null ? frames(view.n_steps, period) : [],
      bands: epochs?.kind === 'segments' ? epochs.segments : [],
    };
  const selected = useStore(elements, (current) => current.selection);
  const axis = useTimeAxisStore();
  const change = (next: TrialClock) => setWindowState((current) => ({ ...current, clock: next }));
  const open = !!shelfView && shelfOpen(clock, shelfView);
  const choose = (problem: Problem) => {
    const element = task && taskDocument && problemElement(task, problem);
    if (!element) return;
    setWindowState((current) => showProjectWindowView(current, 'task', 'primary'));
    selectElement(elements, { document: taskDocument!, element: { kind: 'named', id: element } });
  };
  const unavailable = shown?.kind === 'problems' && shown.problems.map((problem, at) => task && problemElement(task, problem)
    ? <Button key={at} variant="bare" size="content" align="start" className="block underline-offset-2 hover:underline"
      onClick={() => choose(problem)}>{problem.reason}</Button>
    : <span key={at} className="block">{problem.reason}</span>);
  const showing = pick?.showing.kind === 'all' ? 'all' : 'one';
  return {
    replay,
    /** What the Variables table and the Inspector show, when it can be drawn. */
    trial: shown?.kind === 'problems' ? undefined : shown,
    clock: pick && <ClockGroup offer={offer} clock={clock} shown={pick} recordedLabels={recorded.labels} onChange={change}
      timeline={timeline} time={time} onTimeChange={setTime} axis={axis} shelfOpen={open}
      onShelfChange={(next) => shelfView && change({ ...clock, shelf: { ...clock.shelf, [shelfView]: next } })} />,
    shelf: pick && open && timeline.duration > 0 && (rows
      ? <Shelf rows={rows} duration={timeline.duration} stepSeconds={replayTimeline ? undefined : period ?? undefined} time={time}
        onTimeChange={setTime} axis={axis} selected={selected} onSelect={(address) => selectElement(elements, address)}
        unavailable={unavailable || undefined}
        trailing={offer.samples > 0 && <SegmentedControl aria-label={studioName('trials')} value={showing}
          items={[{ value: 'one', label: studioName('oneTrial') }, { value: 'all', label: studioName('allTrials') }]}
          onValueChange={(value) => change({
            ...clock, picked: { trials: 'samples', showing: value === 'all' ? { kind: 'all' } : { kind: 'one', index: 0 } },
          })} />} />
      : <StudioRegion state={query?.error
        ? { kind: 'failed', issue: studioRegionIssue('shelf', query.error), retry: () => { void query.refetch(); } }
        : { kind: 'loading' }} />),
  };
}
