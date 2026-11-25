import type { Dispatch, SetStateAction } from 'react';
import { useStore } from 'zustand';
import { Shelf, type ShelfRow } from '@/components/timeline/Shelf';
import { ClockGroup, type ClockTimeline } from '@/components/timeline/ClockGroup';
import { useTimeAxisStore } from '@/components/timeline/timeAxis';
import {
  defaultTrialClock, meanSelection, sampleCount, shelfOpen, shownPick, usePlayhead,
  type PlayheadStore, type TrialClock, type TrialOffer, type TrialPick,
} from '@/components/timeline/trialClock';
import { studioName } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { studioRegionIssue, StudioRegion } from '@/components/activity/StudioRegion';
import type { SceneTrial } from '@/components/scene/trialScene';
import { frameAt } from '@/components/timeline/trialClock';
import type {
  Problem, StudioSavedEvaluationRow, TaskDocument, TrialMean, TrialRecord,
} from '@/generated/studioContracts';
import { selectElement, type ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import { showProjectWindowView, type ProjectWindowState, type ProjectWindowView } from '@/features/projectWindow/projectWindowState';
import { clockRun, type FigureClock } from './figureClock';
import { recordedTrials, useSavedEvaluationScene, type StoredModel } from './managedProjectDetails';
import type { TrialView } from './StudioTrialAuthoring';
import { batchRows, meanRows, nominalRows, trialRows, trialTermRows, type ShelfTask } from './task/shelfRows';
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
  /** Whether a Scene is shown, which draws the samples a trial may be beside the one shown. */
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
 * recorded trial) or the aggregate of the samples or of the recorded trials; the Problems that keep it from being
 * drawn; or nothing while it is read. With it, what the Scene draws: the picked trial and the trials beside it.
 */
function useShown(pick: TrialPick | undefined, clock: TrialClock, offer: TrialOffer, view: TrialView | undefined,
  stored: StoredModel | undefined, replay: ReturnType<typeof useSavedEvaluationScene>, sceneVisible: boolean) {
  const count = sampleCount(clock, offer);
  const recorded = replay.read.kind === 'read' ? replay.read : undefined;
  const sampled = pick?.trials === 'samples';
  const nominal = useNominalTrial(view, stored?.value, pick?.trials === 'nominal');
  const possibleShown = sceneVisible && count > 0 && (pick?.trials === 'nominal' || pick?.showing.kind === 'all');
  const samples = useTaskSamples(view, stored?.value, clock.seed, count, sampled && pick.showing.kind === 'one' || possibleShown);
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
      case 'recorded':
        switch (showing.kind) {
          case 'one': return recorded && { kind: 'one', index: showing.index, count: offer.recorded, record: recorded.picked };
          case 'all': return replay.aggregate.data && { kind: 'all', count: offer.recorded, aggregate: replay.aggregate.data };
          case 'mean': return undefined;
        }
      case 'samples':
        switch (showing.kind) {
          case 'one': return one(samples.data, showing.index, count);
          case 'mean': return undefined;
          case 'all': {
            const drawn = batch.data;
            return !drawn || drawn.kind === 'problems' ? drawn : { kind: 'all', count, aggregate: drawn };
          }
        }
    }
  };
  const query = pick?.trials === 'nominal' ? nominal : pick?.trials === 'samples' ? pick.showing.kind === 'all' ? batch : samples
    : pick?.showing.kind === 'all' ? replay.aggregate : undefined;
  const shown = read();
  /** Whether what the Scene draws of the pick (its trial, and the samples beside the nominal one) is still being read. */
  const reading = !!pick && (pick.trials === 'recorded' ? replay.read.kind === 'loading'
    : pick.trials === 'nominal' && nominal.isLoading || (sampled || possibleShown) && samples.isLoading);
  const drawnSamples = possibleShown && Array.isArray(samples.data) ? samples.data : noRecords;
  const possible = pick?.trials === 'recorded' ? recorded && pick.showing.kind !== 'one' ? recorded.records : noRecords : drawnSamples;
  const picked = pick?.showing.kind === 'mean' ? undefined : pick?.trials === 'recorded' ? recorded?.picked
    : pick?.showing.kind === 'all' ? drawnSamples[pick.showing.index] : shown?.kind === 'one' ? shown.record : undefined;
  return { shown, query, possible, picked, reading };
}

const noRecords: readonly TrialRecord[] = [];

/**
 * The shelf's rows for what is shown: the nominal trial as its definition draws it, with each drawn length's spread, and
 * its terms as its record has them once it is read; one trial as its record has it; the aggregate's shares; under Mean,
 * the recorded trials' shares on time from Align's epoch.
 */
function rowsOf(task: ShelfTask, pick: TrialPick, shown: TrialShown | TrialsUnavailable | undefined, stepSeconds: number,
  mean: { value: TrialMean; records: readonly TrialRecord[] } | undefined): ShelfRow[] | undefined {
  if (pick.showing.kind === 'mean') return mean && meanRows(task, mean.value, mean.records, stepSeconds);
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

/**
 * Under Mean, the time from Align's epoch over which some trial has a record, which the figures draw every trial over,
 * and the times every trial's samples are at, on that axis.
 */
function meanTimeline(mean: TrialMean, stateTime: readonly number[]): Omit<ClockTimeline, 'bands'> | undefined {
  const alignments = mean.conditions.flatMap((condition) => Object.values(condition.trials));
  if (!alignments.length) return undefined;
  const start = Math.min(...alignments.map(({ span }) => span.start));
  const end = Math.max(...alignments.map(({ span }) => span.end));
  const times = new Set(alignments.flatMap(({ origin, span }) => stateTime.map((time) => Math.round((time - origin) * 1e9) / 1e9)
    .filter((time) => time >= span.start - 1e-9 && time <= span.end + 1e-9)));
  return { start, duration: end - start, frameTimes: [...times].sort((a, b) => a - b) };
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
  const replay = useSavedEvaluationScene(projectId, revision, evaluation,
    pick?.trials === 'recorded' ? pick.showing : undefined, { by: clock.mean.by && [...clock.mean.by], align: clock.mean.align }, playhead);
  const averaging = pick?.showing.kind === 'mean';
  const mean = averaging ? replay.mean : undefined;
  const recordedTrial = replay.read.kind === 'read' ? replay.read : undefined;
  const { shown, query, possible, picked, reading } = useShown(pick, clock, offer, view, stored, replay, sceneVisible);
  const drawn = view && taskDocument ? shelfTask(view, taskDocument, objectiveDocument) : undefined;
  const [taskTime, setTaskTime] = usePlayhead(playhead, JSON.stringify(['task', taskDocument]));
  const [heldTime, setTime] = pick?.trials === 'recorded' ? [replay.time, replay.setTime] : [taskTime, setTaskTime];
  const stateTime = pick?.trials === 'recorded' ? recordedTrial?.stateTime : undefined;
  const replayTimeline = averaging ? mean && recordedTrial ? meanTimeline(mean, recordedTrial.stateTime) ?? null : null
    : stateTime && stateTime.length > 1 && stateTime[stateTime.length - 1] > 0
      ? { start: 0, duration: stateTime[stateTime.length - 1], frameTimes: [...stateTime] } : null;
  /** A recorded trial's step is its run's; any other trial's is the step's Model's. */
  const step = replayTimeline && stateTime ? stateTime[1] : period;
  const averaged = mean && recordedTrial && { value: mean, records: recordedTrial.records };
  const rows = pick && (drawn && step != null ? rowsOf(drawn, pick, shown, step, averaged) : shown && []);
  const epochs = rows?.find((row) => row.kind === 'segments');
  const timeline: ClockTimeline = replayTimeline
    ? { ...replayTimeline, bands: [] }
    : {
      start: 0,
      duration: drawn?.duration ?? 0,
      frameTimes: view?.n_steps && period != null ? frames(view.n_steps, period) : [],
      bands: epochs?.kind === 'segments' ? epochs.segments : [],
    };
  /** The playhead is kept across trials of one time axis, so it is brought within the shown trial's time. */
  const time = timeline.duration > 0 ? Math.min(Math.max(heldTime, timeline.start), timeline.start + timeline.duration) : heldTime;
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
  const trails = pick?.trials === 'recorded' && <Button variant="ghost" pressed={clock.trails}
    onClick={() => change({ ...clock, trails: !clock.trails })}>{studioName('trails')}</Button>;
  const scene: SceneTrial = {
    picked,
    possible,
    reading,
    trails: clock.trails,
    frame: frameAt(timeline.frameTimes, time),
    task: task && taskDocument ? { document: taskDocument, value: task } : undefined,
    ...mean && recordedTrial && { mean: { value: mean, time, stateTime: recordedTrial.stateTime } },
    ...pick?.trials === 'recorded' && replay.read.kind === 'failed' && { failure: replay.read },
  };
  const recordedPick = pick?.trials === 'recorded' ? pick : undefined;
  const pickedTrial = recordedPick && recorded.ids[recordedPick.showing.index];
  const pickedKind = recordedPick?.showing.kind;
  /** Under Mean the time is measured from Align's epoch and no single trial is drawn. */
  const figure: FigureClock = {
    time, setTime, epoch: averaging ? clock.mean.align : null, run: recordedPick && clockRun(evaluation),
    picked: pickedTrial === undefined || !pickedKind || pickedKind === 'mean' ? undefined : { trial: pickedTrial, kind: pickedKind },
    pickTrial: (trial) => {
      const index = recorded.ids.indexOf(trial);
      const kind = pickedKind === 'all' ? 'all' : 'one';
      if (index >= 0) change({ ...clock, picked: { trials: 'recorded', showing: { kind, index } } });
    },
  };
  return {
    replay,
    /** What a figure draws from the clock. */
    figure,
    /** What the Scene draws: the shown trial at the playhead's frame, the trials it may be, and the task. */
    scene,
    /** What the Variables table and the Inspector show, when it can be drawn. */
    trial: shown?.kind === 'problems' ? undefined : shown,
    /** The Mean every figure draws, while Mean is picked. */
    mean: meanSelection(clock),
    clock: pick && <ClockGroup offer={offer} clock={clock} shown={pick} recordedLabels={recorded.labels} recordedOutOfDate={recorded.outOfDate} meanOptions={replay.meanOptions} onChange={change}
      timeline={timeline} time={time} onTimeChange={setTime} axis={axis} shelfOpen={open}
      onShelfChange={(next) => shelfView && change({ ...clock, shelf: { ...clock.shelf, [shelfView]: next } })} />,
    shelf: pick && open && timeline.duration > 0 && (rows
      ? <Shelf rows={rows} start={timeline.start} duration={timeline.duration} stepSeconds={replayTimeline ? undefined : period ?? undefined} time={time}
        onTimeChange={setTime} axis={axis} selected={selected} onSelect={(address) => selectElement(elements, address)}
        unavailable={unavailable || undefined}
        trailing={trails || undefined} />
      : <StudioRegion state={query?.error
        ? { kind: 'failed', issue: studioRegionIssue('shelf', query.error), retry: () => { void query.refetch(); } }
        : { kind: 'loading' }} />),
  };
}
