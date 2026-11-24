import type { Dispatch, SetStateAction } from 'react';
import { useStore } from 'zustand';
import { Shelf } from '@/components/timeline/Shelf';
import { ClockGroup } from '@/components/timeline/ClockGroup';
import {
  defaultTrialClock, sampleCount, shownEntry, showsAll, usePlayhead, type PlayheadStore, type TrialOffer,
} from '@/components/timeline/trialClock';
import { studioName } from '@/components/activity/studioVocabulary';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { studioRegionIssue, StudioRegion } from '@/components/activity/StudioRegion';
import type { StudioSavedEvaluationRow } from '@/generated/studioContracts';
import { selectElement, type ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import type { ProjectWindowState } from '@/features/projectWindow/projectWindowState';
import { recordedTrials, useSavedEvaluationScene, type StoredModel } from './managedProjectDetails';
import type { TrialView } from './StudioTrialAuthoring';
import { batchRows, nominalRows, trialRows } from './task/shelfRows';
import { shelfTask, taskDocumentOf, useTaskBatch, useTaskSamples } from './task/taskTrials';
import type { TrialShown } from './task/trialSample';

interface TrialClockInputs {
  readonly windowState: ProjectWindowState;
  readonly setWindowState: Dispatch<SetStateAction<ProjectWindowState>>;
  readonly projectId: string;
  readonly revision: string;
  /** The selected step's Trial authoring, whose task the clock samples. */
  readonly view: TrialView | undefined;
  /** The task document the view reads, whose elements the shelf's rows select. */
  readonly taskDocument: string | undefined;
  readonly stored: StoredModel | undefined;
  readonly evaluation: StudioSavedEvaluationRow | undefined;
  readonly sceneVisible: boolean;
  readonly elements: ElementSelectionStore;
  readonly playhead: PlayheadStore;
}

/**
 * The bottom bar's clock for the selection: which trial its picker shows, of the task document's nominal trial, its
 * samples, and the trials the selected run recorded, or all the samples at once; the shelf of that trial's rows, or of
 * the samples' aggregate; and the recorded trial's replay.
 */
export function useTrialClock({
  windowState, setWindowState, projectId, revision, view, taskDocument, stored, evaluation, sceneVisible, elements, playhead,
}: TrialClockInputs) {
  const clock = windowState.clock ?? defaultTrialClock;
  const recorded = recordedTrials(evaluation);
  const task = view && taskDocumentOf(view);
  const offer: TrialOffer = {
    nominal: !!task, samples: task && stored && !view.reads_model ? view.batch_size ?? 0 : 0, recorded: recorded.ids.length,
  };
  const shown = shownEntry(clock, offer);
  const all = showsAll(clock, offer);
  const count = sampleCount(clock, offer);
  const replay = useSavedEvaluationScene(projectId, revision, evaluation, sceneVisible,
    shown?.kind === 'recorded' ? shown.index : undefined, playhead);
  const samples = useTaskSamples(view, stored?.value, clock.seed, count, !all && shown?.kind === 'sample');
  const batch = useTaskBatch(view, stored?.value, clock.seed, count, all);
  const sample = !all && shown?.kind === 'sample' ? samples.data?.[shown.index] : undefined;
  const trial: TrialShown | undefined = all ? batch.data && { kind: 'batch', batch: batch.data }
    : sample && shown?.kind === 'sample' ? { kind: 'trial', index: shown.index, count, trial: sample } : undefined;
  const drawn = view && taskDocument ? shelfTask(view, taskDocument) : undefined;
  const rows = !drawn || !shown ? undefined
    : trial?.kind === 'batch' ? batchRows(drawn, trial.batch, view!.period_seconds)
      : all ? undefined
        : shown.kind === 'nominal' ? nominalRows(drawn)
          : sample ? trialRows(drawn, sample, view!.period_seconds) : undefined;
  const [time, setTime] = usePlayhead(playhead, JSON.stringify(['task', taskDocument, all || shown, clock.seed]));
  const selected = useStore(elements, (current) => current.selection);
  const change = (next: typeof clock) => setWindowState((current) => ({ ...current, clock: next }));
  const shelfShown = !!drawn && (all || shown?.kind !== 'recorded');
  const failed = (all ? batch : samples).error;
  return {
    replay,
    /** The sampled trial, or the samples' aggregate, that the Variables table and the Inspector show. */
    trial,
    clock: shown && <ClockGroup offer={offer} clock={clock} shown={shown} recordedLabels={recorded.labels} onChange={change}
      transport={!all && shown.kind === 'recorded' ? replay.transport : undefined} shelf={shelfShown} />,
    shelf: shelfShown && clock.shelf && (rows
      ? <Shelf rows={rows} duration={drawn.duration} stepSeconds={view!.period_seconds} time={time} onTimeChange={setTime}
        selected={selected} onSelect={(address) => selectElement(elements, address)}
        trailing={offer.samples > 0 && <SegmentedControl aria-label={studioName('trials')} value={all ? 'all' : 'one'}
          items={[{ value: 'one', label: studioName('oneTrial') }, { value: 'all', label: studioName('allTrials') }]}
          onValueChange={(value) => change({ ...clock, all: value === 'all' })} />} />
      : <StudioRegion state={failed
        ? { kind: 'failed', issue: studioRegionIssue('shelf', failed), retry: () => { void (all ? batch : samples).refetch(); } }
        : { kind: 'loading' }} />),
  };
}
