import type { Dispatch, SetStateAction } from 'react';
import { useStore } from 'zustand';
import { Shelf } from '@/components/timeline/Shelf';
import { ClockGroup } from '@/components/timeline/ClockGroup';
import { defaultTrialClock, shownEntry, usePlayhead, type PlayheadStore, type TrialOffer } from '@/components/timeline/trialClock';
import { studioRegionIssue, StudioRegion } from '@/components/activity/StudioRegion';
import type { StudioSavedEvaluationRow } from '@/generated/studioContracts';
import { selectElement, type ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import type { ProjectWindowState } from '@/features/projectWindow/projectWindowState';
import { recordedTrials, useSavedEvaluationScene, type StoredModel } from './managedProjectDetails';
import type { TrialView } from './StudioTrialAuthoring';
import { nominalRows, trialRows } from './task/shelfRows';
import { shelfTask, taskDocumentOf, useTaskSamples } from './task/taskTrials';

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
 * samples, and the trials the selected run recorded; the shelf of that trial's rows; and the recorded trial's replay.
 */
export function useTrialClock({
  windowState, setWindowState, projectId, revision, view, taskDocument, stored, evaluation, sceneVisible, elements, playhead,
}: TrialClockInputs) {
  const clock = windowState.clock ?? defaultTrialClock;
  const recorded = recordedTrials(evaluation);
  const task = view && taskDocumentOf(view);
  const offer: TrialOffer = {
    nominal: !!task, samples: !!task && !!stored && !view.reads_model, recorded: recorded.ids.length,
  };
  const shown = shownEntry(clock, offer);
  const replay = useSavedEvaluationScene(projectId, revision, evaluation, sceneVisible,
    shown?.kind === 'recorded' ? shown.index : undefined, playhead);
  const samples = useTaskSamples(view, stored?.value, clock.seed, clock.count, shown?.kind === 'sample');
  const sample = shown?.kind === 'sample' ? samples.data?.[shown.index] : undefined;
  const drawn = view && taskDocument ? shelfTask(view, taskDocument) : undefined;
  const rows = !drawn || !shown ? undefined
    : shown.kind === 'nominal' ? nominalRows(drawn)
      : shown.kind === 'sample' && sample ? trialRows(drawn, sample, view!.period_seconds) : undefined;
  const [time, setTime] = usePlayhead(playhead, JSON.stringify(['task', taskDocument, shown, clock.seed]));
  const selected = useStore(elements, (current) => current.selection);
  const change = (next: typeof clock) => setWindowState((current) => ({ ...current, clock: next }));
  const shelfShown = !!drawn && shown?.kind !== 'recorded';
  return {
    replay,
    sample,
    clock: shown && <ClockGroup offer={offer} clock={clock} shown={shown} recordedLabels={recorded.labels} onChange={change}
      transport={shown.kind === 'recorded' ? replay.transport : undefined} shelf={shelfShown} />,
    shelf: shelfShown && clock.shelf && (rows
      ? <Shelf rows={rows} duration={drawn.duration} stepSeconds={view!.period_seconds} time={time} onTimeChange={setTime}
        selected={selected} onSelect={(address) => selectElement(elements, address)} />
      : <StudioRegion state={samples.error
        ? { kind: 'failed', issue: studioRegionIssue('shelf', samples.error), retry: () => { void samples.refetch(); } }
        : { kind: 'loading' }} />),
  };
}
