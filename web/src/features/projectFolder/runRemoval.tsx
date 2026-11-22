import { createContext, useCallback, useContext, useState, useSyncExternalStore, type ReactNode } from 'react';
import type {
  StudioProjectFilePatch, StudioProjectWindowMapRow, StudioSavedTrainKey, StudioSavedTrainRuns, AttemptProvenance,
} from '@/generated/studioContracts';
import { reviewRemovedRuns } from '@/api/studioAuthoring';
import { studioFormat, studioName } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { Dialog } from '@/components/ui/Dialog';

const thresholdKey = 'feedbax.studio.removal-threshold';
const thresholdEvent = 'feedbax-removal-threshold';
const defaultThreshold = 30;

/** Seconds a removed run's result must have taken for its removal to be confirmed. Train results always are. */
export function removalThreshold(view: Pick<Window, 'localStorage'> = window) {
  const stored = view.localStorage.getItem(thresholdKey);
  const seconds = stored === null ? NaN : Number(stored);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : defaultThreshold;
}

export function setRemovalThreshold(seconds: number, view: Window = window) {
  view.localStorage.setItem(thresholdKey, String(seconds));
  view.dispatchEvent(new Event(thresholdEvent));
}

function subscribe(listener: () => void) {
  window.addEventListener(thresholdEvent, listener);
  window.addEventListener('storage', listener);
  return () => {
    window.removeEventListener(thresholdEvent, listener);
    window.removeEventListener('storage', listener);
  };
}

export function useRemovalThreshold() {
  return useSyncExternalStore(subscribe, () => removalThreshold());
}

/** Selects the window-map rows a command removes. */
export type RemovedRuns = (row: StudioProjectWindowMapRow) => boolean;

/** The rows of the sweep values an edit removes (`removed_values` of a field edit), downstream rows included. */
export function sweepValueRuns(removed: Readonly<Record<string, readonly string[]>> | undefined): RemovedRuns | undefined {
  const entries = Object.entries(removed ?? {}).filter(([, values]) => values.length);
  return entries.length ? (row) => entries.some(([axis, values]) => {
    const value = row.coordinates[axis]?.value_id;
    return value !== undefined && values.includes(value);
  }) : undefined;
}

const sameKey = (a: StudioSavedTrainKey, b: StudioSavedTrainKey) =>
  a.variant === b.variant && a.step === b.step && a.row === b.row;

function seconds(provenance: AttemptProvenance | null | undefined) {
  return provenance ? (Date.parse(provenance.ended_at) - Date.parse(provenance.started_at)) / 1000 : 0;
}

/** The removed runs, and whether any holds a Train result or a result that took at least `threshold` seconds. */
export function removalCost(plan: StudioSavedTrainRuns, removes: RemovedRuns, threshold: number) {
  const rows = plan.window_map?.rows.filter(removes) ?? [];
  const training = rows.map(({ key }) => plan.rows.find((row) => sameKey(row.key, key)));
  const costly = rows.some((row, index) => {
    if (training[index]?.result) return true;
    const evaluation = plan!.window_map!.evaluations?.find(({ key }) => sameKey(key, row.key))?.result;
    return Math.max(seconds(row.light?.result?.provenance), seconds(evaluation?.provenance)) >= threshold;
  });
  return { count: rows.length, training: rows.length > 0 && training.every(Boolean), costly };
}

/** The rows of the runs the files would remove, as the server's Plan reads the edited project. */
export function filesRemoval(
  projectId: string, revision: string, files: readonly StudioProjectFilePatch[], signal: AbortSignal,
): RemovalReview {
  return { review: async () => {
    const runs = new Set((await reviewRemovedRuns(projectId, { revision, files: [...files] }, signal))
      .map((key) => JSON.stringify([key.variant, key.step, key.row])));
    return ({ key }) => runs.has(JSON.stringify([key.variant, key.step, key.row]));
  } };
}

export interface RemovalReview { review(): Promise<RemovedRuns> }

/** What a command removes: the rows, or a review that finds them, asked only when some run could cost a confirmation. */
export type Removal = RemovedRuns | RemovalReview | undefined;
export type ConfirmRunRemoval = (removes: Removal) => Promise<boolean>;
export const RunRemovalContext = createContext<ConfirmRunRemoval | null>(null);

/**
 * The one confirmation for commands that remove runs: it asks only when a removed run holds a Train result or a
 * result that took at least the removal threshold, by the plan `read` gives of the latest revision. Results stay in the
 * Store; Undo restores the runs. The window draws `dialog` and passes `confirm` to its commands, through
 * `RunRemovalContext` or directly.
 */
export function useRunRemovalDialog(read: () => Promise<StudioSavedTrainRuns>) {
  const [asking, setAsking] = useState<{ count: number; training: boolean; answer(remove: boolean): void }>();
  const confirm = useCallback<ConfirmRunRemoval>(async (removal) => {
    if (!removal) return true;
    const threshold = removalThreshold();
    const plan = await read();
    if (!removalCost(plan, () => true, threshold).costly) return true;
    const removes = typeof removal === 'function' ? removal : await removal.review();
    const cost = removalCost(plan, removes, threshold);
    if (!cost.costly) return true;
    return new Promise<boolean>((resolve) => setAsking({ ...cost, answer: (remove) => { setAsking(undefined); resolve(remove); } }));
  }, [read]);
  const title = asking && studioFormat(asking.training
    ? asking.count === 1 ? 'removeTrainingRun' : 'removeTrainingRuns'
    : asking.count === 1 ? 'removeRun' : 'removeRuns', { count: asking.count });
  const dialog: ReactNode = asking && <Dialog ariaLabel={title} manageFocus onClose={() => asking.answer(false)}
    panelClassName="grid w-80 gap-3">
    <h2 className="font-medium">{title}</h2>
    <p className="text-shell-muted">{studioFormat('removedResultsRemain')}</p>
    <div className="flex justify-end gap-2">
      <Button onClick={() => asking.answer(false)}>{studioName('cancel')}</Button>
      <Button variant="primary" data-dialog-autofocus onClick={() => asking.answer(true)}>{studioName('remove')}</Button>
    </div>
  </Dialog>;
  return { confirm, dialog };
}

export function useRunRemoval() {
  const confirm = useContext(RunRemovalContext);
  if (!confirm) throw new Error('useRunRemoval needs a RunRemovalContext');
  return confirm;
}
