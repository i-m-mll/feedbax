import { UpdateCadenceSchema, type StudioSavedEvaluationRow } from '@/generated/studioContracts';
import { studioCheckpointStatus, studioEvaluationPlacement, studioFormat, studioName, studioReason, studioReplayLabel } from '@/components/activity/studioVocabulary';

const percent = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 });

export function StudioEvaluationAttemptDetails({ evaluation }: { evaluation: StudioSavedEvaluationRow }) {
  const attempt = evaluation.checkpoint_attempt;
  const placement = attempt?.placement.location ?? evaluation.result?.placement;
  const pending = attempt?.pending;
  const subject = evaluation.checkpoint_series?.subject;
  let detail = evaluation.reason;
  if (pending && subject) {
    const step = subject.step;
    const { next_update: update, recovery_update: recovery } = pending;
    if (pending.state === 'resumable') {
      detail = recovery == null ? studioFormat('waitingForTrainResume', { step })
        : update == null ? studioFormat('waitingForCheckpointRecovery', { step, recovery })
          : studioFormat('waitingForCheckpointResume', { step, recovery, update });
    } else {
      detail = update == null ? studioFormat('waitingForCheckpoints', { step })
        : studioFormat('waitingForCheckpoint', { step, update });
    }
  }
  let budgetNotice: string | undefined;
  if (attempt?.placement.reason === 'over_budget' && attempt.placement.predicted_update_fraction != null) {
    const fraction = percent.format(attempt.placement.predicted_update_fraction);
    const cadence = UpdateCadenceSchema.safeParse(evaluation.checkpoint_rule);
    budgetNotice = cadence.success
      ? studioFormat('checkpointCadenceBudget', {
        cadence: studioFormat('checkpointCadence', { count: cadence.data.every }), fraction,
      })
      : studioFormat('checkpointBudget', { fraction });
  }
  return <>
    {placement && <dl className="flex gap-2">
      <dt className="font-medium">{studioName('timing')}</dt>
      <dd>{studioEvaluationPlacement(placement)}</dd>
    </dl>}
    {detail && <p className="text-shell-muted">{detail}</p>}
    {budgetNotice && <p className="text-shell-muted">{budgetNotice}</p>}
  </>;
}

export function StudioEvaluationResults({ evaluation, onReplay }: {
  evaluation: StudioSavedEvaluationRow;
  onReplay(update: number): void;
}) {
  const series = evaluation.checkpoint_series;
  if (!series) return null;
  const { checkpoint_attempt: attempt } = evaluation;
  const result = attempt && attempt.job_id !== evaluation.result?.job_id ? null : evaluation.result;
  const sources = new Set(result?.checkpoint_sources?.map(({ completed_updates }) => completed_updates));
  const gaps = new Map(result?.checkpoint_gaps?.map(({ update, reason }) => [update, reason]));
  const points = new Map(attempt?.points?.map(({ completed_updates, status }) => [completed_updates, status]));
  return <div className="space-y-2 p-3 text-sm">
    <StudioEvaluationAttemptDetails evaluation={evaluation} />
    <table aria-label={studioName('checkpointResults')} className="w-full border-collapse text-left">
      <thead>
        <tr className="border-b border-shell-divider">
          <th scope="col" className="py-1 pr-3 font-medium">{studioName('update')}</th>
          <th scope="col" className="py-1 font-medium">{studioName('result')}</th>
        </tr>
      </thead>
      <tbody>{series.requested.map((update) => <tr key={update} className="border-b border-shell-divider">
        <th scope="row" className="py-1 pr-3 font-normal tabular-nums">{update}</th>
        <td className="py-1">
          {sources.has(update) ? <button type="button"
            className="rounded px-1 underline underline-offset-2 hover:bg-shell-hover"
            aria-label={studioReplayLabel(update)}
            onClick={() => onReplay(update)}>{studioName('replay')}</button>
            : gaps.get(update) === 'checkpoint_unavailable' ? studioReason('unavailable', studioName('checkpoint'))
              : studioCheckpointStatus(points.get(update) ?? 'requested')}
        </td>
      </tr>)}</tbody>
    </table>
  </div>;
}
