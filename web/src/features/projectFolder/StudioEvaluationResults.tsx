import { Fragment, type ReactNode } from 'react';
import { UpdateCadenceSchema, type StudioSavedEvaluationRow } from '@/generated/studioContracts';
import { Button } from '@/components/ui/Button';
import { Table, TableCell, TableHeaderCell, TableRowHeader } from '@/components/ui/Table';
import { studioCheckpointStatus, studioEvaluationPlacement, studioFormat, studioFormatParts, studioName, studioReason, studioReplayLabel } from '@/components/activity/studioVocabulary';

const percent = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 });

export function StudioEvaluationAttemptDetails({ evaluation, onShowStep }: {
  evaluation: StudioSavedEvaluationRow;
  /** Select the Train step the evaluation waits for. */
  onShowStep?: (step: string) => void;
}) {
  const attempt = evaluation.checkpoint_attempt;
  const placement = attempt?.placement.location ?? evaluation.result?.placement;
  const pending = attempt?.pending;
  const subject = evaluation.checkpoint_series?.subject;
  let detail: ReactNode = evaluation.reason;
  if (pending && subject) {
    const step = onShowStep
      ? <Button variant="bare" size="content" className="hover:underline" onClick={() => onShowStep(subject.step)}>{subject.step}</Button>
      : subject.step;
    const { next_update: update, recovery_update: recovery } = pending;
    const parts = pending.state === 'resumable'
      ? recovery == null ? studioFormatParts('waitingForTrainResume', { step })
        : update == null ? studioFormatParts('waitingForCheckpointRecovery', { step, recovery })
          : studioFormatParts('waitingForCheckpointResume', { step, recovery, update })
      : update == null ? studioFormatParts('waitingForCheckpoints', { step })
        : studioFormatParts('waitingForCheckpoint', { step, update });
    detail = parts.map((part, index) => <Fragment key={index}>{part}</Fragment>);
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
  return <div className="space-y-2 text-sm">
    <StudioEvaluationAttemptDetails evaluation={evaluation} />
    <Table aria-label={studioName('checkpointResults')} held={1}
      columns={[{ least: '6rem', greatest: '16rem' }, { least: '8rem' }]}>
      <thead>
        <tr>
          <TableHeaderCell scope="col">{studioName('update')}</TableHeaderCell>
          <TableHeaderCell scope="col">{studioName('result')}</TableHeaderCell>
        </tr>
      </thead>
      <tbody>{series.requested.map((update) => <tr key={update}>
        <TableRowHeader className="font-normal tabular-nums">{update}</TableRowHeader>
        <TableCell>
          {sources.has(update) ? <Button variant="ghost" className="-ml-control-inline"
            aria-label={studioReplayLabel(update)}
            onClick={() => onReplay(update)}>{studioName('replay')}</Button>
            : gaps.get(update) === 'checkpoint_unavailable' ? studioReason('unavailable', studioName('checkpoint'))
              : studioCheckpointStatus(points.get(update) ?? 'requested')}
        </TableCell>
      </tr>)}</tbody>
    </Table>
  </div>;
}
