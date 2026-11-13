import type { StudioSavedEvaluationRow } from '@/generated/studioContracts';

export function StudioEvaluationResults({ evaluation, onReplay }: {
  evaluation: StudioSavedEvaluationRow;
  onReplay(update: number): void;
}) {
  const series = evaluation.checkpoint_series;
  if (!series) return null;
  const { result } = evaluation;
  const sources = new Set(result?.checkpoint_sources?.map(({ completed_updates }) => completed_updates));
  const gaps = new Map(result?.checkpoint_gaps?.map(({ update, reason }) => [update, reason]));
  return <div className="space-y-2 p-3 text-sm">
    {result?.placement && <dl className="flex gap-2">
      <dt className="font-medium">Ran</dt>
      <dd>{result.placement === 'live' ? 'During training' : 'From saved checkpoints'}</dd>
    </dl>}
    {evaluation.reason && <p className="text-shell-muted">{evaluation.reason}</p>}
    <table aria-label="Checkpoint results" className="w-full border-collapse text-left">
      <thead>
        <tr className="border-b border-shell-divider">
          <th scope="col" className="py-1 pr-3 font-medium">Update</th>
          <th scope="col" className="py-1 font-medium">Result</th>
        </tr>
      </thead>
      <tbody>{series.requested.map((update) => <tr key={update} className="border-b border-shell-divider">
        <th scope="row" className="py-1 pr-3 font-normal tabular-nums">{update}</th>
        <td className="py-1">
          {sources.has(update) ? <button type="button"
            className="rounded px-1 underline underline-offset-2 hover:bg-shell-hover"
            aria-label={`Replay update ${update}`} onClick={() => onReplay(update)}>Replay</button>
            : gaps.get(update) === 'checkpoint_unavailable' ? 'Checkpoint unavailable'
              : 'Requested'}
        </td>
      </tr>)}</tbody>
    </table>
  </div>;
}
