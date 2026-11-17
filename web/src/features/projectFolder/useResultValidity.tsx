import { useEffect, useMemo, useRef, useState } from 'react';
import { acceptResult, checkResult } from '@/api/resultValidity';
import { apiErrorMessage } from '@/api/request';
import { studioName } from '@/components/activity/studioVocabulary';
import { Popover, PopoverContent } from '@/components/ui/Popover';
import { TextField } from '@/components/ui/TextField';
import { resultCheckable, resultCodeDifferences, type PlanResultView, type PlanRunView } from '@/features/pipeline/planView';
import { bindStudioCommand } from '@/features/projectWindow/studioCommands';
import type { CodeDifference } from '@/generated/studioContracts';
import { useStudioJobHistory } from '@/hooks/useStudioJobs';

/** The judgement's author. Studio has no user identity, so it records the client. */
const author = 'Studio';

/** The wire omits defaulted `kind` fields, but the backend's union is discriminated by them. */
function tagged(difference: CodeDifference): CodeDifference {
  return 'path' in difference ? { ...difference, kind: 'source' } : { ...difference, kind: 'registry' };
}

/** The element an Accept popover opened by a command points at: the invoker, else a selected run's mark or row. */
function resultAnchor(runs: readonly PlanRunView[]) {
  const active = document.activeElement;
  if (active?.closest('[data-command-id="accept"], [data-result-run]')) return active;
  for (const { id } of runs) {
    const element = document.querySelector(`[data-result-run="${CSS.escape(id)}"]`);
    if (element) return element;
  }
  return document.querySelector('[data-command-id="accept"]');
}

export function useResultValidity({ projectId, revision, refresh, onError }: {
  projectId: string;
  revision: string;
  refresh: () => Promise<unknown>;
  onError: (message: string) => void;
}) {
  const history = useStudioJobHistory(projectId);
  const cursor = history.data?.at(-1)?.cursor ?? 0;
  const latestCursor = useRef(cursor);
  latestCursor.current = cursor;
  // Request ids whose check is running, each with the event cursor at which it started.
  const [checks, setChecks] = useState<ReadonlyMap<string, number>>(new Map());
  const [accepting, setAccepting] = useState<{ anchor: Element | null; results: readonly PlanResultView[] }>();

  useEffect(() => {
    if (!checks.size) return;
    const settled = new Set((history.data ?? []).flatMap((event) =>
      event.kind === 'validity' && event.cursor > (checks.get(event.request_id) ?? Infinity) ? [event.request_id] : []));
    if (!settled.size) return;
    // The mark leaves checking only once the settled validity has been read.
    void refresh().then(() => settle(settled));
  }, [history.data, checks, refresh]);

  function settle(requests: Iterable<string>) {
    const done = new Set(requests);
    setChecks((current) => new Map([...current].filter(([request]) => !done.has(request))));
  }

  async function check(results: readonly PlanResultView[]) {
    const started = latestCursor.current;
    setChecks((current) => new Map([...current, ...results.map(({ requestId }) => [requestId, started] as const)]));
    try {
      const decisions = await Promise.all(results.map(async (result) => [result, await checkResult(projectId, {
        revision, request_id: result.requestId, receipt: result.validity.receipt,
      })] as const));
      await refresh();
      settle(decisions.flatMap(([result, decided]) =>
        decided.state !== result.validity.state || !decided.checks?.length ? [result.requestId] : []));
    } catch (error) {
      settle(results.map(({ requestId }) => requestId));
      onError(apiErrorMessage(error));
    }
  }

  async function accept(results: readonly PlanResultView[], why: string) {
    setAccepting(undefined);
    try {
      await Promise.all(results.map((result) => acceptResult(projectId, {
        revision, request_id: result.requestId, receipt: result.validity.receipt,
        differences: resultCodeDifferences(result).map(tagged), who: author, why,
      })));
    } catch (error) {
      onError(apiErrorMessage(error));
    }
    await refresh();
  }

  function commands(runs: readonly PlanRunView[], anchor?: Element) {
    const results = runs.flatMap((run) => run.results);
    const checkable = results.filter((result) => resultCheckable(result) && !checks.has(result.requestId));
    const acceptable = results.filter((result) => resultCodeDifferences(result).length > 0);
    return [
      bindStudioCommand('check', () => { void check(checkable); }, { enabled: checkable.length > 0 }),
      bindStudioCommand('accept', () => setAccepting({ anchor: anchor ?? resultAnchor(runs), results: acceptable }),
        { enabled: acceptable.length > 0 }),
    ];
  }

  const popover = (
    <Popover open={!!accepting} onOpenChange={(open) => { if (!open) setAccepting(undefined); }}>
      <PopoverContent anchor={accepting?.anchor} aria-label={studioName('accept')} className="w-72">
        <form onSubmit={(event) => {
          event.preventDefault();
          const why = new FormData(event.currentTarget).get('why');
          if (accepting && typeof why === 'string' && why.trim()) void accept(accepting.results, why.trim());
        }}>
          <TextField name="why" aria-label={studioName('reason')} autoFocus required />
        </form>
      </PopoverContent>
    </Popover>
  );

  const checking = useMemo<ReadonlySet<string>>(() => new Set(checks.keys()), [checks]);
  return { checking, commands, popover };
}
