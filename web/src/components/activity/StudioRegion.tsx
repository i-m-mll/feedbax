import { Component, useEffect, useState, type ErrorInfo, type ReactNode } from 'react';
import { ApiRequestError } from '@/api/request';
import { isStudioDefect, shownError, shownRenderFailure } from '@/api/shownErrors';
import { studioName, studioReason, studioRunStatus, type StudioNameId, type StudioRunStatusId } from './studioVocabulary';
import { Button } from '@/components/ui/Button';
import { Section } from '@/components/ui/Section';

export interface StudioRegionIssue {
  reason: string;
  detail?: string;
  /** Studio's own defect, which asking again repeats. */
  defect?: boolean;
}

export type StudioRegionState =
  /** A region that has never had content; one that has keeps it while its next read is pending (`useRegionPending`). */
  | { kind: 'loading' }
  | { kind: 'empty' }
  | { kind: 'failed'; issue: StudioRegionIssue; retry: () => void }
  | { kind: 'result'; missing: ReadonlyArray<{ status: StudioRunStatusId; count: number }> }
  | { kind: 'partial'; unread: ReadonlyArray<{ key: string; reason: string }> };

export function studioReadState(
  name: StudioNameId, data: unknown, error: unknown, retry: () => void,
): StudioRegionState {
  if (error) return { kind: 'failed', issue: studioRegionIssue(name, error), retry };
  return data === undefined ? { kind: 'loading' } : { kind: 'result', missing: [] };
}

export function studioRegionIssue(name: StudioNameId, error: unknown): StudioRegionIssue {
  const shown = shownError(error, studioReason('unavailable', studioName(name)));
  if (isStudioDefect(error)) return { reason: shown, defect: true };
  return {
    reason: error instanceof Error && !(error instanceof ApiRequestError && (error.status ?? 0) >= 500)
      ? studioReason('observed', shown)
      : studioReason('unavailable', studioName(name)),
    detail: shown,
  };
}

/** How long a region's pending read may take before the region's content is dimmed. */
export const regionPendingMs = 500;

/**
 * Marks a region whose content is of an earlier revision while its read for the shown revision is pending: it is busy at
 * once and, once the read has taken longer than `regionPendingMs`, its content is dimmed as a whole until the read arrives.
 */
export function useRegionPending(pending: boolean) {
  const [late, setLate] = useState(false);
  useEffect(() => {
    if (!pending) return;
    const timer = window.setTimeout(() => setLate(true), regionPendingMs);
    return () => {
      window.clearTimeout(timer);
      setLate(false);
    };
  }, [pending]);
  return { 'aria-busy': pending || undefined, 'data-region-pending': pending && late ? '' : undefined };
}

export function StudioRegion({
  state,
  skeleton,
  empty,
  children,
}: {
  state: StudioRegionState;
  skeleton?: ReactNode;
  empty?: ReactNode;
  children?: ReactNode;
}) {
  switch (state.kind) {
    case 'loading':
      return <RegionLoading>{skeleton ?? studioRegionSkeleton}</RegionLoading>;
    case 'empty':
      return <>{empty}</>;
    case 'failed':
      return <RegionFailure issue={state.issue} retry={state.retry} />;
    case 'result':
      return (
        <>
          {children}
          {state.missing.length > 0 && (
            <ul className="flex flex-wrap gap-2 py-1 text-sm text-shell-muted" aria-live="polite">
              {state.missing.map(({ status, count }) => (
                <li key={status}>{count} {studioRunStatus(status)}</li>
              ))}
            </ul>
          )}
        </>
      );
    case 'partial':
      return (
        <>
          {children}
          <ul className="space-y-1 py-1 text-sm text-shell-status-failure">
            {state.unread.map(({ key, reason }) => <li key={key} role="alert">{reason}</li>)}
          </ul>
        </>
      );
  }
}

const studioRegionSkeleton = <div className="h-20 rounded bg-shell-subtle motion-safe:animate-pulse" />;

function RegionLoading({ children }: { children: ReactNode }) {
  const [showSkeleton, setShowSkeleton] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(() => setShowSkeleton(true), 200);
    return () => window.clearTimeout(timer);
  }, []);
  return <div aria-busy="true">{showSkeleton ? children : null}</div>;
}

function RegionFailure({ issue, retry, recoveryLabel = 'retry' }: {
  issue: StudioRegionIssue; retry: () => void; recoveryLabel?: 'retry' | 'reload';
}) {
  if (issue.defect) return <p role="alert" className="p-3 text-sm text-shell-status-failure">{issue.reason}</p>;
  return (
    <div className="space-y-2 rounded-md border border-shell-divider bg-shell-subtle p-3 text-sm">
      <p role="alert" className="text-shell-status-failure">{issue.reason}</p>
      <Button
        onClick={(event) => {
          event.currentTarget.closest<HTMLElement>('[data-studio-focus-region]')?.focus();
          retry();
        }}
      >
        {studioName(recoveryLabel)}
      </Button>
      {issue.detail && issue.detail !== issue.reason && (
        <Section level="field" name={studioName('details')} defaultOpen={false}>
          <pre className="select-text whitespace-pre-wrap break-words text-xs text-shell-muted">{issue.detail}</pre>
        </Section>
      )}
    </div>
  );
}

export class StudioRegionBoundary extends Component<
  { name: StudioNameId; children: ReactNode; recoveryLabel?: 'reload' },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };
  private reported = false;

  static getDerivedStateFromError(failure: unknown) {
    return { error: asRenderError(failure) };
  }

  componentDidCatch(failure: unknown, info: ErrorInfo) {
    // Strict Mode's second effect pass throws again within the same failure; report the failure once.
    if (this.reported) return;
    this.reported = true;
    const { name } = this.props;
    shownRenderFailure(asRenderError(failure), name, info.componentStack ?? '', renderFailure(name));
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <RegionFailure
        issue={{
          reason: renderFailure(this.props.name),
          detail: this.state.error.message,
        }}
        retry={() => {
          this.reported = false;
          this.setState({ error: null });
        }}
        recoveryLabel={this.props.recoveryLabel}
      />
    );
  }
}

const renderFailure = (name: StudioNameId) => studioReason('renderFailed', studioName(name));

function asRenderError(failure: unknown) {
  if (failure instanceof Error) return failure;
  return new Error(typeof failure === 'string' ? failure : undefined);
}
