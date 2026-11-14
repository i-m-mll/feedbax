import { Component, useEffect, useState, type ErrorInfo, type ReactNode } from 'react';
import { ApiRequestError } from '@/api/request';
import { reportStudioRenderFailure } from '@/api/studioDiagnostics';
import { studioName, studioReason, studioRunStatus, type StudioNameId, type StudioRunStatusId } from './studioVocabulary';

export interface StudioRegionIssue {
  reason: string;
  detail?: string;
}

export type StudioRegionState =
  | { kind: 'loading' }
  | { kind: 'empty' }
  | { kind: 'failed'; issue: StudioRegionIssue; retry: () => void }
  | { kind: 'result'; missing: ReadonlyArray<{ status: StudioRunStatusId; count: number }> };

export function studioReadState(
  name: StudioNameId, data: unknown, error: unknown, retry: () => void, disconnected = false,
): StudioRegionState {
  if (error || disconnected) return { kind: 'failed', issue: studioRegionIssue(name, error, disconnected), retry };
  return data === undefined ? { kind: 'loading' } : { kind: 'result', missing: [] };
}

export function studioRegionIssue(name: StudioNameId, error: unknown, disconnected = false): StudioRegionIssue {
  const message = error instanceof Error ? error.message : undefined;
  const serviceUnavailable = disconnected || (error instanceof ApiRequestError && error.kind === 'network');
  return {
    reason: serviceUnavailable
      ? studioReason('unreachable', studioName('service'))
      : message && !(error instanceof ApiRequestError && (error.status ?? 0) >= 500)
        ? studioReason('observed', message)
        : studioReason('unavailable', studioName(name)),
    detail: message,
  };
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
  return (
    <div className="space-y-2 rounded-md border border-shell-divider bg-shell-subtle p-3 text-sm">
      <p role="alert" className="text-shell-status-failure">{issue.reason}</p>
      <button
        type="button"
        className="rounded-md border border-shell-control bg-shell-surface px-2 py-1 font-medium hover:bg-shell-hover"
        onClick={(event) => {
          event.currentTarget.closest<HTMLElement>('[data-studio-focus-region]')?.focus();
          retry();
        }}
      >
        {studioName(recoveryLabel)}
      </button>
      {issue.detail && issue.detail !== issue.reason && (
        <details className="text-xs text-shell-muted">
          <summary className="cursor-pointer">{studioName('details')}</summary>
          <pre className="mt-2 select-text whitespace-pre-wrap break-words">{issue.detail}</pre>
        </details>
      )}
    </div>
  );
}

export class StudioRegionBoundary extends Component<
  { name: StudioNameId; children: ReactNode; recoveryLabel?: 'reload' },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    const { name } = this.props;
    void reportStudioRenderFailure(name, error, info.componentStack ?? '').catch((reportError: unknown) => {
      console.error('Studio render failure could not be sent', name, error, reportError);
    });
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <RegionFailure
        issue={{
          reason: studioReason('renderFailed', studioName(this.props.name)),
          detail: this.state.error.message,
        }}
        retry={() => this.setState({ error: null })}
        recoveryLabel={this.props.recoveryLabel}
      />
    );
  }
}
