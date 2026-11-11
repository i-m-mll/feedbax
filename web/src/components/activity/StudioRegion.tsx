import { Component, useEffect, useState, type ErrorInfo, type ReactNode } from 'react';
import { ApiRequestError } from '@/api/request';
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

export function studioRegionIssue(name: StudioNameId, error: unknown, disconnected = false): StudioRegionIssue {
  const message = error instanceof Error ? error.message : undefined;
  const serviceUnavailable = disconnected || (error instanceof ApiRequestError && error.kind === 'network');
  return {
    reason: serviceUnavailable
      ? studioReason('unreachable', studioName('service'))
      : message
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
  skeleton: ReactNode;
  empty: ReactNode;
  children: ReactNode;
}) {
  switch (state.kind) {
    case 'loading':
      return <RegionLoading>{skeleton}</RegionLoading>;
    case 'empty':
      return <>{empty}</>;
    case 'failed':
      return <RegionFailure issue={state.issue} retry={state.retry} />;
    case 'result':
      return (
        <>
          {children}
          {state.missing.length > 0 && (
            <ul className="flex flex-wrap gap-2 py-1 text-sm text-slate-700" aria-live="polite">
              {state.missing.map(({ status, count }) => (
                <li key={status}>{count} {studioRunStatus(status)}</li>
              ))}
            </ul>
          )}
        </>
      );
  }
}

function RegionLoading({ children }: { children: ReactNode }) {
  const [showSkeleton, setShowSkeleton] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(() => setShowSkeleton(true), 200);
    return () => window.clearTimeout(timer);
  }, []);
  return <div aria-busy="true">{showSkeleton ? children : null}</div>;
}

function RegionFailure({ issue, retry }: { issue: StudioRegionIssue; retry: () => void }) {
  return (
    <div className="space-y-2 rounded-md border border-slate-200 bg-slate-50 p-3 text-sm">
      <p role="alert" className="text-red-700">{issue.reason}</p>
      <button
        type="button"
        className="rounded-md border border-slate-300 bg-white px-2 py-1 font-medium hover:bg-slate-100"
        onClick={retry}
      >
        {studioName('retry')}
      </button>
      {issue.detail && issue.detail !== issue.reason && (
        <details className="text-xs text-slate-700">
          <summary className="cursor-pointer">{studioName('details')}</summary>
          <pre className="mt-2 select-text whitespace-pre-wrap break-words">{issue.detail}</pre>
        </details>
      )}
    </div>
  );
}

export class StudioRegionBoundary extends Component<
  { name: StudioNameId; children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(studioName(this.props.name), error, info.componentStack);
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
      />
    );
  }
}
