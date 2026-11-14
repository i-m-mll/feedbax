import { Component, type ReactNode } from 'react';
import { studioName, studioReason } from '@/components/activity/studioVocabulary';

type StudioErrorBoundaryProps = {
  region: string;
  children: ReactNode;
};

export class StudioErrorBoundary extends Component<StudioErrorBoundaryProps, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (!this.state.failed) return this.props.children;

    return (
      <div className="flex items-center gap-3 p-3 text-sm text-shell-foreground">
        <p role="alert">{studioReason('renderFailed', this.props.region)}</p>
        <button
          type="button"
          onClick={() => this.setState({ failed: false })}
          className="rounded-md border border-shell-control px-2 py-1 font-medium hover:bg-shell-hover"
        >
          {studioName('reload')}
        </button>
      </div>
    );
  }
}
