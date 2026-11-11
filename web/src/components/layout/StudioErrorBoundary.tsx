import { Component, type ReactNode } from 'react';

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
      <div className="flex items-center gap-3 p-3 text-sm text-slate-700">
        <p role="alert">{this.props.region} failed to render</p>
        <button
          type="button"
          onClick={() => this.setState({ failed: false })}
          className="rounded-md border border-slate-300 px-2 py-1 font-medium hover:bg-slate-100"
        >
          Reload
        </button>
      </div>
    );
  }
}
