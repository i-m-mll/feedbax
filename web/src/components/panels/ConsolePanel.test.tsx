// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { act } from 'react';
import { afterEach, describe, expect, it } from 'vitest';
import { useTrainingStore } from '@/stores/trainingStore';
import { ConsolePanel } from './ConsolePanel';

const initialTrainingState = useTrainingStore.getState();

afterEach(() => {
  act(() => {
    useTrainingStore.setState(initialTrainingState, true);
  });
});

describe('ConsolePanel', () => {
  it('renders the empty log state', () => {
    render(<ConsolePanel />);

    expect(screen.getByText('No logs yet')).toBeInTheDocument();
  });

  it('renders streamed training log lines', () => {
    act(() => {
      useTrainingStore.setState({
        consoleLogs: [
          {
            batch: 4,
            level: 'warning',
            message: 'loss plateaued',
            timestamp: 0,
          },
        ],
      });
    });

    render(<ConsolePanel />);

    expect(screen.getByText('[4]')).toBeInTheDocument();
    expect(screen.getByText('loss plateaued')).toBeInTheDocument();
  });

  it('renders structured training diagnostics line by line', () => {
    act(() => {
      useTrainingStore.setState({
        trainingDiagnostics: [
          {
            schema_id: 'feedbax.diagnostic.domain',
            schema_version: 'feedbax.diagnostic.domain.v1',
            severity: 'error',
            code: 'graph.missing_subgraph',
            message: "Network node 'network' has no subgraph",
            node_ids: ['network'],
            details: {},
          },
        ],
      });
    });

    render(<ConsolePanel />);

    expect(screen.getByText('[diagnostic]')).toBeInTheDocument();
    expect(
      screen.getByText("ERROR graph.missing_subgraph [network]: Network node 'network' has no subgraph")
    ).toBeInTheDocument();
  });
});
