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
});
