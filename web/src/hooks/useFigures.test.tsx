// @vitest-environment jsdom

import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fetchEvaluationsWithFigures, fetchFigures } from '@/api/figureAPI';
import { useFigures } from '@/hooks/useFigures';

vi.mock('@/api/figureAPI', () => ({
  fetchEvaluationsWithFigures: vi.fn(),
  fetchFigures: vi.fn(),
  fetchFigureDetail: vi.fn(),
  fetchFigureFile: vi.fn(),
}));

let latestGallery: ReturnType<typeof useFigures> | null = null;

function FiguresHarness() {
  latestGallery = useFigures();
  return null;
}

describe('useFigures', () => {
  beforeEach(() => {
    latestGallery = null;
    vi.mocked(fetchEvaluationsWithFigures).mockReset().mockResolvedValue([]);
    vi.mocked(fetchFigures).mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it.each(['404 Not Found', '500 Internal Server Error'])(
    'surfaces %s instead of converting it to an empty registry',
    async (message) => {
      vi.mocked(fetchFigures).mockRejectedValue(new Error(message));

      await act(async () => {
        render(<FiguresHarness />);
      });

      expect(latestGallery).toMatchObject({
        figures: [],
        total: 0,
        loading: false,
        error: message,
      });
    },
  );

  it('preserves a successful empty registry without an error', async () => {
    vi.mocked(fetchFigures).mockResolvedValue({ items: [], total: 0, limit: 24, offset: 0 });

    await act(async () => {
      render(<FiguresHarness />);
    });

    expect(latestGallery).toMatchObject({
      figures: [],
      total: 0,
      loading: false,
      error: null,
    });
  });
});
