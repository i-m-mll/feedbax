// @vitest-environment jsdom

import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { FigureGalleryPanel } from '@/components/panels/FigureGalleryPanel';
import { useFigures } from '@/hooks/useFigures';

vi.mock('@/hooks/useFigures', () => ({
  useFigures: vi.fn(),
}));

describe('FigureGalleryPanel', () => {
  it('shows the existing error UI without the clean empty state', () => {
    vi.mocked(useFigures).mockReturnValue({
      filters: {},
      evaluations: [],
      exptNames: [],
      figureTypes: [],
      pertTypes: [],
      updateFilter: vi.fn(),
      clearFilters: vi.fn(),
      figures: [],
      total: 0,
      loading: false,
      error: '500 Internal Server Error',
      currentPage: 1,
      totalPages: 0,
      goToPage: vi.fn(),
      selectedFigure: null,
      figureData: null,
      viewerLoading: false,
      viewerError: null,
      selectFigure: vi.fn(),
      closeFigure: vi.fn(),
    });

    render(<FigureGalleryPanel />);

    expect(screen.getByText('500 Internal Server Error')).toBeTruthy();
    expect(screen.queryByText('No figures yet')).toBeNull();
  });
});
