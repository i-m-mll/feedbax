// @vitest-environment jsdom

import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { StudioRegionBoundary } from '@/components/activity/StudioRegion';

vi.mock('@/api/studioDiagnostics', () => ({ reportStudioRenderFailure: vi.fn(async () => {}) }));

function Train({ broken }: { broken: boolean }) {
  if (broken) throw new Error('Train renderer failed');
  return <button type="button">Run</button>;
}

it('contains a panel crash and reloads that panel while the editor keeps its draft', () => {
  const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
  const studio = (broken: boolean) => (
    <StudioRegionBoundary name="studio" recoveryLabel="reload">
      <input aria-label="Project name" defaultValue="Reach" />
      <StudioRegionBoundary name="train">
        <Train broken={broken} />
      </StudioRegionBoundary>
    </StudioRegionBoundary>
  );

  try {
    const view = render(studio(false));
    const editor = screen.getByRole<HTMLInputElement>('textbox', { name: 'Project name' });
    fireEvent.change(editor, { target: { value: 'Point mass' } });

    view.rerender(studio(true));
    expect(screen.getByRole('alert').textContent).toContain('Train');
    expect(editor.value).toBe('Point mass');

    view.rerender(studio(false));
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(screen.getByRole('button', { name: 'Run' })).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(editor.value).toBe('Point mass');
    view.unmount();
  } finally {
    consoleError.mockRestore();
  }
});
