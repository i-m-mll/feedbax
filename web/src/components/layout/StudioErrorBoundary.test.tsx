// @vitest-environment jsdom

import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { StudioErrorBoundary } from '@/components/layout/StudioErrorBoundary';

function Train({ broken }: { broken: boolean }) {
  if (broken) throw new Error('Train renderer failed');
  return <button type="button">Run</button>;
}

it('contains a panel crash and reloads that panel while the editor keeps its draft', () => {
  const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
  const studio = (broken: boolean) => (
    <StudioErrorBoundary region="Studio">
      <input aria-label="Project name" defaultValue="Reach" />
      <StudioErrorBoundary region="Train">
        <Train broken={broken} />
      </StudioErrorBoundary>
    </StudioErrorBoundary>
  );

  try {
    const view = render(studio(false));
    const editor = screen.getByRole<HTMLInputElement>('textbox', { name: 'Project name' });
    fireEvent.change(editor, { target: { value: 'Point mass' } });

    view.rerender(studio(true));
    expect(screen.getByRole('alert').textContent).toContain('Train');
    expect(editor.value).toBe('Point mass');

    view.rerender(studio(false));
    fireEvent.click(screen.getByRole('button', { name: 'Reload' }));
    expect(screen.getByRole('button', { name: 'Run' })).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(editor.value).toBe('Point mass');
    view.unmount();
  } finally {
    consoleError.mockRestore();
  }
});
