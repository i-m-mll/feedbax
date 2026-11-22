// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { ProjectWindow, ProjectWindowPalette } from './ProjectWindow';
import { defaultProjectWindowState, type ProjectWindowState } from './projectWindowState';
import { bindStudioCommand } from './studioCommands';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function Window({ zoom }: { zoom: () => void }) {
  const [state, setState] = useState<ProjectWindowState>(defaultProjectWindowState);
  return (
    <ProjectWindow state={state} onStateChange={setState} projectName="Reaching" views={{ available: ['runs', 'scene'], absent: {} }}
      renderView={(view) => ({ body: <p>{view} body</p> })} map={<p>map</p>} mapDrawer={null} compactMap={null}
      mapCommands={[bindStudioCommand('zoomIn', zoom)]} commands={[]} />
  );
}

it('covers the main area with Settings while the rail and gear keep working', () => {
  vi.stubGlobal('ResizeObserver', class {
    constructor(private readonly report: ResizeObserverCallback) {}
    observe() { this.report([{ contentRect: { width: 1440, height: 860 } } as ResizeObserverEntry], this as never); }
    disconnect() {}
  });
  const zoom = vi.fn();
  render(<ProjectWindowPalette><Window zoom={zoom} /></ProjectWindowPalette>);
  const gear = screen.getByRole('button', { name: 'Settings' });
  const runs = screen.getByRole('tab', { name: 'Runs' });
  expect(runs.getAttribute('aria-selected')).toBe('true');

  fireEvent.click(gear);
  expect(gear.getAttribute('aria-pressed')).toBe('true');
  expect(screen.getByRole('region', { name: 'Settings' })).toBeTruthy();
  expect(runs.getAttribute('aria-selected')).toBe('false');
  fireEvent.click(screen.getByRole('button', { name: 'Zoom In' }));
  expect(zoom).toHaveBeenCalledOnce();
  expect(screen.getByRole('region', { name: 'Settings' })).toBeTruthy();

  fireEvent.click(gear);
  expect(gear.getAttribute('aria-pressed')).toBe('false');
  expect(screen.queryByRole('region', { name: 'Settings' })).toBeNull();
  expect(runs.getAttribute('aria-selected')).toBe('true');
  expect(screen.getByText('runs body')).toBeTruthy();

  fireEvent.click(gear);
  fireEvent.click(screen.getByRole('tab', { name: 'Scene' }));
  expect(screen.queryByRole('region', { name: 'Settings' })).toBeNull();
  expect(screen.getByRole('tab', { name: 'Scene' }).getAttribute('aria-selected')).toBe('true');
});
