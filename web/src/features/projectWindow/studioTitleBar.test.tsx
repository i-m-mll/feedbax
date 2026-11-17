// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useRef } from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { useStudioTitleBar } from './studioTitleBar';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  delete document.documentElement.dataset.shellHost;
});

function Bar({ name }: { name: string }) {
  const bar = useRef<HTMLElement>(null);
  useStudioTitleBar(bar, name);
  return <header ref={bar} data-shell-drag-region>
    <h1>{name}</h1>
    <button type="button">Open command palette</button>
  </header>;
}

it('publishes the title and drags the native window from the bar', async () => {
  const windowAction = vi.fn(async () => {});
  vi.stubGlobal('__FEEDBAX_STUDIO_MENU__', { windowAction });
  document.documentElement.dataset.shellHost = 'native';
  const view = render(<Bar name="original" />);
  view.rerender(<Bar name="renamed" />);
  await waitFor(() => expect(windowAction.mock.calls).toEqual([
    [{ kind: 'title', title: 'original' }], [{ kind: 'title', title: 'renamed' }],
  ]));
  windowAction.mockClear();

  fireEvent.mouseDown(screen.getByRole('button'), { button: 0, detail: 1 });
  fireEvent.mouseDown(screen.getByRole('heading'), { button: 0, detail: 1 });
  expect(windowAction.mock.calls).toEqual([[{ kind: 'drag' }]]);
});

it('leaves a browser title bar to the browser', () => {
  render(<Bar name="project" />);
  expect(fireEvent.mouseDown(screen.getByRole('heading'), { button: 0, detail: 1 })).toBe(true);
});
