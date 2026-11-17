// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { useStudioTitleBar } from './studioTitleBar';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function Bar({ name }: { name: string }) {
  return <header {...useStudioTitleBar(name)} data-studio-drag-region>
    <h1 data-studio-drag-region>{name}</h1>
    <button type="button">Open command palette</button>
  </header>;
}

it('uses the acknowledged title and the noninteractive macOS drag surface', async () => {
  const windowAction = vi.fn(async () => {});
  vi.stubGlobal('__FEEDBAX_STUDIO_MENU__', { windowAction });
  const view = render(<Bar name="original" />);
  view.rerender(<Bar name="renamed" />);
  await waitFor(() => expect(windowAction.mock.calls).toEqual([
    [{ kind: 'title', title: 'original' }], [{ kind: 'title', title: 'renamed' }],
  ]));
  windowAction.mockClear();

  const title = screen.getByRole('heading');
  const button = screen.getByRole('button');
  fireEvent.mouseDown(button, { button: 0, detail: 1 });
  fireEvent.mouseDown(title, { button: 2, detail: 1 });
  fireEvent.mouseDown(title, { button: 0, detail: 1 });
  const double = { button: 0, detail: 2, clientX: 10, clientY: 20 };
  fireEvent.mouseDown(title, double);
  fireEvent.mouseUp(title, { ...double, clientX: 12 });
  fireEvent.mouseDown(title, double);
  fireEvent.mouseUp(button, double);
  fireEvent.mouseDown(title, double);
  fireEvent.mouseUp(title, double);
  expect(windowAction.mock.calls).toEqual([[{ kind: 'drag' }], [{ kind: 'zoom' }]]);
});

it('leaves a normal browser title selectable', () => {
  render(<Bar name="project" />);
  expect(fireEvent.mouseDown(screen.getByRole('heading'), { button: 0, detail: 1 })).toBe(true);
});
