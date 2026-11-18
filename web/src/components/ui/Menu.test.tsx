// @vitest-environment jsdom

import { act, render, screen, waitFor } from '@testing-library/react';
import { useRef } from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { Menu, MenuContent, MenuItem, MenuTrigger } from './Menu';

function Fields({ open }: { open: boolean }) {
  const text = useRef<HTMLInputElement>(null);
  return <>
    <Menu open={open}>
      <MenuTrigger>Kind</MenuTrigger>
      <MenuContent aria-label="Kinds" finalFocus={text}><MenuItem>Sweep</MenuItem></MenuContent>
    </Menu>
    <input aria-label="Rate" ref={text} />
    <input aria-label="Type" />
  </>;
}

/** Holds the menu's exit animation until the test finishes it, as a browser does for its duration. */
function holdExitAnimation() {
  let finish!: () => void;
  const animation = {
    pending: false, playState: 'running',
    effect: { getTiming: () => ({ duration: 150, iterations: 1 }) },
    finished: new Promise<void>((resolve) => { finish = resolve; }),
  };
  vi.spyOn(Element.prototype, 'getAnimations').mockImplementation(() => [animation as unknown as Animation]);
  return () => act(async () => { animation.playState = 'finished'; finish(); });
}

afterEach(() => { vi.restoreAllMocks(); });

async function close(rerender: (ui: React.ReactElement) => void, during: () => void) {
  if (!('getAnimations' in Element.prototype)) Object.defineProperty(Element.prototype, 'getAnimations', { configurable: true, value: () => [] });
  const finish = holdExitAnimation();
  rerender(<Fields open={false} />);
  await act(async () => {});
  act(during);
  await finish();
  await waitFor(() => expect(screen.queryByRole('menu', { name: 'Kinds' })).toBeNull());
}

it('leaves focus where the person moved it while the menu was closing', async () => {
  const { rerender } = render(<Fields open />);
  await act(async () => {});
  await close(rerender, () => screen.getByRole('textbox', { name: 'Type' }).focus());
  expect(document.activeElement).toBe(screen.getByRole('textbox', { name: 'Type' }));
});

it('returns focus to its final target when focus stayed in the menu', async () => {
  const { rerender } = render(<Fields open />);
  await act(async () => {});
  act(() => { screen.getByRole('menuitem', { name: 'Sweep' }).focus(); });
  await close(rerender, () => {});
  expect(document.activeElement).toBe(screen.getByRole('textbox', { name: 'Rate' }));
});
