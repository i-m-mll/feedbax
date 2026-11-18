// @vitest-environment jsdom

import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it } from 'vitest';
import { Button } from '@/components/ui/Button';
import { keyHandled } from './studioCommands';

const seen: boolean[] = [];
const listen = (event: KeyboardEvent) => seen.push(keyHandled(event));
afterEach(() => {
  window.removeEventListener('keydown', listen);
  seen.length = 0;
});

it('leaves a window shortcut to the window when focus rests on a disabled control', () => {
  render(<>
    <Button disabled focusableWhenDisabled>Run</Button>
    <Button onKeyDown={(event) => event.preventDefault()}>Takes keys</Button>
  </>);
  window.addEventListener('keydown', listen);
  fireEvent.keyDown(screen.getByRole('button', { name: 'Run' }), { key: 'z', metaKey: true });
  fireEvent.keyDown(screen.getByRole('button', { name: 'Takes keys' }), { key: 'z', metaKey: true });
  expect(seen).toEqual([false, true]);
});
