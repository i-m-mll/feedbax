// @vitest-environment jsdom

import { act, fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { StudioCommandPalette } from './StudioCommandPalette';
import { bindStudioCommand } from './studioCommands';

HTMLDialogElement.prototype.showModal = function showModal() { this.open = true; };
HTMLDialogElement.prototype.close = function close() { this.open = false; };

it('lets Return reach a filtered command once it becomes enabled', () => {
  const invoked = vi.fn();
  const commands = (enabled: boolean) => [bindStudioCommand('addComponent', invoked, { name: 'addOperation', enabled })];
  const { rerender } = render(<StudioCommandPalette commands={commands(false)} close={() => {}} />);
  const input = screen.getByRole('combobox');
  fireEvent.change(input, { target: { value: 'Add Operation' } });
  rerender(<StudioCommandPalette commands={commands(true)} close={() => {}} />);
  act(() => { fireEvent.keyDown(input, { key: 'Enter' }); });
  expect(invoked).toHaveBeenCalledOnce();
});
