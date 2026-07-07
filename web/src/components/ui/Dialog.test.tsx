// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { Dialog } from './Dialog';

describe('Dialog', () => {
  it('closes on Escape and backdrop clicks but not panel clicks', () => {
    const onClose = vi.fn();

    render(
      <Dialog onClose={onClose} ariaLabel="Test dialog" panelClassName="p-4">
        <button type="button">Inside</button>
      </Dialog>,
    );

    fireEvent.click(screen.getByText('Inside'));
    expect(onClose).not.toHaveBeenCalled();

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole('dialog'));
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});

