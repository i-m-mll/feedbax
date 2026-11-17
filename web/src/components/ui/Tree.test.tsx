// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { beforeAll, expect, it, vi } from 'vitest';
import { Tree, type TreeNode } from './Tree';

beforeAll(() => { Element.prototype.scrollIntoView = () => {}; });

const nodes: TreeNode[] = [
  { id: 'model', name: 'Model', children: [{ id: 'gru', name: 'GRU' }] },
  { id: 'noise', name: 'Noise' },
];

function Harness({ onActivate, onRename }: { onActivate: (id: string) => void; onRename: (id: string, name: string) => void }) {
  const [selected, setSelected] = useState<string[]>(['model']);
  return <Tree aria-label="Outline" nodes={nodes} selected={selected} onSelectedChange={setSelected}
    onActivate={onActivate} onRename={onRename} renameLabel="Rename" />;
}

it('goes in on Return and renames on F2', () => {
  const onActivate = vi.fn();
  const onRename = vi.fn();
  render(<Harness onActivate={onActivate} onRename={onRename} />);
  const model = screen.getByRole('treeitem', { name: /Model/ });
  expect(screen.queryByRole('treeitem', { name: /GRU/ })).toBeNull();

  fireEvent.keyDown(model, { key: 'Enter' });
  expect(screen.getByRole('treeitem', { name: /GRU/ })).toBeVisible();
  expect(onActivate).toHaveBeenCalledWith('model');
  expect(screen.queryByRole('textbox')).toBeNull();

  fireEvent.keyDown(model, { key: 'F2' });
  const name = screen.getByRole('textbox', { name: 'Rename' });
  fireEvent.change(name, { target: { value: 'Network' } });
  fireEvent.keyDown(name, { key: 'Enter' });
  expect(onRename).toHaveBeenCalledWith('model', 'Network');
});

it('renames on a double-click on the name and goes in on one beside it', () => {
  const onActivate = vi.fn();
  render(<Harness onActivate={onActivate} onRename={vi.fn()} />);
  const noise = screen.getByRole('treeitem', { name: /Noise/ });
  fireEvent.doubleClick(noise);
  expect(onActivate).toHaveBeenCalledWith('noise');
  expect(screen.queryByRole('textbox')).toBeNull();
  fireEvent.doubleClick(screen.getByText('Noise'));
  expect(screen.getByRole('textbox', { name: 'Rename' })).toHaveValue('Noise');
});
