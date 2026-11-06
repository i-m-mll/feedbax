// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { expect, it, vi } from 'vitest';
import { ValueSpecField, type ValueSpecFieldDescriptor } from './ValueSpecField';

const descriptor: ValueSpecFieldDescriptor = {
  id: 'component_param:unit_spec',
  label: 'Unit Spec',
  ownerKind: 'component_param',
  semanticKind: 'static_leaf',
  valueSchema: { type: 'array', nullable: true, shape: ['...'] },
  allowedModes: ['constant'],
  allowedScopes: ['fixed', 'run', 'sweep'],
};

it('authors a nullable array through inline and expanded literal controls', () => {
  const onChange = vi.fn();
  function Field() {
    const [value, setValue] = useState<unknown>(null);
    return (
      <ValueSpecField
        descriptor={descriptor}
        value={value}
        onChange={(next) => {
          onChange(next);
          setValue(next);
        }}
      />
    );
  }

  render(<Field />);
  const inline = screen.getByRole('textbox', { name: 'Unit Spec' });
  const open = () => fireEvent.click(screen.getByTitle('Edit Unit Spec value spec'));

  open();
  expect(screen.getByRole('textbox', { name: 'Literal' })).toHaveValue('');
  fireEvent.click(screen.getByRole('button', { name: 'Cancel value editor' }));

  fireEvent.change(inline, { target: { value: '[1, null]' } });
  fireEvent.blur(inline);
  expect(onChange).toHaveBeenLastCalledWith([1, null]);

  open();
  const literal = screen.getByRole('textbox', { name: 'Literal' });
  const done = screen.getByRole('button', { name: 'Done' });
  fireEvent.change(literal, { target: { value: '[2,' } });
  expect(literal).toHaveValue('[2,');
  expect(done).toBeDisabled();
  fireEvent.change(literal, { target: { value: '[1e999]' } });
  expect(done).toBeDisabled();
  fireEvent.change(literal, { target: { value: '[2, 3, 4]' } });
  fireEvent.click(done);
  expect(onChange).toHaveBeenLastCalledWith([2, 3, 4]);

  open();
  fireEvent.change(screen.getByRole('textbox', { name: 'Literal' }), { target: { value: '' } });
  fireEvent.click(screen.getByRole('button', { name: 'Done' }));
  expect(onChange).toHaveBeenLastCalledWith(null);

  fireEvent.change(inline, { target: { value: '[]' } });
  fireEvent.blur(inline);
  expect(onChange).toHaveBeenLastCalledWith([]);
  fireEvent.change(inline, { target: { value: 'null' } });
  fireEvent.blur(inline);
  expect(onChange).toHaveBeenLastCalledWith(null);
});
