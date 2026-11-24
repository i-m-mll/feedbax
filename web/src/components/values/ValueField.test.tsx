// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, test } from 'vitest';
import type { JsonValue } from '@/generated/studioContracts';
import { fromParamSchema } from './adapters/paramSchema';
import { ValueField } from './ValueField';

afterEach(cleanup);

function field(written: JsonValue[]) {
  const model = fromParamSchema({ name: 'batches', type: 'int' }, 4, (value) => { written.push(value); });
  render(<><ValueField model={model} /><button type="button">next</button></>);
  return screen.getByRole('textbox');
}

test('a typed value is saved when the field is left by Tab, not only by Enter', async () => {
  const written: JsonValue[] = [];
  const input = field(written);
  await userEvent.clear(input);
  await userEvent.type(input, '12');
  await userEvent.tab();
  await waitFor(() => expect(written).toEqual([12]));
});

test('a typed value is saved when the field loses focus to a click elsewhere', async () => {
  const written: JsonValue[] = [];
  const input = field(written);
  await userEvent.clear(input);
  await userEvent.type(input, '7');
  await userEvent.click(screen.getByRole('button', { name: 'next' }));
  await waitFor(() => expect(written).toEqual([7]));
});

test('Escape drops the typed value, and leaving the field afterwards saves nothing', async () => {
  const written: JsonValue[] = [];
  const input = field(written);
  await userEvent.type(input, '9');
  await userEvent.keyboard('{Escape}');
  await userEvent.tab();
  expect(written).toEqual([]);
});
