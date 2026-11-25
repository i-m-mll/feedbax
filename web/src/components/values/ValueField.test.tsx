// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, test } from 'vitest';
import type { JsonValue } from '@/generated/studioContracts';
import { fromParamSchema } from './adapters/paramSchema';
import { rawField } from './adapters/raw';
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

// Catches: a toggle looks unclicked while its edit saves, so the click seems to have done nothing; or a refused toggle stays shown.
test('a toggle shows its new value while it saves, and its stored value again when refused', async () => {
  let answer: (refusal?: string) => void = () => {};
  const model = rawField({ id: 'enabled', label: 'Enabled', type: { kind: 'boolean' }, value: true,
    change: () => new Promise<string | undefined>((resolve) => { answer = resolve; }) });
  render(<ValueField model={model} />);
  const box = screen.getByRole('checkbox');
  await userEvent.click(box);
  expect(box.getAttribute('aria-checked')).toBe('false');
  answer('The term is required');
  await waitFor(() => expect(box.getAttribute('aria-checked')).toBe('true'));
});
