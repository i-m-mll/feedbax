import { expect, test } from 'vitest';
import type { JsonValue } from '@/generated/studioContracts';
import { fromJsonSchema } from './jsonSchema';
import { fromParamSchema } from './paramSchema';

test('a part edit writes the whole parameter back, and text is parsed by type', async () => {
  const written: JsonValue[] = [];
  const optimizer = fromParamSchema({
    name: 'optimizer', type: 'object', nested_schema: [{ name: 'steps', type: 'int' }, { name: 'rate', type: 'float' }],
  }, { steps: 4, rate: 0.1 }, (value) => { written.push(value); });
  const [steps, rate] = optimizer.parts!;
  expect(await steps.commit!({ kind: 'text', text: '1.5' })).toMatch(/whole number/);
  expect(await rate.commit!({ kind: 'text', text: '0.2' })).toBeUndefined();
  expect(written).toEqual([{ steps: 4, rate: 0.2 }]);

  const bounds = fromJsonSchema('workspace', { type: 'array', minItems: 2, maxItems: 2, items: { type: 'number' } },
    [0, 1], (value) => { written.push(value); });
  expect(bounds.type.kind).toBe('vector');
  await bounds.parts![1].commit!({ kind: 'text', text: '3' });
  expect(written.at(-1)).toEqual([0, 3]);
});
