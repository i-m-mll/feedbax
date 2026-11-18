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
    [0, 1], { root: null, write: (path, value) => { written.push([path.join('/'), value]); } });
  expect(bounds.type.kind).toBe('vector');
  await bounds.parts![1].commit!({ kind: 'text', text: '3' });
  expect(written.at(-1)).toEqual(['workspace/1', 3]);
});

test('a settings schema walks references, optional values, free parameters and a tagged union', async () => {
  const schema: JsonValue = {
    type: 'object',
    properties: {
      optimizer: { $ref: '#/$defs/Optimizer' },
      clip: { anyOf: [{ type: 'number' }, { type: 'null' }], default: 1 },
      schedule: { oneOf: [
        { type: 'object', properties: { kind: { const: 'flat' } } },
        { type: 'object', properties: { kind: { const: 'cosine' }, warmup: { type: 'integer', default: 5 } } },
      ] },
    },
    $defs: { Optimizer: { title: 'OptimizerSpec', type: 'object', properties: {
      params: { type: 'object', additionalProperties: { anyOf: [{ type: 'number' }, { type: 'string' }] } },
    } } },
  };
  const written: [string, JsonValue][] = [];
  const projected = fromJsonSchema('', schema, {}, { root: schema }, ['projected']);
  const root = fromJsonSchema('', schema, {
    optimizer: { params: { learning_rate: 0.1, name: 'adam' } }, schedule: { kind: 'flat' },
  }, {
    root: schema, write: (path, value) => { written.push([path.join('/'), value]); },
    projected: (path) => path.join('/') === 'optimizer/params/learning_rate' ? projected : undefined,
  }, []);
  const [optimizer, clip, schedule] = root.parts!;
  expect(optimizer.label).toBe('Optimizer');
  const [params] = optimizer.parts!;
  expect(params.parts![0]).toBe(projected);
  expect(params.parts![1].type.kind).toBe('string');
  expect(clip.type.kind).toBe('number');
  expect(clip.text).toBe('1');
  const [kind] = schedule.parts!;
  expect(kind.type.kind).toBe('choice');
  await kind.commit!({ kind: 'set', value: 'cosine' });
  expect(written).toEqual([['schedule', { warmup: 5, kind: 'cosine' }]]);
});
