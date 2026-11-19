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
  expect(schedule.type.kind).toBe('class');
  expect(schedule.parts).toEqual([]);
  await schedule.commit!({ kind: 'set', value: 'cosine' });
  expect(written).toEqual([['schedule', { warmup: 5, kind: 'cosine' }]]);
});

test('a class chosen from a tagged union draws its parameters, a class-valued one nested, and switching writes its defaults once', async () => {
  const union = (...names: string[]) => ({
    oneOf: names.map((name) => ({ $ref: `#/$defs/${name}` })), discriminator: { propertyName: 'type', mapping: {} },
  });
  const schema: JsonValue = {
    type: 'object',
    properties: { optimizer: { title: 'Optimizer', ...union('Adam', 'SGD') } },
    $defs: {
      Adam: { title: 'Adam', type: 'object', properties: {
        type: { const: 'adam', type: 'string' }, b1: { type: 'number', default: 0.9 },
        rate: { title: 'Rate', ...union('Constant', 'Cosine'), default: { type: 'constant', value: 0.1 } },
      } },
      SGD: { title: 'SGD', type: 'object', properties: {
        type: { const: 'sgd', type: 'string' },
        rate: { anyOf: [union('Constant', 'Cosine'), { type: 'null' }], default: null },
      } },
      Constant: { title: 'Constant', type: 'object', properties: { type: { const: 'constant' }, value: { type: 'number' } } },
      Cosine: { title: 'Cosine', type: 'object', properties: { type: { const: 'cosine' }, steps: { type: 'integer', default: 9 } } },
    },
  };
  const written: [string, JsonValue][] = [];
  const write = (path: readonly string[], value: JsonValue) => { written.push([path.join('/'), value]); };
  const [optimizer] = fromJsonSchema('', schema, { optimizer: { type: 'adam', b1: 0.8, rate: { type: 'cosine', steps: 3 } } },
    { root: schema, write }, []).parts!;
  expect(optimizer.type).toEqual({ kind: 'class', options: [{ value: 'adam', label: 'Adam' }, { value: 'sgd', label: 'SGD' }] });
  expect(optimizer.value).toBe('adam');
  expect(optimizer.parts!.map(({ label }) => label)).toEqual(['B1', 'Rate']);
  const rate = optimizer.parts![1];
  expect(rate.value).toBe('cosine');
  expect(rate.parts!.map(({ label, parameter, text }) => [label, parameter, text])).toEqual([['Rate', 'Steps', '3']]);
  await rate.commit!({ kind: 'set', value: 'constant' });
  await optimizer.commit!({ kind: 'set', value: 'sgd' });
  expect(written).toEqual([
    ['optimizer/rate', { type: 'constant' }],
    ['optimizer', { type: 'sgd', rate: { type: 'cosine', steps: 3 } }],
  ]);

  const [unset] = fromJsonSchema('', schema, { optimizer: { type: 'sgd', rate: null } }, { root: schema, write }, [])
    .parts![0].parts!;
  expect(unset.type.kind).toBe('class');
  expect(unset.value).toBeUndefined();
});

test('choosing another class keeps the parameters it shares with the current one, and takes defaults for the rest', async () => {
  const constant = { $ref: '#/$defs/Constant' };
  const schema: JsonValue = {
    type: 'object',
    properties: { optimizer: { oneOf: [{ $ref: '#/$defs/Adam' }, { $ref: '#/$defs/AdamW' }, { $ref: '#/$defs/Lion' }],
      discriminator: { propertyName: 'type', mapping: {} } } },
    $defs: {
      Adam: { title: 'Adam', type: 'object', properties: {
        type: { const: 'adam' }, learning_rate: { ...constant, default: { type: 'constant', value: 0.001 } },
        b1: { type: 'number', default: 0.9 },
      } },
      AdamW: { title: 'AdamW', type: 'object', properties: {
        type: { const: 'adamw' }, learning_rate: { ...constant, default: { type: 'constant', value: 0.001 } },
        b1: { type: 'number', default: 0.9 }, weight_decay: { type: 'number', default: 1e-4 },
      } },
      Lion: { title: 'Lion', type: 'object', properties: {
        type: { const: 'lion' }, learning_rate: { type: 'number', default: 1e-4 }, b1: { type: 'number', default: 0.9 },
      } },
      Constant: { title: 'Constant', type: 'object', properties: { type: { const: 'constant' }, value: { type: 'number' } } },
    },
  };
  const written: JsonValue[] = [];
  const [optimizer] = fromJsonSchema('', schema, {
    optimizer: { type: 'adam', learning_rate: { type: 'constant', value: 0.003 }, b1: 0.95 },
  }, { root: schema, write: (_, value) => { written.push(value); } }, []).parts!;
  await optimizer.commit!({ kind: 'set', value: 'adamw' });
  await optimizer.commit!({ kind: 'set', value: 'lion' });
  expect(written).toEqual([
    { type: 'adamw', learning_rate: { type: 'constant', value: 0.003 }, b1: 0.95, weight_decay: 1e-4 },
    { type: 'lion', learning_rate: 1e-4, b1: 0.95 },
  ]);
});
