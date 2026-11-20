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
    choose: (path, key) => { written.push([path.join('/'), key]); },
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
  expect(written).toEqual([['schedule', 'cosine']]);
});

test('a class chosen from a tagged union draws its parameters, a class-valued one nested, and switching sends its key', async () => {
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
  const choose = (path: readonly string[], key: JsonValue) => { written.push([path.join('/'), key]); };
  const [optimizer] = fromJsonSchema('', schema, { optimizer: { type: 'adam', b1: 0.8, rate: { type: 'cosine', steps: 3 } } },
    { root: schema, choose }, []).parts!;
  expect(optimizer.type).toEqual({ kind: 'class', options: [{ value: 'adam', label: 'Adam' }, { value: 'sgd', label: 'SGD' }] });
  expect(optimizer.value).toBe('adam');
  expect(optimizer.parts!.map(({ label }) => label)).toEqual(['B1', 'Rate']);
  const rate = optimizer.parts![1];
  expect(rate.value).toBe('cosine');
  expect(rate.parts!.map(({ label, parameter, text }) => [label, parameter, text])).toEqual([['Rate', 'Steps', '3']]);
  await rate.commit!({ kind: 'set', value: 'constant' });
  await optimizer.commit!({ kind: 'set', value: 'sgd' });
  expect(written).toEqual([['optimizer/rate', 'constant'], ['optimizer', 'sgd']]);
  expect(fromJsonSchema('', schema, { optimizer: { type: 'adam' } }, { root: schema }, []).parts![0].commit).toBeUndefined();

  const [unset] = fromJsonSchema('', schema, { optimizer: { type: 'sgd', rate: null } }, { root: schema, choose }, [])
    .parts![0].parts!;
  expect(unset.type.kind).toBe('class');
  expect(unset.value).toBeUndefined();
});

test('a class field the source shows with several classes draws only the parameters all of them have', () => {
  const schema: JsonValue = {
    type: 'object',
    properties: { optimizer: { title: 'Optimizer', oneOf: [{ $ref: '#/$defs/Adam' }, { $ref: '#/$defs/SGD' }],
      discriminator: { propertyName: 'type', mapping: {} } } },
    $defs: {
      Adam: { title: 'Adam', type: 'object', properties: {
        type: { const: 'adam' }, rate: { type: 'number' }, b1: { type: 'number' } } },
      SGD: { title: 'SGD', type: 'object', properties: {
        type: { const: 'sgd' }, rate: { type: 'number' }, momentum: { type: 'number' } } },
    },
  };
  const value = { optimizer: { type: 'adam', rate: 0.1, b1: 0.8 } };
  const members = { values: [
    { id: 'adam', text: 'Adam', option: 'adam', selected: false }, { id: 'sgd', text: 'SGD', option: 'sgd', selected: false },
  ] };
  const field = { kind: 'sweep', kinds: [], members } as const;
  const draw = (classes: JsonValue[]) => fromJsonSchema('', schema, value, {
    root: schema, classed: (path) => path.join('/') === 'optimizer' ? { classes, field } : undefined,
  }, []).parts![0];

  const both = draw(['adam', 'sgd']);
  expect(both.parts!.map(({ label }) => label)).toEqual(['Rate']);
  expect([both.value, both.text, both.kind, both.members]).toEqual([undefined, 'Adam, SGD', 'sweep', members]);
  const adam = draw(['adam']);
  expect(adam.parts!.map(({ label }) => label)).toEqual(['Rate', 'B1']);
  expect(adam.value).toBe('adam');
});
