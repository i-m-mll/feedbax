import type { Expression, Repeat, Variable } from '@/generated/studioContracts';

/**
 * A count as the task fixes it when it compiles: an integer, or an Integer variable whose value is one. A count the
 * client cannot read so (a formula) is undefined.
 */
export function fixedCount(count: Expression, variables: readonly Pick<Variable, 'id' | 'value'>[]): number | undefined {
  const literal = (value: Expression) => value.kind === 'literal' && Number.isInteger(value.value) && !value.unit
    ? value.value as number : undefined;
  if (count.kind === 'name') {
    const variable = variables.find(({ id }) => id === count.reference);
    return variable && literal(variable.value);
  }
  return literal(count);
}

/**
 * The epochs of one trial in order, one entry per instance: a bracket's epochs once per repetition, each with its
 * repetition. A bracket whose count is not fixed is drawn once.
 */
export function epochInstances<Epoch extends { readonly id: string }>(
  epochs: readonly Epoch[], repeats: readonly Pick<Repeat, 'first_epoch' | 'last_epoch' | 'count'>[],
  variables: readonly Pick<Variable, 'id' | 'value'>[],
): { readonly epoch: Epoch; readonly repetition?: number }[] {
  const index = (id: string) => epochs.findIndex((epoch) => epoch.id === id);
  const instances: { epoch: Epoch; repetition?: number }[] = [];
  for (let at = 0; at < epochs.length;) {
    const repeat = repeats.find(({ first_epoch }) => index(first_epoch) === at);
    const last = repeat ? index(repeat.last_epoch) : -1;
    if (!repeat || last < at) {
      instances.push({ epoch: epochs[at] });
      at += 1;
      continue;
    }
    const span = epochs.slice(at, last + 1);
    const times = Math.max(1, fixedCount(repeat.count, variables) ?? 1);
    for (let repetition = 0; repetition < times; repetition += 1) {
      instances.push(...span.map((epoch) => ({ epoch, repetition })));
    }
    at = last + 1;
  }
  return instances;
}
