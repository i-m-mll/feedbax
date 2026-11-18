import { describe, expect, it } from 'vitest';
import { awaitedValue, commandQueue } from './commandQueue';

function held() {
  let release!: () => void;
  const done = new Promise<void>((resolve) => { release = resolve; });
  return { done, release };
}

describe('commandQueue', () => {
  it('sends a command issued while a request is pending once that request completes', async () => {
    const run = commandQueue();
    const sent: string[] = [];
    const edit = held();
    const first = run(async () => { sent.push('edit'); await edit.done; return 'sent'; });
    const move = run(async () => { sent.push('move'); return 'sent'; });
    await Promise.resolve();
    expect(sent).toEqual(['edit']);
    edit.release();
    await expect(first).resolves.toBe('sent');
    await expect(move).resolves.toBe('sent');
    expect(sent).toEqual(['edit', 'move']);
  });

  it('runs the next command after one that failed', async () => {
    const run = commandQueue();
    const failed = run(async () => { throw new Error('refused'); });
    const next = run(async () => 'sent');
    await expect(failed).rejects.toThrow('refused');
    await expect(next).resolves.toBe('sent');
  });
});

describe('awaitedValue', () => {
  it('holds a queued edit, and the commands behind it, until the plan of its revision is set', async () => {
    const run = commandQueue();
    const plan = awaitedValue<{ revision: string }>();
    const sent: string[] = [];
    plan.set({ revision: 'a' });
    const edit = run(async () => { sent.push(`edit with ${(await plan.next(({ revision }) => revision === 'b')).revision}`); });
    const next = run(async () => { sent.push('next'); });
    plan.set(undefined);
    await Promise.resolve();
    expect(sent).toEqual([]);
    plan.set({ revision: 'b' });
    await Promise.all([edit, next]);
    expect(sent).toEqual(['edit with b', 'next']);
  });
});
