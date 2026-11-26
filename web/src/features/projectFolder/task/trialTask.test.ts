import { describe, expect, it, vi } from 'vitest';
import type { StoredTaskDocument, StoredVariable } from '@/generated/studioContracts';
import { variableEdits } from './trialTask';

const number: StoredVariable['type'] = { base: 'real', shape: [] };
const variable = (label: string): StoredVariable => ({ label, type: number, value: { kind: 'literal', value: 0 } });

describe('variableEdits with Set entries', () => {
  const onset = variable('onset');
  const links = { pulse: 'onset' };
  const task = { variables: { target: variable('target'), 'set-1a2b': onset } } as unknown as StoredTaskDocument;

  // Catches: an entry's variable edited on the Task (where it is not) or under its key in the trials rather than the entry's,
  // its links left dangling, Add of the selected entry landing on the Task, or a new key a member key cannot be.
  it("stores an entry's variables under their own keys for its variables and for Add, and edits the Task for the Task's", async () => {
    const commit = vi.fn(async () => ({ kind: 'stored' }));
    const write = vi.fn(async () => undefined);
    const entry = { key: 'force', label: 'force', variables: { onset }, links, write };
    const edits = variableEdits(task, commit as never, new Map([['set-1a2b', { entry, variable: 'onset' }]]), entry);
    await edits.remove!('set-1a2b');
    expect(write).toHaveBeenLastCalledWith(expect.any(String), { task_variables: {}, task_links: {} });
    await edits.add!('target');
    expect(write).toHaveBeenLastCalledWith(expect.any(String), {
      task_variables: { onset, 'target-2': expect.objectContaining({ label: 'target' }) },
    });
    await edits.add!('2nd goal');
    const added = Object.keys((write.mock.lastCall as unknown as [string, { task_variables: object }])[1].task_variables);
    expect(added.at(-1)).toMatch(/^[A-Za-z_][A-Za-z0-9_-]*$/);
    expect(commit).not.toHaveBeenCalled();
    await edits.remove!('target');
    expect(commit).toHaveBeenCalledOnce();
  });
});
