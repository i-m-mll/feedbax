import { describe, expect, it, vi } from 'vitest';
import type { Link, TaskDocument, Variable } from '@/generated/studioContracts';
import { variableEdits } from './trialTask';

const number: Variable['type'] = { base: 'real', shape: [] };
const variable = (id: string): Variable => ({ id, name: id, type: number, value: { kind: 'literal', value: 0 } });

describe('variableEdits with Set entries', () => {
  const onset = variable('onset');
  const links: Link[] = [{ model_field: 'pulse', variable: 'onset' }];
  const scoped = { ...onset, id: 'set-1a2b' };
  const task = { variables: [variable('target'), scoped] } as unknown as TaskDocument;

  // Catches: an entry's variable edited on the Task (where it is not) or under its id in the trials rather than the entry's,
  // its links left dangling, or Add of the selected entry landing on the Task.
  it("stores an entry's variables under their own ids for its variables and for Add, and edits the Task for the Task's", async () => {
    const commit = vi.fn(async () => ({ kind: 'stored' }));
    const write = vi.fn(async () => undefined);
    const entry = { key: 'force', label: 'force', variables: [onset], links, write };
    const edits = variableEdits(task, commit as never, new Map([[scoped.id, { entry, variable: 'onset' }]]), entry);
    await edits.remove!(scoped.id);
    expect(write).toHaveBeenLastCalledWith(expect.any(String), { task_variables: [], task_links: [] });
    await edits.add!('target');
    expect(write).toHaveBeenLastCalledWith(expect.any(String), {
      task_variables: [onset, expect.objectContaining({ id: 'target-2', name: 'target' })],
    });
    expect(commit).not.toHaveBeenCalled();
    await edits.remove!('target');
    expect(commit).toHaveBeenCalledOnce();
  });
});
