import { describe, expect, it, vi } from 'vitest';
import type { Link, TaskDocument, Variable } from '@/generated/studioContracts';
import { variableEdits } from './trialTask';

const number: Variable['type'] = { base: 'real', shape: [] };
const variable = (id: string): Variable => ({ id, name: id, type: number, value: { kind: 'literal', value: 0 } });

describe('variableEdits with a selected Set entry', () => {
  const onset = variable('onset');
  const links: Link[] = [{ model_field: 'pulse', variable: 'onset' }];
  const task = { variables: [variable('target'), onset] } as unknown as TaskDocument;

  // Catches: an entry's variable removed or added on the Task (where it is not), or its links left dangling.
  it("stores the entry's own variables for its variables and for Add, and edits the Task for the Task's", async () => {
    const commit = vi.fn(async () => ({ kind: 'stored' }));
    const write = vi.fn(async () => undefined);
    const edits = variableEdits(task, commit as never, { label: 'force', variables: [onset], links, write });
    await edits.remove!('onset');
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
