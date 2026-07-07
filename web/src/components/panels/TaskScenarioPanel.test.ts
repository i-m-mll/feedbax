import { describe, expect, it } from 'vitest';
import { descriptorForTaskParam } from './TaskScenarioPanel';

describe('TaskScenarioPanel value spec descriptors', () => {
  it('allows fixed constants for task params so default ValueSpec v2 drafts can commit', () => {
    expect(descriptorForTaskParam('target_radius')).toMatchObject({
      ownerKind: 'task_param',
      semanticKind: 'static_leaf',
      allowedScopes: ['fixed', 'run', 'sweep'],
    });
  });

  it('keeps static shape params out of sweep axes', () => {
    expect(descriptorForTaskParam('n_targets')).toMatchObject({
      semanticKind: 'static_shape',
      allowedScopes: ['fixed', 'run'],
    });
  });
});
