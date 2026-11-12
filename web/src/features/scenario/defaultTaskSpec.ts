import type { TaskSpec } from '@/types/training';

export const defaultTaskSpec = {
  type: 'SimpleReaches',
  params: {
    n_steps: 200,
    workspace: [
      [-1.0, -1.0],
      [1.0, 1.0],
    ],
    eval_n_directions: 7,
    eval_reach_length: 0.5,
    eval_grid_n: 1,
  },
} satisfies TaskSpec;
