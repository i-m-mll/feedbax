import { describe, expect, it } from 'vitest';
import type { StudioActivityJob } from '@/generated/studioContracts';
import { activityGroups } from './StudioActivity';

const job = (job_id: string, project_id: string, submitted_at_ms: number, ended_at_ms: number | null): StudioActivityJob => ({
  job_id, project_id, request_id: job_id, kind: 'train', run_key: job_id, stage_id: 'train', scenario_id: 'default', label: job_id,
  coordinates: {}, status: ended_at_ms === null ? 'running' : 'succeeded', submitted_at_ms, ended_at_ms, allowed_actions: [],
});

describe('activityGroups', () => {
  it('lists cards newest first by when they ended, or were submitted while still running', () => {
    const train = job('a-train', 'p', 100, 200);
    const evaluate = job('b-evaluate', 'p', 210, 260);
    const analyze = job('c-analyze', 'p', 270, null);
    const groups = activityGroups([train, analyze, evaluate], 'p');
    expect(groups.map((group) => group.runs.map((run) => run.job_id))).toEqual([['c-analyze', 'b-evaluate', 'a-train']]);
  });

  it('puts the window project first, then others by their newest card', () => {
    const groups = activityGroups([job('x', 'old', 0, 10), job('y', 'new', 0, 50), job('z', 'own', 0, 5)], 'own');
    expect(groups.map((group) => group.projectId)).toEqual(['own', 'new', 'old']);
  });
});
