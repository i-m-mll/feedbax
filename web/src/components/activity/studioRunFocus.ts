import { create } from 'zustand';
import type { StudioActivityJob } from '@/generated/studioContracts';

export interface StudioRunFocus {
  projectId: string;
  stageId: string;
  runKey: string;
  requestId: string;
  jobId?: string;
}

export const useStudioRunFocus = create<{
  focus: StudioRunFocus | null;
  select: (focus: StudioRunFocus | null) => void;
}>((set) => ({
  focus: null,
  select: (focus) => set({ focus }),
}));

export function focusForJob(job: StudioActivityJob): StudioRunFocus {
  return {
    projectId: job.project_id,
    stageId: job.stage_id,
    runKey: job.run_key,
    requestId: job.request_id,
    jobId: job.job_id,
  };
}
