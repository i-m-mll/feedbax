import type { StudioJobEvent, StudioProjectRevision } from '@/generated/studioContracts';

export const STUDIO_JOBS_WORKER_NAME = 'feedbax-studio-jobs-v1';

export type StudioJobConnection = 'connecting' | 'connected' | 'disconnected';

export type StudioJobWorkerCommand =
  | { type: 'subscribe'; projectId: string | null }
  | { type: 'subscribe-folder'; folderId: StudioProjectRevision['project_id'] }
  | { type: 'release' }
  | { type: 'retry' }
  | { type: 'seen'; jobId: string };

export type StudioJobWorkerMessage =
  | { type: 'attached' }
  | { type: 'reset' }
  | { type: 'state'; state: StudioJobConnection }
  | { type: 'event'; event: StudioJobEvent }
  | { type: 'seen'; jobId: string }
  | { type: 'folder-connected' }
  | { type: 'folder-revision'; revision: StudioProjectRevision };
