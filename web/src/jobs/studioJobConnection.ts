import type { StudioJobEvent } from '@/generated/studioContracts';

export type StudioJobConnection = 'connecting' | 'connected' | 'disconnected';

export type StudioJobWorkerCommand =
  | { type: 'subscribe'; projectId: string | null }
  | { type: 'release' }
  | { type: 'retry' }
  | { type: 'seen'; jobId: string };

export type StudioJobWorkerMessage =
  | { type: 'attached' }
  | { type: 'reset' }
  | { type: 'state'; state: StudioJobConnection }
  | { type: 'event'; event: StudioJobEvent }
  | { type: 'seen'; jobId: string };
