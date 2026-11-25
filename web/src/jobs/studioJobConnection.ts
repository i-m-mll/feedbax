import type { StudioJobEvent, StudioProjectPlanChanged, StudioProjectRevision } from '@/generated/studioContracts';

export type StudioJobConnection = 'connecting' | 'connected' | 'disconnected';

export type StudioJobCommand =
  | { type: 'subscribe'; projectId: string | null }
  | { type: 'subscribe-folder'; folderId: StudioProjectRevision['project_id'] }
  | { type: 'release' }
  | { type: 'retry' }
  | { type: 'seen'; jobId: string };

export type StudioJobMessage =
  | { type: 'attached' }
  | { type: 'reset' }
  | { type: 'machines-changed' }
  | { type: 'state'; state: StudioJobConnection }
  | { type: 'event'; event: StudioJobEvent }
  | { type: 'seen'; jobId: string }
  | { type: 'folder-connected' }
  | { type: 'project-plan-changed'; change: StudioProjectPlanChanged }
  | { type: 'folder-revision'; revision: StudioProjectRevision };

/** A subscriber's end of the window's job stream, shaped like a MessagePort. */
export interface StudioJobPort {
  onmessage: ((event: { data: StudioJobMessage }) => void) | null;
  postMessage(command: StudioJobCommand): void;
  close(): void;
}
