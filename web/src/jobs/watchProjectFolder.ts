import type { StudioProjectRevision } from '@/generated/studioContracts';
import JobsWorker from '@/jobs/studioJobWorker?sharedworker';
import {
  STUDIO_JOBS_WORKER_NAME,
  type StudioJobWorkerCommand,
  type StudioJobWorkerMessage,
} from './studioJobConnection';

export function watchProjectFolder(
  id: StudioProjectRevision['project_id'],
  handlers: {
    revision: (event: StudioProjectRevision) => void;
    connected: () => void;
  },
): () => void {
  let port: MessagePort | null = null;

  function open() {
    if (port) return;
    const opened = new JobsWorker({ name: STUDIO_JOBS_WORKER_NAME }).port;
    port = opened;
    opened.onmessage = ({ data }: MessageEvent<StudioJobWorkerMessage>) => {
      if (port !== opened) return;
      if (data.type === 'folder-connected') handlers.connected();
      if (data.type === 'folder-revision' && data.revision.project_id === id) handlers.revision(data.revision);
    };
    opened.postMessage({ type: 'subscribe-folder', folderId: id } satisfies StudioJobWorkerCommand);
  }

  function close() {
    if (!port) return;
    const closing = port;
    port = null;
    closing.onmessage = null;
    closing.postMessage({ type: 'release' } satisfies StudioJobWorkerCommand);
    closing.close();
  }

  open();
  window.addEventListener('pagehide', close);
  window.addEventListener('pageshow', open);
  return () => {
    window.removeEventListener('pagehide', close);
    window.removeEventListener('pageshow', open);
    close();
  };
}
