import type { StudioProjectRevision } from '@/generated/studioContracts';
import type { StudioJobCommand, StudioJobPort } from './studioJobConnection';
import { openJobPort } from './studioJobStream';

export function watchProjectFolder(
  id: StudioProjectRevision['project_id'],
  handlers: {
    revision: (event: StudioProjectRevision) => void;
    connected: () => void;
  },
): () => void {
  let port: StudioJobPort | null = null;

  function open() {
    if (port) return;
    const opened = openJobPort();
    port = opened;
    opened.onmessage = ({ data }) => {
      if (port !== opened) return;
      if (data.type === 'folder-connected') handlers.connected();
      if (data.type === 'folder-revision' && data.revision.project_id === id) handlers.revision(data.revision);
    };
    opened.postMessage({ type: 'subscribe-folder', folderId: id } satisfies StudioJobCommand);
  }

  function close() {
    if (!port) return;
    const closing = port;
    port = null;
    closing.onmessage = null;
    closing.postMessage({ type: 'release' } satisfies StudioJobCommand);
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
