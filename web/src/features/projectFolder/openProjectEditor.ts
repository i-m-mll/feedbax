import { fetchProjectFolder, patchProjectFolder, writeProjectPresentation } from '@/api/projectFolders';
import type { PresentationDocument } from '@/generated/studioContracts';
import { watchProjectFolder } from '@/jobs/watchProjectFolder';
import { ProjectEditController } from './projectEditController';

export async function openProjectEditor(projectId: string, signal?: AbortSignal) {
  const initial = await fetchProjectFolder(projectId, signal);
  signal?.throwIfAborted();
  const editor = new ProjectEditController(initial, { read: fetchProjectFolder, write: patchProjectFolder });
  let unwatch: () => void;
  try {
    unwatch = watchProjectFolder(projectId, {
      revision: (notice) => editor.observe(notice),
      connected: () => editor.connected(),
    });
  } catch (error) {
    editor.dispose();
    throw error;
  }
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    signal?.removeEventListener('abort', close);
    unwatch();
    editor.dispose();
  };
  signal?.addEventListener('abort', close, { once: true });
  return {
    editor,
    close,
    async savePresentation(presentation: PresentationDocument, writeSignal?: AbortSignal) {
      if (closed) throw new Error('The project editor is closed.');
      const { saved, outside } = editor.getSnapshot();
      const result = await writeProjectPresentation(projectId, {
        revision: (outside ?? saved).presentation_revision,
        presentation,
      }, writeSignal);
      editor.observe(result);
      return result;
    },
  };
}
