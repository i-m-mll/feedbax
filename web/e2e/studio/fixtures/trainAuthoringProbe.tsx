import { createRoot } from 'react-dom/client';
import { openProjectEditor, newProjectEditor } from '@/features/projectFolder/openProjectEditor';
import { StudioTrainAuthoringForm } from '@/features/projectFolder/StudioTrainAuthoringForm';
import type { StudioProjectCreateFile } from '@/generated/studioContracts';

export async function mountTrainAuthoring(
  container: HTMLElement, project: string | StudioProjectCreateFile['file'][],
) {
  const session = typeof project === 'string' ? await openProjectEditor(project) : await newProjectEditor();
  const { editor } = session;
  if (typeof project !== 'string') {
    for (const file of project) await editor.createFile(file);
  }
  const root = createRoot(container);
  root.render(<StudioTrainAuthoringForm editor={editor} stepId="train" />);
  return {
    snapshot: editor.getSnapshot,
    keep: (choice: 'disk' | 'edits') => editor.keep(choice),
    flush: () => editor.flush(),
    undo: () => { editor.undo(); return editor.flush(); },
    redo: () => { editor.redo(); return editor.flush(); },
    close: () => { root.unmount(); session.close(); },
  };
}
