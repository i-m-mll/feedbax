import { shellHost } from '@window-shell/browser/titlebar';
/** Opens the system's file dialog; resolves to the chosen file, or nothing when it is dismissed. */
export function chooseFile(): Promise<File | undefined> {
  const input = document.createElement('input');
  input.type = 'file';
  return new Promise((resolve) => {
    input.addEventListener('change', () => resolve(input.files?.[0]), { once: true });
    input.addEventListener('cancel', () => resolve(undefined), { once: true });
    input.click();
  });
}

interface SavePicker {
  showSaveFilePicker(options: { suggestedName: string }): Promise<{
    createWritable(): Promise<{ write(data: Blob): Promise<void>; close(): Promise<void> }>;
  }>;
}

/**
 * Saves `file`, suggesting `name`, where the user chooses: the desktop app's save panel, else the browser's save
 * dialog, else, in a browser without one, a download. Answers false when the user cancels.
 */
export async function saveFile(file: Blob, name: string): Promise<boolean> {
  if (shellHost() === 'native') {
    const save = window.__FEEDBAX_STUDIO_MENU__?.saveFile;
    if (!save) throw new Error('The desktop window has no save panel.');
    return save(name, new Uint8Array(await file.arrayBuffer()));
  }
  if ('showSaveFilePicker' in window) {
    const handle = await (window as unknown as SavePicker).showSaveFilePicker({ suggestedName: name }).catch((error: unknown) => {
      if (error instanceof DOMException && error.name === 'AbortError') return undefined;
      throw error;
    });
    if (!handle) return false;
    const writable = await handle.createWritable();
    await writable.write(file);
    await writable.close();
    return true;
  }
  const url = URL.createObjectURL(file);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url));
  return true;
}
