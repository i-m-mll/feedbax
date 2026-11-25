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

/** Hands `file` to the system as a download named `name`. */
export function saveFile(file: Blob, name: string) {
  const url = URL.createObjectURL(file);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url));
}
