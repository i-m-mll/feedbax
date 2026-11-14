export function humanizeLabel(label: string): string {
  return label
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}
