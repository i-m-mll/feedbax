import { Combobox } from './Combobox';

/** A registered declaration: a training method, task type, operation or figure template. */
export interface CatalogEntry {
  readonly key: string;
  readonly label: string;
  /** The package that declares it. */
  readonly package?: string;
}

export interface CatalogPickerProps {
  entries: readonly CatalogEntry[];
  value: string | null;
  onChoose: (key: string) => void;
  disabled?: boolean;
  placeholder?: string;
  'aria-label'?: string;
  'aria-labelledby'?: string;
  className?: string;
}

/** A combobox over a registry: each row the declaration's label, with its package as trailing text. */
export function CatalogPicker({ entries, value, onChoose, disabled, placeholder, className, ...aria }: CatalogPickerProps) {
  return <Combobox items={entries.map(({ key, label, package: source }) => ({ value: key, label, detail: source }))}
    value={value} disabled={disabled} placeholder={placeholder} className={className} {...aria}
    onValueChange={(key) => { if (key !== null && key !== value) onChoose(key); }} />;
}
