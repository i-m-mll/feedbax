import { createContext } from 'react';
import type { FieldKind } from '@/generated/studioContracts';

/**
 * A command's request that the field whose anchor is `target` take `kind` from the text it next commits, as Add
 * Sweep's: the field shows the kind at once and writes nothing until then. `key` tells one request from the next.
 */
export interface FieldKindIntent {
  readonly key: string;
  readonly target: string;
  readonly kind: FieldKind;
}

export const FieldIntent = createContext<FieldKindIntent | undefined>(undefined);
