import { createContext, useContext } from 'react';

/** Which field groups of a view's forms are open, by field id. Without one, each group keeps its own state, open at first. */
export interface FieldGroupFolds {
  isOpen(id: string): boolean;
  setOpen(id: string, open: boolean): void;
}

export const FieldGroups = createContext<FieldGroupFolds | undefined>(undefined);

export function useFieldGroup(id: string) {
  const folds = useContext(FieldGroups);
  return folds ? { open: folds.isOpen(id), onOpenChange: (open: boolean) => folds.setOpen(id, open) } : {};
}
