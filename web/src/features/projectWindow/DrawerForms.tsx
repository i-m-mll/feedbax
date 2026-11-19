import { useMemo, type Dispatch, type ReactNode, type SetStateAction } from 'react';
import { Section, SectionStack, type SectionProps } from '@/components/ui/Section';
import {
  sectionOpen, withSectionOpen, type ProjectWindowSection, type ProjectWindowState,
} from './projectWindowState';

/** Which of a window's drawer sections are open, kept in the window's state. */
export interface DrawerSections {
  isOpen(section: ProjectWindowSection, initially?: boolean): boolean;
  setOpen(section: ProjectWindowSection, open: boolean): void;
}

export function useDrawerSections(
  state: ProjectWindowState, setState: Dispatch<SetStateAction<ProjectWindowState>>,
): DrawerSections {
  return useMemo(() => ({
    isOpen: (section, initially) => sectionOpen(state, section, initially),
    setOpen: (section, open) => setState((current) => withSectionOpen(current, section, open)),
  }), [state, setState]);
}

/** A drawer's body: a stack of sections that share its height. */
export function Drawer({ children }: { children: ReactNode }) {
  return <SectionStack>{children}</SectionStack>;
}

export interface DrawerSectionProps extends Omit<SectionProps, 'open' | 'defaultOpen' | 'onOpenChange'> {
  sections: DrawerSections;
  id: ProjectWindowSection;
  /** Whether the section is open before the user has folded or opened it. */
  initially?: boolean;
  /** Holds the section open regardless of its kept state, as a search does. */
  forceOpen?: boolean;
}

export function DrawerSection({ sections, id, initially = true, forceOpen, ...props }: DrawerSectionProps) {
  return <Section {...props} open={forceOpen || sections.isOpen(id, initially)}
    onOpenChange={(open) => sections.setOpen(id, open)} />;
}

const inspectorField = [
  'input:not([disabled]):not([type="hidden"])', 'textarea:not([disabled])', 'select:not([disabled])',
  '[role="combobox"]:not([aria-disabled="true"])',
].map((field) => `[data-studio-focus-region="inspector"] ${field}`).join(', ');

/** Return on a drawer row with nothing inside it: focus moves to the inspector's first field once the inspector shows it. */
export function focusInspectorField(frames = 30) {
  const field = document.querySelector<HTMLElement>(inspectorField);
  if (field) field.focus();
  else if (frames > 0) requestAnimationFrame(() => focusInspectorField(frames - 1));
}
