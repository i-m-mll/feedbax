import { useMemo, type Dispatch, type ReactNode, type SetStateAction } from 'react';
import { Section, type SectionProps } from '@/components/ui/Section';
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
  return <div data-drawer-sections className="flex h-full min-h-0 flex-col">{children}</div>;
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
