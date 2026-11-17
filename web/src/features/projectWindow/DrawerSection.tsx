import { createContext, useContext, type Dispatch, type ReactNode, type SetStateAction } from 'react';
import { Section, type SectionProps } from '@/components/ui/Section';
import { sectionOpen, withSectionOpen, type ProjectWindowSection, type ProjectWindowState } from './projectWindowState';

type WindowSections = readonly [ProjectWindowState, Dispatch<SetStateAction<ProjectWindowState>>];

const WindowSectionsContext = createContext<WindowSections | null>(null);

export function WindowSectionsProvider({ state, onStateChange, children }: {
  state: ProjectWindowState;
  onStateChange: Dispatch<SetStateAction<ProjectWindowState>>;
  children: ReactNode;
}) {
  return <WindowSectionsContext.Provider value={[state, onStateChange]}>{children}</WindowSectionsContext.Provider>;
}

/** Whether a drawer section is open, kept in the window's state under `<view>.<name>`, and its setter. */
export function useDrawerSection(section: ProjectWindowSection, defaultOpen = true) {
  const window = useContext(WindowSectionsContext);
  if (!window) throw new Error('A drawer section is used outside a WindowSectionsProvider');
  const [state, setState] = window;
  return [
    sectionOpen(state, section, defaultOpen),
    (open: boolean) => setState((current) => withSectionOpen(current, section, open)),
  ] as const;
}

/** A drawer's section whose fold is kept in the window. `revealed` shows it open without changing that. */
export function DrawerSection({ section, defaultOpen = true, revealed = false, ...props }: Omit<SectionProps, 'open' | 'onOpenChange'> & {
  section: ProjectWindowSection;
  revealed?: boolean;
}) {
  const [open, setOpen] = useDrawerSection(section, defaultOpen);
  return <Section {...props} open={revealed || open} onOpenChange={setOpen} />;
}

/** A drawer: a column of sections that share its height. */
export function DrawerSections({ children }: { children: ReactNode }) {
  return <div className="flex h-full min-h-0 min-w-0 flex-col text-sm">{children}</div>;
}
