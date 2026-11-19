import { useLayoutEffect, useMemo, useRef, type Dispatch, type ReactNode, type RefObject, type SetStateAction } from 'react';
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

const field = [
  'input:not([disabled]):not([type="hidden"]):not([readonly]):not([aria-hidden="true"])', 'textarea:not([disabled])',
  'select:not([disabled])', '[role="combobox"]:not([aria-disabled="true"])',
].join(', ');
const inspectorField = field.split(', ').map((item) => `[data-studio-focus-region="inspector"] ${item}`).join(', ');

/** Return on a drawer row with nothing inside it: focus moves to the inspector's first field once the inspector shows it. */
export function focusInspectorField(frames = 30) {
  const found = document.querySelector<HTMLElement>(inspectorField);
  if (found) found.focus();
  else if (frames > 0) requestAnimationFrame(() => focusInspectorField(frames - 1));
}

/** A request to put the keyboard on a field: the one whose `data-field` is `target`, or else the first with a `data-field`. */
export interface FieldFocusRequest {
  readonly key: string;
  readonly target?: string;
}

/** Puts the keyboard on the requested field of `root` as soon as `root` shows a field. */
export function useFieldFocus(root: RefObject<HTMLElement>, request: FieldFocusRequest | undefined, onFocused: () => void) {
  const focused = useRef(onFocused);
  focused.current = onFocused;
  const target = request?.target;
  useLayoutEffect(() => {
    const element = root.current;
    if (!request || !element) return;
    const attempt = () => {
      // The named field, or without a name the step's own first field, once it can take the keyboard;
      // the first field of any kind when the fields shown do not include it.
      const scope = target === undefined ? '[data-field]' : `[data-field="${CSS.escape(target)}"]`;
      const named = element.querySelector(scope);
      const shown = named === null && (element.querySelector('[data-field]') !== null ||
        element.querySelector('[aria-busy="true"]') === null);
      const chosen = named || shown ? element.querySelector<HTMLElement>(named ? `${scope} :is(${field})` : field) : null;
      if (!chosen) return false;
      chosen.focus();
      focused.current();
      return true;
    };
    if (attempt()) return;
    const observer = new MutationObserver(() => { if (attempt()) observer.disconnect(); });
    observer.observe(element, {
      childList: true, subtree: true, attributes: true, attributeFilter: ['disabled', 'readonly', 'aria-disabled', 'aria-busy'],
    });
    return () => observer.disconnect();
  }, [request?.key]);
}
