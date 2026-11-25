import { useLayoutEffect, useMemo, useRef, type Dispatch, type ReactNode, type RefObject, type SetStateAction } from 'react';
import { Section, SectionStack, type SectionProps } from '@/components/ui/Section';
import type { FieldKind } from '@/generated/studioContracts';
import {
  sectionOpen, withSectionOpen, type ProjectWindowSection, type ProjectWindowState,
} from './projectWindowState';
import { commandsRun } from './heldGestures';

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

/** A control that holds a value; a command in a field's header (`.field-row-actions`) is not one. */
const field = [
  'input:not([disabled]):not([type="hidden"]):not([readonly]):not([aria-hidden="true"])', 'textarea:not([disabled])',
  'select:not([disabled])', '[role="combobox"]:not([aria-disabled="true"])',
].map((item) => `${item}:not(.field-row-actions *)`).join(', ');
const inspectorField = field.split(', ').map((item) => `[data-studio-focus-region="inspector"] ${item}`).join(', ');

/** Return on a drawer row with nothing inside it: focus moves to the inspector's first field once the inspector shows it. */
export function focusInspectorField(frames = 30) {
  const found = document.querySelector<HTMLElement>(inspectorField);
  if (found) found.focus();
  else if (frames > 0) requestAnimationFrame(() => focusInspectorField(frames - 1));
}

/**
 * A request to put the keyboard on a field: the one whose `data-field` is `target`, or else the first with a
 * `data-field`; and, with `kind`, to put that field in that kind until it commits its text.
 */
export interface FieldFocusRequest {
  readonly key: string;
  readonly target?: string;
  readonly kind?: FieldKind;
}

/** Unfolds each folded section between `root` and `element`. */
function unfold(element: Element, root: Element) {
  for (let body = element.closest('[hidden]'); body && root.contains(body); body = body.parentElement?.closest('[hidden]') ?? null) {
    root.querySelector<HTMLElement>(`[aria-controls="${CSS.escape(body.id)}"][aria-expanded="false"]`)?.click();
  }
}

/**
 * Puts the keyboard on a field with its text read from the start: a text box focused by a command, not by the
 * pointer, would otherwise put the caret after its last character and scroll a long value's start out of view.
 */
export function focusField(field: HTMLElement) {
  field.focus();
  if (field instanceof HTMLInputElement && /^(text|search)$/.test(field.type)) {
    field.setSelectionRange(0, 0);
    field.scrollLeft = 0;
  }
}

/**
 * Puts the keyboard on the requested field of `root` as soon as `root` shows a field, unless a command has run since
 * the request, or the keyboard is already on a field of `root`: it then belongs to what that command opened, or to the
 * field being written.
 */
export function useFieldFocus(root: RefObject<HTMLElement>, request: FieldFocusRequest | undefined, onFocused: () => void) {
  const focused = useRef(onFocused);
  focused.current = onFocused;
  const target = request?.target;
  useLayoutEffect(() => {
    const element = root.current;
    if (!request || !element) return;
    const requested = commandsRun();
    const attempt = () => {
      const active = document.activeElement;
      if (commandsRun() !== requested || (active && element.contains(active) && active.matches(field))) {
        focused.current();
        return true;
      }
      // The named field, or without a name the step's own first field once no region before it is loading, as soon
      // as it can take the keyboard; the first field of any kind when the fields shown do not include it.
      const busy = element.querySelector('[aria-busy="true"]');
      const scope = target === undefined ? '[data-field]' : `[data-field="${CSS.escape(target)}"]`;
      const named = element.querySelector(scope);
      if (target === undefined && named && busy && busy.compareDocumentPosition(named) & Node.DOCUMENT_POSITION_FOLLOWING) return false;
      if (named) unfold(named, element);
      const shown = named === null && (element.querySelector('[data-field]') !== null || busy === null);
      const chosen = named || shown ? element.querySelector<HTMLElement>(named ? `${scope} :is(${field})` : field) : null;
      if (!chosen) return false;
      focusField(chosen);
      focused.current();
      return true;
    };
    if (attempt()) return;
    const observer = new MutationObserver(() => { if (attempt()) observer.disconnect(); });
    observer.observe(element, {
      childList: true, subtree: true, attributes: true, attributeFilter: ['disabled', 'readonly', 'aria-disabled', 'aria-busy', 'hidden'],
    });
    return () => observer.disconnect();
  }, [request?.key]);
}
