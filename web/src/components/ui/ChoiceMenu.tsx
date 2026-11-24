import { appTip } from '@window-shell/browser/tooltip';
import { useCallback, useState, type ReactNode } from 'react';
import { Menu, MenuContent, MenuItem } from './Menu';

/** An entry of a choice menu: a short label, and a tip saying what choosing it affects. */
export interface Choice {
  readonly id: string;
  readonly label: string;
  readonly tip: string;
}

/** Asks for one of `choices` in a menu at `anchor`; Escape or a click outside answers `undefined`. */
export type AskChoice = (anchor: Element, label: string, choices: readonly Choice[]) => Promise<string | undefined>;

const editedObject = 'td, th, [role="treeitem"], [role="row"], .field-row, .react-flow__node, .react-flow__edge';

/**
 * The field, cell, row or canvas element an edit was made on: that of the element focused when it was committed, or, for an edit
 * committed from a menu, that of the element focus returned to; else `otherwise`, such as a canvas's selected element.
 */
export function editedElement(committedFrom: Element | null, otherwise?: Element | null): Element {
  for (const element of [committedFrom, document.activeElement]) {
    const object = element?.isConnected ? element.closest(editedObject) : null;
    if (object) return object;
  }
  if (otherwise?.isConnected) return otherwise;
  throw new Error('The edit has no field, cell, row or canvas element to ask at.');
}

/**
 * The one menu that asks a pending command a question at the element it was made on. The window draws `menu` and
 * passes `ask` to its commands.
 */
export function useChoiceMenu() {
  const [asking, setAsking] = useState<{
    anchor: Element; label: string; choices: readonly Choice[]; answer(id: string | undefined): void;
  }>();
  const ask = useCallback<AskChoice>((anchor, label, choices) => new Promise((resolve) => {
    let answered = false;
    const from = document.activeElement;
    setAsking({ anchor, label, choices, answer: (id) => {
      if (answered) return;
      answered = true;
      setAsking(undefined);
      // The menu leaves as it is answered, so focus goes back to where the edit was made.
      if (from instanceof HTMLElement && from.isConnected) from.focus();
      resolve(id);
    } });
  }), []);
  const menu: ReactNode = asking && <Menu open modal={false} onOpenChange={(open) => { if (!open) asking.answer(undefined); }}>
    <MenuContent anchor={asking.anchor} aria-label={asking.label}>
      {asking.choices.map((choice) => <MenuItem key={choice.id} aria-description={choice.tip} {...appTip(choice.tip, 'right')}
        onClick={() => asking.answer(choice.id)}>{choice.label}</MenuItem>)}
    </MenuContent>
  </Menu>;
  return { ask, menu };
}
