import { useState, type ReactNode } from 'react';
import { studioStep } from '@/components/activity/studioVocabulary';
import { Menu, MenuContent, MenuItem, type MenuContentProps } from '@/components/ui/Menu';

export const stepKinds = ['train', 'evaluate', 'analyze', 'report'] as const;
export type StepKind = typeof stepKinds[number];
/** The kinds that can be created at one place, each with its action. */
export type StepInsertion = Readonly<Partial<Record<StepKind, () => void>>>;

type Point = { x: number; y: number };
let origin: Element | Point | null = null;
let fromKeyboard = false;

// A command does not know which control invoked it; the event being dispatched does.
function recordOrigin(event: Event) {
  const target = event.target instanceof Element ? event.target : null;
  origin = event.type === 'contextmenu' && event instanceof MouseEvent && (event.clientX || event.clientY)
    ? { x: event.clientX, y: event.clientY }
    : target?.closest('[data-command-id]') ?? target;
  fromKeyboard = event.type === 'keydown';
  setTimeout(() => { origin = null; fromKeyboard = false; });
}

if (typeof document !== 'undefined') {
  for (const type of ['click', 'keydown', 'contextmenu']) document.addEventListener(type, recordOrigin, true);
}

function pointAnchor({ x, y }: Point) {
  return { getBoundingClientRect: () => DOMRect.fromRect({ x, y, width: 0, height: 0 }) };
}

interface Placement {
  anchor: NonNullable<MenuContentProps['anchor']>;
  align: NonNullable<MenuContentProps['align']>;
  returnFocus: Element | null;
  fromKeyboard: boolean;
}

function placement(from: Element | Point | null): Omit<Placement, 'fromKeyboard'> {
  if (from instanceof Element && from.isConnected) return { anchor: from, align: 'start', returnFocus: from };
  if (from && !(from instanceof Element)) return { anchor: pointAnchor(from), align: 'start', returnFocus: null };
  const bar = document.querySelector('.shell-frame__bar')?.getBoundingClientRect().bottom ?? 0;
  return { anchor: pointAnchor({ x: window.innerWidth / 2, y: bar }), align: 'center', returnFocus: null };
}

/** One menu of the four step kinds, anchored to whatever invoked it. */
export function useStepInsertMenu(): { open(insertion: StepInsertion): void; menu: ReactNode } {
  const [request, setRequest] = useState<{ insertion: StepInsertion } & Placement>();
  const menu = (
    <Menu open={!!request} onOpenChange={(open) => { if (!open) setRequest(undefined); }}
      onOpenChangeComplete={(open) => {
        // A menu opened from the keyboard starts on its first item, as one opened by its own trigger does.
        if (open && request?.fromKeyboard) {
          document.querySelector<HTMLElement>('[data-step-insert]:not([data-disabled])')?.focus();
        }
      }}>
      {request && <MenuContent anchor={request.anchor} align={request.align}
        finalFocus={request.returnFocus instanceof HTMLElement ? { current: request.returnFocus } : true}>
        {stepKinds.map((kind) => {
          const { Icon, tint, name } = studioStep(kind);
          return <MenuItem key={kind} data-step-insert={kind} glyph={<Icon aria-hidden="true" style={{ color: tint }} />}
            disabled={!request.insertion[kind]} onClick={request.insertion[kind]}>{name}</MenuItem>;
        })}
      </MenuContent>}
    </Menu>
  );
  return { open: (insertion) => setRequest({ insertion, fromKeyboard, ...placement(origin) }), menu };
}
