import { PreviewCard } from '@base-ui/react/preview-card';
import clsx from 'clsx';
import { useEffect, useRef, type ComponentProps, type ReactNode } from 'react';
import { popupSurface } from './styles';
import { usePopupLayer } from './popupLayer';

export type HoverCardHandle<Payload> = PreviewCard.Handle<Payload>;
export const createHoverCardHandle = PreviewCard.createHandle;

/** How long after the last scroll a card stays shut, so rows passing under a still pointer do not open one. */
const scrollQuiet = 200;

export interface HoverCardTriggerProps<Payload> extends Omit<ComponentProps<typeof PreviewCard.Trigger>, 'handle' | 'payload' | 'delay' | 'closeDelay'> {
  handle: HoverCardHandle<Payload>;
  payload: Payload;
}

/** Opens its card after the pointer rests on it for 400 ms, or when it takes keyboard focus. */
export function HoverCardTrigger<Payload>(props: HoverCardTriggerProps<Payload>) {
  return <PreviewCard.Trigger {...props} delay={400} closeDelay={0} />;
}

export interface HoverCardProps<Payload> {
  handle: HoverCardHandle<Payload>;
  children: (payload: Payload) => ReactNode;
  /** The region the card stands beside, so that it never covers it; the trigger by default. */
  beside?: (trigger: Element) => Element | null;
  className?: string;
}

/** A card describing the trigger the pointer rests on; it stays shut while anything scrolls or drags, and lets the pointer through. */
export function HoverCard<Payload>({ handle, children, beside, className }: HoverCardProps<Payload>) {
  const layer = usePopupLayer();
  const actions = useRef<PreviewCard.Root.Actions>(null);
  const trigger = useRef<Element>();
  const blocked = useRef({ dragging: false, scrolledAt: -Infinity });
  useEffect(() => {
    const state = blocked.current;
    const close = () => actions.current?.close();
    const scrolled = () => { state.scrolledAt = performance.now(); close(); };
    const dragged = () => { state.dragging = true; close(); };
    const dropped = () => { state.dragging = false; };
    document.addEventListener('scroll', scrolled, { capture: true, passive: true });
    document.addEventListener('dragstart', dragged, true);
    document.addEventListener('dragend', dropped, true);
    return () => {
      document.removeEventListener('scroll', scrolled, { capture: true });
      document.removeEventListener('dragstart', dragged, true);
      document.removeEventListener('dragend', dropped, true);
    };
  }, []);
  const anchor = () => {
    const row = trigger.current;
    if (!row) return null;
    const region = beside?.(row) ?? row;
    return {
      contextElement: row,
      getBoundingClientRect: () => {
        const { left, width } = region.getBoundingClientRect();
        const { top, height } = row.getBoundingClientRect();
        return new DOMRect(left, top, width, height);
      },
    };
  };
  return <PreviewCard.Root handle={handle} actionsRef={actions} onOpenChange={(open, details) => {
    const { dragging, scrolledAt } = blocked.current;
    if (open && (dragging || performance.now() - scrolledAt < scrollQuiet)) details.cancel();
    else if (open) trigger.current = details.trigger;
  }}>
    {({ payload }) => payload === undefined ? null : <PreviewCard.Portal container={layer}>
      <PreviewCard.Positioner className="pointer-events-none z-50 outline-none" side="right" align="start" sideOffset={4}
        collisionPadding={8} anchor={anchor}>
        <PreviewCard.Popup className={clsx(popupSurface, 'max-w-80 p-3', className)}>{children(payload)}</PreviewCard.Popup>
      </PreviewCard.Positioner>
    </PreviewCard.Portal>}
  </PreviewCard.Root>;
}
