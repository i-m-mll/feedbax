import { PreviewCard } from '@base-ui/react/preview-card';
import clsx from 'clsx';
import { useEffect, useMemo, useRef, useState, type ComponentProps, type ReactNode } from 'react';
import { popupSurface } from './styles';
import { usePopupLayer } from './popupLayer';

export type HoverCardHandle<Payload> = PreviewCard.Handle<Payload>;
export const createHoverCardHandle = PreviewCard.createHandle;

/** How long after the last scroll a card stays shut, so rows passing under a still pointer do not open one. */
const scrollQuiet = 200;
/** How long the pointer rests on a trigger before its card opens. */
const restDelay = 400;

/** Whether anything scrolled or dragged recently enough that no card opens; one document holds one pointer. */
const quiet = { dragging: false, scrolledAt: -Infinity };
const blocked = () => quiet.dragging || performance.now() - quiet.scrolledAt < scrollQuiet;

export interface HoverCardTriggerProps<Payload> extends Omit<ComponentProps<typeof PreviewCard.Trigger>, 'handle' | 'payload' | 'delay' | 'closeDelay'> {
  handle: HoverCardHandle<Payload>;
  payload: Payload;
}

/**
 * Opens its card after the pointer rests on it for `restDelay`, or when it takes keyboard focus. A rest after moving
 * on it counts too, so a card kept shut while the trigger scrolled under the pointer opens once the pointer moves.
 */
export function HoverCardTrigger<Payload>(props: HoverCardTriggerProps<Payload>) {
  const { handle, onPointerMove, onPointerLeave } = props;
  const rest = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(rest.current), []);
  return <PreviewCard.Trigger {...props} delay={restDelay} closeDelay={0}
    onPointerMove={(event) => {
      onPointerMove?.(event);
      clearTimeout(rest.current);
      if (event.pointerType !== 'mouse' || handle.isOpen) return;
      const trigger = event.currentTarget;
      rest.current = setTimeout(() => {
        if (!handle.isOpen && !blocked() && trigger.matches(':hover')) handle.open(trigger.id);
      }, restDelay);
    }}
    onPointerLeave={(event) => {
      onPointerLeave?.(event);
      clearTimeout(rest.current);
    }} />;
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
  const [trigger, setTrigger] = useState<Element | null>(null);
  useEffect(() => {
    const close = () => actions.current?.close();
    const scrolled = () => { quiet.scrolledAt = performance.now(); close(); };
    const dragged = () => { quiet.dragging = true; close(); };
    const dropped = () => { quiet.dragging = false; };
    document.addEventListener('scroll', scrolled, { capture: true, passive: true });
    document.addEventListener('dragstart', dragged, true);
    document.addEventListener('dragend', dropped, true);
    return () => {
      document.removeEventListener('scroll', scrolled, { capture: true });
      document.removeEventListener('dragstart', dragged, true);
      document.removeEventListener('dragend', dropped, true);
    };
  }, []);
  const region = useRef(beside);
  region.current = beside;
  const anchor = useMemo(() => trigger && {
    contextElement: trigger,
    getBoundingClientRect: () => {
      const { left, width } = (region.current?.(trigger) ?? trigger).getBoundingClientRect();
      const { top, height } = trigger.getBoundingClientRect();
      return new DOMRect(left, top, width, height);
    },
  }, [trigger]);
  return <PreviewCard.Root handle={handle} actionsRef={actions} onOpenChange={(open, details) => {
    if (open && blocked()) details.cancel();
    else if (open && details.trigger) setTrigger(details.trigger);
  }}>
    {({ payload }) => payload === undefined ? null : <PreviewCard.Portal container={layer}>
      <PreviewCard.Positioner className="pointer-events-none z-50 outline-none" side="right" align="start" sideOffset={4}
        collisionPadding={8} anchor={anchor}>
        <PreviewCard.Popup className={clsx(popupSurface, 'max-w-80 p-3', className)}>{children(payload)}</PreviewCard.Popup>
      </PreviewCard.Positioner>
    </PreviewCard.Portal>}
  </PreviewCard.Root>;
}
