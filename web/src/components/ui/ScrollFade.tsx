import clsx from 'clsx';
import { forwardRef, useCallback, useLayoutEffect, useRef, type HTMLAttributes } from 'react';

export type ScrollAxis = 'x' | 'y' | 'both';

export interface ScrollFadeProps extends HTMLAttributes<HTMLDivElement> {
  axis?: ScrollAxis;
}

const overflow: Record<ScrollAxis, string> = {
  x: 'overflow-x-auto overflow-y-hidden',
  y: 'overflow-y-auto overflow-x-hidden',
  both: 'overflow-auto',
};

function mark(element: HTMLElement) {
  const { scrollTop, scrollLeft, scrollHeight, scrollWidth, clientHeight, clientWidth } = element;
  element.toggleAttribute('data-overflow-top', scrollTop > 0.5);
  element.toggleAttribute('data-overflow-bottom', scrollHeight - clientHeight - scrollTop > 0.5);
  element.toggleAttribute('data-overflow-left', scrollLeft > 0.5);
  element.toggleAttribute('data-overflow-right', scrollWidth - clientWidth - scrollLeft > 0.5);
}

const revealDuration = 200;
/** The longest an action's own update takes to land; content after that was not brought in by the action. */
const revealLifetime = 10_000;
/** How long after its first content an action's update keeps arriving, such as a group's values after its schema. */
const settleTime = 500;
const scrollKeys = new Set(['PageUp', 'PageDown', 'Home', 'End', 'ArrowUp', 'ArrowDown', ' ']);
/** Popups an action in a region opens: choosing in one continues that action. */
const popup = '[role="listbox"], [role="menu"], [role="dialog"], [role="alertdialog"]';
/** Content that appears without the user asking for it. */
const unasked = '[role="alert"], [role="status"], [role="progressbar"]';

const interactive = 'button, input, select, textarea, a[href], [role="combobox"], [role="checkbox"], [role="switch"], [role="option"], [tabindex]';

/** The control an event acted on, and the `aria-labelledby` that names it, which outlives a remounted control. */
function controlOf(target: Element, node: HTMLElement) {
  const found = target.closest(interactive);
  const control = found && node.contains(found) && found !== node ? found : target;
  return { control, labelledBy: control.getAttribute('aria-labelledby') };
}

/** The trigger's elements: its control (found again by its label when it was remounted), then its labels. */
function triggerParts({ control, labelledBy }: ReturnType<typeof controlOf>, node: HTMLElement): Element[] {
  const current = control.isConnected ? control
    : labelledBy && node.querySelector(interactive.split(', ').map((control) => `${control}[aria-labelledby="${CSS.escape(labelledBy)}"]`).join());
  if (!current) return [];
  const labels = [...labelledBy?.split(/\s+/).map((id) => document.getElementById(id)) ?? [],
    ...(current as HTMLInputElement).labels ?? []];
  return [current, ...labels.filter((label): label is HTMLElement => !!label && node.contains(label))];
}

/**
 * Shows what a user's action in `node` brings in below its trigger (the fields of a newly chosen class, an unfolded
 * group, a new list item) by the least movement: until the content ends at the bottom of the clear area, but never
 * past putting the trigger, label included, at its top. Nothing else moves the region: removed content, content that
 * changes in place, and content nobody asked for. The movement is a fixed-length tween that any input stops, and a
 * jump under reduced motion.
 */
function revealEdits(node: HTMLElement) {
  let trigger: ReturnType<typeof controlOf> & { until: number } | null = null;
  let revealed = -Infinity;
  let tween = 0;
  const stop = () => { cancelAnimationFrame(tween); tween = 0; };
  const disarm = () => { trigger = null; stop(); };
  const act = (event: Event) => {
    stop();
    const target = event.target;
    const inside = target instanceof Element && node.contains(target);
    if (event instanceof KeyboardEvent && scrollKeys.has(event.key) && (!inside || target === node ||
        !(target as Element).matches('input, textarea, select, [contenteditable], [role="combobox"], [role="listbox"], [role="option"]'))) {
      disarm();
    } else if (inside && target !== node) {
      trigger = { ...controlOf(target, node), until: performance.now() + revealLifetime };
      revealed = -Infinity;
    } else if (!(target instanceof Element && target.closest(popup))) {
      disarm();
    }
  };
  const offset = (top: number) => top - node.getBoundingClientRect().top + node.scrollTop;
  const scrollTo = (top: number) => {
    stop();
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      node.scrollTop = top;
      return;
    }
    const from = node.scrollTop;
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / revealDuration);
      node.scrollTop = from + (top - from) * (1 - (1 - t) ** 3);
      tween = t < 1 ? requestAnimationFrame(step) : 0;
    };
    step(start);
  };
  const changes = new MutationObserver((records) => {
    if (trigger && performance.now() > trigger.until) trigger = null;
    if (!trigger) return;
    const parts = triggerParts(trigger, node);
    if (!parts.length) return;
    const last = parts.reduce((a, b) => a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? b : a);
    const floor = Math.max(...parts.map((part) => offset(part.getBoundingClientRect().bottom))) + 1;
    const before = revealed;
    for (const record of records) {
      const shown = record.type === 'childList' ? [...record.addedNodes]
        : record.target instanceof Element && !record.target.hasAttribute('hidden') ? [record.target] : [];
      for (const item of shown) {
        if (!(item instanceof Element) || item.closest(unasked)) continue;
        const extent = document.createRange();
        if (item.contains(last)) extent.setStartAfter(last);
        else if (last.compareDocumentPosition(item) & Node.DOCUMENT_POSITION_FOLLOWING) extent.setStartBefore(item);
        else continue;
        extent.setEndAfter(item);
        const bottom = offset(extent.getBoundingClientRect().bottom);
        if (bottom > floor) revealed = Math.max(revealed, bottom);
      }
    }
    if (revealed === before) return;
    trigger.until = Math.min(trigger.until, performance.now() + settleTime);
    const { scrollPaddingTop, scrollPaddingBottom } = getComputedStyle(node);
    const triggerTop = Math.min(...parts.map((part) => offset(part.getBoundingClientRect().top)));
    const top = Math.min(revealed + parseFloat(scrollPaddingBottom) - node.clientHeight, triggerTop - parseFloat(scrollPaddingTop));
    if (top > node.scrollTop + 0.5) scrollTo(Math.min(top, node.scrollHeight - node.clientHeight));
  });
  changes.observe(node, { childList: true, subtree: true, attributes: true, attributeFilter: ['hidden'] });
  const listeners = [['pointerdown', act], ['keydown', act], ['wheel', disarm], ['touchstart', disarm]] as const;
  for (const [type, listener] of listeners) document.addEventListener(type, listener, { capture: true, passive: true });
  return () => {
    disarm();
    changes.disconnect();
    for (const [type, listener] of listeners) document.removeEventListener(type, listener, { capture: true });
  };
}

/** A scrolling region whose content fades at each edge it overflows. */
export const ScrollFade = forwardRef<HTMLDivElement, ScrollFadeProps>(function ScrollFade(
  { axis = 'y', className, onScroll, ...props }, ref,
) {
  const element = useRef<HTMLDivElement | null>(null);
  const setRef = useCallback((node: HTMLDivElement | null) => {
    element.current = node;
    if (typeof ref === 'function') ref(node);
    else if (ref) ref.current = node;
  }, [ref]);

  useLayoutEffect(() => {
    const node = element.current;
    if (!node) return;
    const update = () => mark(node);
    const sizes = new ResizeObserver(update);
    const children = new MutationObserver(() => { observeChildren(); update(); });
    const observeChildren = () => {
      sizes.disconnect();
      children.disconnect();
      sizes.observe(node);
      children.observe(node, { childList: true });
      const observe = (parent: Element) => {
        for (const child of parent.children) {
          if (getComputedStyle(child).display !== 'contents') sizes.observe(child);
          else {
            children.observe(child, { childList: true });
            observe(child);
          }
        }
      };
      observe(node);
    };
    observeChildren();
    update();
    return () => { sizes.disconnect(); children.disconnect(); };
  }, []);

  useLayoutEffect(() => {
    const node = element.current;
    if (!node || axis === 'x') return;
    return revealEdits(node);
  }, [axis]);

  return (
    <div
      ref={setRef}
      className={clsx('studio-scroll-fade min-h-0 min-w-0 overscroll-contain', overflow[axis], className)}
      onScroll={(event) => { mark(event.currentTarget); onScroll?.(event); }}
      {...props}
    />
  );
});
