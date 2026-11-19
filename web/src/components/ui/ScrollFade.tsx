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

  return (
    <div
      ref={setRef}
      className={clsx('studio-scroll-fade min-h-0 min-w-0 overscroll-contain', overflow[axis], className)}
      onScroll={(event) => { mark(event.currentTarget); onScroll?.(event); }}
      {...props}
    />
  );
});
