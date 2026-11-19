/** How long a scrollbar stays shown after its region last scrolled. */
const linger = 700;

/**
 * Marks every scrolling element `data-scrolling` while it scrolls, from any cause, and briefly after; the stylesheet
 * shows overlay scrollbars only then and while the pointer is over the region.
 */
export function showScrollbarsWhileScrolling() {
  const timers = new WeakMap<Element, number>();
  document.addEventListener('scroll', (event) => {
    const element = event.target instanceof Element ? event.target : document.documentElement;
    window.clearTimeout(timers.get(element));
    element.setAttribute('data-scrolling', '');
    timers.set(element, window.setTimeout(() => element.removeAttribute('data-scrolling'), linger));
  }, { capture: true, passive: true });
}
