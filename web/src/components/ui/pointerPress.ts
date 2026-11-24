/** A pointer position in client coordinates. */
export interface Point {
  x: number;
  y: number;
}

/** A press that has become a drag: it follows the pointer until it is released or cancelled. */
export interface PressDrag {
  move(at: Point): void;
  release(at: Point): void;
  cancel(): void;
}

/** How far the pointer moves, by default, before a press is a drag. */
export const pressSlop = 3;

const pointOf = (event: { clientX: number; clientY: number }): Point => ({ x: event.clientX, y: event.clientY });

/**
 * Follows the primary-button press `event` begins. Once the pointer has moved `slop` from where it was pressed (as
 * `distance` measures), `begin` is called with the pointer's position; the drag it returns follows the pointer until
 * release, and the click that release would make is swallowed. A press that never moves so far, or whose `begin`
 * returns nothing, is left to the element's own handlers.
 */
export function followPress(
  event: { pointerId: number; button: number; clientX: number; clientY: number },
  begin: (at: Point, from: Point) => PressDrag | undefined,
  { slop = pressSlop, distance = (from: Point, at: Point) => Math.hypot(at.x - from.x, at.y - from.y) } = {},
) {
  if (event.button !== 0) return;
  const pointer = event.pointerId;
  const from = pointOf(event);
  let drag: PressDrag | undefined;
  const onMove = (moved: PointerEvent) => {
    if (moved.pointerId !== pointer) return;
    const at = pointOf(moved);
    if (!drag) {
      if (distance(from, at) < slop) return;
      drag = begin(at, from);
      if (!drag) return finish();
      document.documentElement.dataset.pressDrag = '';
      getSelection()?.removeAllRanges();
    }
    drag.move(at);
  };
  const onUp = (released: PointerEvent) => {
    if (released.pointerId !== pointer) return;
    const done = drag;
    finish();
    if (!done) return;
    swallowClick();
    done.release(pointOf(released));
  };
  const onCancel = (cancelled: PointerEvent) => {
    if (cancelled.pointerId !== pointer) return;
    const done = drag;
    finish();
    done?.cancel();
  };
  function finish() {
    delete document.documentElement.dataset.pressDrag;
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    window.removeEventListener('pointercancel', onCancel);
  }
  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
  window.addEventListener('pointercancel', onCancel);
}

/** The click a drag's release makes, which would otherwise reach the element pressed or the one released over. */
function swallowClick() {
  const swallow = (click: MouseEvent) => {
    click.stopPropagation();
    click.preventDefault();
  };
  window.addEventListener('click', swallow, { capture: true, once: true });
  setTimeout(() => window.removeEventListener('click', swallow, { capture: true }));
}
