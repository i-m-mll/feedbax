import { useLayoutEffect, useRef, type MouseEvent } from 'react';
import type { StudioWindowAction } from './studioNativeMenu';

async function windowAction(action: StudioWindowAction) {
  try {
    await window.__FEEDBAX_STUDIO_MENU__?.windowAction?.(action);
  } catch (error) {
    console.error('Studio title bar action failed', error);
  }
}

function dragSurface(event: MouseEvent<HTMLElement>): boolean {
  return event.button === 0 && event.target instanceof Element &&
    event.target.hasAttribute('data-studio-drag-region');
}

export function useStudioTitleBar(title: string) {
  const click = useRef<{ target: EventTarget; x: number; y: number }>();
  const publishingTitle = useRef(Promise.resolve());

  useLayoutEffect(() => {
    if (!window.__FEEDBAX_STUDIO_MENU__?.windowAction) return;
    publishingTitle.current = publishingTitle.current.then(() => windowAction({ kind: 'title', title }));
  }, [title]);

  return {
    onMouseDown(event: MouseEvent<HTMLElement>) {
      click.current = undefined;
      if (!window.__FEEDBAX_STUDIO_MENU__?.windowAction || !dragSurface(event)) return;
      if (event.detail === 1) {
        event.preventDefault();
        void windowAction({ kind: 'drag' });
      } else if (event.detail === 2) {
        click.current = { target: event.target, x: event.clientX, y: event.clientY };
      }
    },
    onMouseUp(event: MouseEvent<HTMLElement>) {
      const start = click.current;
      click.current = undefined;
      if (start && dragSurface(event) && event.detail === 2 && start.target === event.target &&
          start.x === event.clientX && start.y === event.clientY) {
        void windowAction({ kind: 'zoom' });
      }
    },
  };
}
