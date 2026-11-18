import { useEffect, useLayoutEffect, useRef, type RefObject } from 'react';
import { installTitleBarDrag, shellHost } from '@window-shell/browser/titlebar';
import type { StudioWindowAction } from './studioNativeMenu';

async function windowAction(action: StudioWindowAction) {
  try {
    await window.__FEEDBAX_STUDIO_MENU__?.windowAction?.(action);
  } catch (error) {
    console.error('Studio title bar action failed', error);
  }
}

/** Publishes the window title to the native host and gives the bar the shell's drag and zoom. */
export function useStudioTitleBar(bar: RefObject<HTMLElement>, title: string) {
  const publishingTitle = useRef(Promise.resolve());

  useLayoutEffect(() => {
    if (!window.__FEEDBAX_STUDIO_MENU__?.windowAction) return;
    publishingTitle.current = publishingTitle.current.then(() => windowAction({ kind: 'title', title }));
  }, [title]);

  useEffect(() => {
    if (!bar.current || shellHost() !== 'native') return;
    return installTitleBarDrag(bar.current, {
      startDrag: () => void windowAction({ kind: 'drag' }),
      toggleZoom: () => void windowAction({ kind: 'zoom' }),
    });
  }, [bar]);
}
