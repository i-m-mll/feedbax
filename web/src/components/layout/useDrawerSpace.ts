import { useLayoutEffect, useRef, useState } from 'react';

export const WORKSPACE_MIN_WIDTH = 200;

export function useDrawerSpace() {
  const ref = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(() => window.innerWidth);
  useLayoutEffect(() => {
    const element = ref.current!;
    setWidth(element.clientWidth);
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return { ref, width };
}
