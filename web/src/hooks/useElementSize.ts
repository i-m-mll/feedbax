import { useLayoutEffect, useRef, useState } from 'react';

export function useElementSize<T extends Element>(observe = true) {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element || !observe) return;
    const measure = (width: number, height: number) => {
      const next = { width: Math.round(width), height: Math.round(height) };
      setSize((previous) => next.width === previous.width && next.height === previous.height ? previous : next);
    };
    measure(element.clientWidth, element.clientHeight);
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) measure(entry.contentRect.width, entry.contentRect.height);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [observe]);
  return { ref, ...size };
}
