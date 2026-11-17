import { useSyncExternalStore } from 'react';

export function useRootAttribute<Value extends string>(name: string) {
  return useSyncExternalStore((notify) => {
    const observer = new MutationObserver(notify);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: [name] });
    return () => observer.disconnect();
  }, () => document.documentElement.getAttribute(name) as Value);
}
