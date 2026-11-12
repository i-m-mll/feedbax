import { useLayoutEffect, useRef, type CSSProperties } from 'react';
import type { PaneLayout } from '@window-shell/browser/layout';
import { installSeparator, separatorValues } from '@window-shell/browser/separator';

export function ResizeSeparator(props: {
  className: string;
  style?: CSSProperties;
  label: string;
  controls: string;
  axis: 'x' | 'y';
  direction: 1 | -1;
  layout: PaneLayout;
  onPreview: (size: number | null) => void;
  onResize: (size: number) => void;
}) {
  const element = useRef<HTMLDivElement>(null);
  const current = useRef(props);
  current.current = props;
  const { className, style, label, controls, axis, direction, layout } = props;
  useLayoutEffect(() => installSeparator(element.current!, {
    axis,
    direction,
    layout: () => current.current.layout,
    onPreview: (size) => current.current.onPreview(size),
    onResize: (size) => current.current.onResize(size),
  }), [axis, direction]);
  return <div ref={element} className={`shell-separator ${className}`} style={style}
    role="separator" tabIndex={0} aria-label={label} aria-controls={controls} {...separatorValues(layout, axis)} />;
}
