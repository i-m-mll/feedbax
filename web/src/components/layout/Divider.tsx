import type { PaneLayout } from '@window-shell/browser/layout';
import { DIVIDER_HEIGHT } from '@/stores/layoutStore';
import { ResizeSeparator } from './ResizeSeparator';

export function Divider({ layout, resizable, onPreview, onResize }: {
  layout: PaneLayout;
  resizable: boolean;
  onPreview: (size: number | null) => void;
  onResize: (size: number) => void;
}) {
  if (!resizable) return <div className="border-t border-shell-divider" aria-hidden="true" />;
  return (
    <ResizeSeparator
      className="z-10 cursor-row-resize border-t border-shell-divider"
      style={{ height: DIVIDER_HEIGHT }}
      label="Resize bottom pane"
      controls="studio-bottom-pane"
      axis="y"
      direction={-1}
      layout={layout}
      onPreview={onPreview}
      onResize={onResize}
    />
  );
}
