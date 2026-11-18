import { Scan, ZoomIn, ZoomOut, type LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';
import { IconButton } from '@/components/ui/Button';
import { commandAttributes } from '@/features/projectWindow/studioCommands';
import type { CanvasViewportCommand } from './StudioCanvas';
import './canvasStrip.css';

export const canvasViewportIcons: Record<CanvasViewportCommand['id'], LucideIcon> = {
  fitView: Scan, zoomIn: ZoomIn, zoomOut: ZoomOut,
};

/** A canvas with its control strip: the view's tools at the top, the viewport at the bottom. */
export function CanvasStrip({ tools, viewport, children }: {
  tools?: ReactNode;
  viewport: readonly CanvasViewportCommand[];
  children: ReactNode;
}) {
  return <div className="studio-canvas">
    <div role="toolbar" aria-orientation="vertical" className="studio-canvas-strip">
      {tools}
      <div className="studio-canvas-strip__gap" />
      {viewport.map((command) => {
        const Icon = canvasViewportIcons[command.id];
        return <IconButton key={command.id} label={command.label} icon={<Icon />} {...commandAttributes(command, 'right')}
          disabled={!command.enabled} onClick={command.invoke} />;
      })}
    </div>
    <div className="studio-canvas-surface">{children}</div>
  </div>;
}
