import {
  ReactFlow, getNodesBounds, getViewportForBounds, useReactFlow, useStore, useStoreApi,
  type ReactFlowProps, type Rect,
} from '@xyflow/react';
import {
  useCallback, useEffect, useImperativeHandle, useRef, useState,
  type KeyboardEvent, type ReactNode, type Ref,
} from 'react';
import { createPortal } from 'react-dom';
import { studioName } from '@/components/activity/studioVocabulary';
import { Dialog } from '@/components/ui/Dialog';
import { bindStudioCommand, commandAttributes } from '@/features/projectWindow/studioCommands';

export interface CanvasViewport {
  fitView(): void;
  zoomIn(): void;
  zoomOut(): void;
}

export function canvasViewportCommands(viewport: () => CanvasViewport | null, enabled: boolean) {
  return (['fitView', 'zoomIn', 'zoomOut'] as const).map((id) =>
    bindStudioCommand(id, () => viewport()?.[id](), { enabled }));
}

export type CanvasViewportCommand = ReturnType<typeof canvasViewportCommands>[number];

type Props = ReactFlowProps & {
  /** HTML documents supply their own bounds in canvas coordinates. */
  contentBounds?: Rect | null;
  fitReady?: boolean;
  fitNewContent?: boolean;
  lockAutoFit?: boolean;
  viewportRef?: Ref<CanvasViewport>;
  navigateContent?: boolean;
  children?: ReactNode;
};

const directions = {
  ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1],
} as const;

function adjacentControl(root: HTMLElement, current: HTMLElement, [dx, dy]: readonly [number, number]) {
  const origin = current.getBoundingClientRect();
  const cx = origin.left + origin.width / 2;
  const cy = origin.top + origin.height / 2;
  const column = current.closest('[data-canvas-column]');
  let nearest: HTMLElement | undefined;
  let score = Infinity;
  for (const item of root.querySelectorAll<HTMLElement>('[data-canvas-focus]')) {
    if (item === current || !item.getClientRects().length) continue;
    if (column && (item.closest('[data-canvas-column]') === column) !== Boolean(dy)) continue;
    const bounds = item.getBoundingClientRect();
    const x = bounds.left + bounds.width / 2 - cx;
    const y = bounds.top + bounds.height / 2 - cy;
    const forward = dx * x + dy * y;
    if (forward <= 1) continue;
    const distance = forward + 2 * Math.abs(dy * x - dx * y);
    if (distance < score) {
      nearest = item;
      score = distance;
    }
  }
  return nearest;
}

/** The shared mounted canvas, with a viewport independent of the region's size. */
export function StudioCanvas({
  contentBounds, fitReady = true, fitNewContent = false, lockAutoFit = false, viewportRef, navigateContent = false,
  children, onMove, minZoom = 0.5, maxZoom = 2, ...flow
}: Props) {
  const { viewportInitialized, getViewport, setViewport, zoomIn, zoomOut } = useReactFlow();
  const store = useStoreApi();
  const { width, height } = useStore((state) => ({ width: state.width, height: state.height }),
    (a, b) => a.width === b.width && a.height === b.height);
  const nodesBounds = useStore(({ nodeLookup }) => {
    if (contentBounds !== undefined || !fitReady) return null;
    const nodes = [...nodeLookup.values()];
    return nodes.length && nodes.every((node) => !!node.measured.width && !!node.measured.height)
      ? getNodesBounds(nodes) : null;
  }, (a, b) => a?.x === b?.x && a?.y === b?.y && a?.width === b?.width && a?.height === b?.height);
  const bounds = contentBounds === undefined ? nodesBounds : contentBounds;
  const boundsKey = bounds && [bounds.x, bounds.y, bounds.width, bounds.height].join('/');
  const fitted = useRef<string>();
  const moved = useRef(false);
  const root = useRef<HTMLDivElement>(null);
  const revealFrame = useRef<number>();
  const [menu, setMenu] = useState<{ x: number; y: number }>();
  const fit = useCallback(() => {
    const liveBounds = contentBounds === undefined && !fitReady
      ? getNodesBounds([...store.getState().nodeLookup.values()]) : bounds;
    if (!liveBounds || !viewportInitialized || !width || !height) return;
    void setViewport(getViewportForBounds(liveBounds, width, height, minZoom, fitNewContent ? maxZoom : Math.min(1, maxZoom), 0.14));
  }, [bounds, contentBounds, fitReady, store, viewportInitialized, width, height, minZoom, fitNewContent, maxZoom, setViewport]);
  const viewport = {
    fitView: () => { moved.current = true; fit(); },
    zoomIn: () => { moved.current = true; void zoomIn(); },
    zoomOut: () => { moved.current = true; void zoomOut(); },
  };
  useImperativeHandle(viewportRef, () => viewport);
  useEffect(() => {
    if (lockAutoFit) moved.current = true;
    if (!fitReady || !bounds || !boundsKey || !viewportInitialized || !width || !height || moved.current) return;
    if (fitted.current === boundsKey || (fitted.current && !fitNewContent)) return;
    const frame = requestAnimationFrame(() => {
      if (moved.current) return;
      fitted.current = boundsKey;
      if (fitNewContent || bounds.x < 0 || bounds.y < 0 || bounds.width + bounds.x > width ||
          bounds.height + bounds.y > height) fit();
    });
    return () => cancelAnimationFrame(frame);
  }, [fit, boundsKey, bounds, fitReady, viewportInitialized, width, height, fitNewContent, lockAutoFit]);
  useEffect(() => () => {
    if (revealFrame.current !== undefined) cancelAnimationFrame(revealFrame.current);
  }, []);

  function reveal(control: HTMLElement) {
    const region = root.current?.getBoundingClientRect();
    if (!region?.width || !region.height) return;
    const rect = control.getBoundingClientRect();
    const x = rect.left < region.left + 8 ? region.left + 8 - rect.left
      : Math.min(0, region.right - 8 - rect.right);
    const y = rect.top < region.top + 8 ? region.top + 8 - rect.top
      : Math.min(0, region.bottom - 8 - rect.bottom);
    if (x || y) {
      moved.current = true;
      const current = getViewport();
      void setViewport({ ...current, x: current.x + x, y: current.y + y });
    }
  }
  function navigate(event: KeyboardEvent<HTMLDivElement>) {
    if (event.defaultPrevented || !(event.target instanceof HTMLElement)) return;
    if (event.key === 'ContextMenu' || (event.key === 'F10' && event.shiftKey)) {
      const bounds = event.target.getBoundingClientRect();
      event.preventDefault();
      setMenu({ x: bounds.left, y: bounds.bottom });
      return;
    }
    const direction = directions[event.key as keyof typeof directions];
    if (!direction || !event.target.matches('[data-canvas-focus]') || event.metaKey || event.ctrlKey || event.altKey) return;
    const next = adjacentControl(event.currentTarget, event.target, direction);
    event.preventDefault();
    next?.focus({ preventScroll: true });
  }
  return <div ref={root} className="relative h-full min-h-0 min-w-0 flex-1 overflow-hidden"
    onFocusCapture={navigateContent ? (event) => {
      const control = event.target;
      if (!control.closest('[data-canvas-item]')) return;
      if (revealFrame.current !== undefined) cancelAnimationFrame(revealFrame.current);
      revealFrame.current = requestAnimationFrame(() => {
        revealFrame.current = undefined;
        if (document.activeElement === control) reveal(control);
      });
    } : undefined}
    onKeyDown={navigateContent ? navigate : undefined}
    onContextMenu={navigateContent ? (event) => {
      event.preventDefault();
      setMenu({ x: event.clientX, y: event.clientY });
    } : undefined}>
    <ReactFlow nodesDraggable={false} nodesConnectable={false} deleteKeyCode={null}
      {...flow} minZoom={minZoom} maxZoom={maxZoom} proOptions={{ hideAttribution: true }}
      onMove={(event, camera) => { if (event) moved.current = true; onMove?.(event, camera); }}>
      {children}
    </ReactFlow>
    {menu && createPortal(
      <Dialog ariaLabel={studioName('pipelineMap')} onClose={() => setMenu(undefined)} manageFocus
        className="fixed inset-0">
        <div role="menu" aria-label={studioName('pipelineMap')}
          className="fixed grid w-36 rounded-shell-control border border-shell-divider bg-shell-chrome p-1 text-sm text-shell-foreground shadow-lg"
          style={{ left: Math.max(0, Math.min(menu.x, window.innerWidth - 152)),
            top: Math.max(0, Math.min(menu.y, window.innerHeight - 104)) }}
          onKeyDown={(event) => {
            if (event.key === 'Escape') {
              event.preventDefault();
              event.stopPropagation();
              setMenu(undefined);
              return;
            }
            if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
            const items = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role=menuitem]')];
            const index = items.indexOf(document.activeElement as HTMLButtonElement);
            event.preventDefault();
            items[(index + (event.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length]?.focus();
          }}>
          {canvasViewportCommands(() => viewport, !!bounds).map((command) =>
            <button key={command.id} type="button" role="menuitem" {...commandAttributes(command)}
              disabled={!command.enabled} className="rounded px-2 py-1 text-left hover:bg-shell-hover"
              onClick={() => { command.invoke(); setMenu(undefined); }}>{command.label}</button>)}
        </div>
      </Dialog>, document.body,
    )}
  </div>;
}
