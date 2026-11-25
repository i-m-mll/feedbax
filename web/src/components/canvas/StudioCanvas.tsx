import {
  ReactFlow, getNodesBounds, getViewportForBounds, useReactFlow, useStore, useStoreApi,
  type ReactFlowProps, type Rect,
} from '@xyflow/react';
import {
  useCallback, useEffect, useId, useImperativeHandle, useRef,
  type KeyboardEvent, type MouseEvent, type ReactNode, type Ref,
} from 'react';
import { bindStudioCommand } from '@/features/projectWindow/studioCommands';

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
  /**
   * When the view is fitted to the content again, until the user moves it: `once`, when content first appears;
   * `overflow`, also whenever a change of content or region leaves content past the region's edge; `always`, whenever
   * the content or the region changes, zooming in as far as `maxZoom`.
   */
  refit?: 'once' | 'overflow' | 'always';
  /** Holds the view still while set, as for a gesture that places content where the pointer is; it refits after. */
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
  contentBounds, fitReady = true, refit = 'once', lockAutoFit = false, viewportRef, navigateContent = false,
  children, onMove, minZoom = 0.5, maxZoom = 2, zoomOnDoubleClick = true, ...flow
}: Props) {
  const flowId = useId();
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
  const fit = useCallback(() => {
    const liveBounds = contentBounds === undefined && !fitReady
      ? getNodesBounds([...store.getState().nodeLookup.values()]) : bounds;
    if (!liveBounds || !viewportInitialized || !width || !height) return;
    void setViewport(getViewportForBounds(liveBounds, width, height, minZoom, refit === 'always' ? maxZoom : Math.min(1, maxZoom), 0.14));
  }, [bounds, contentBounds, fitReady, store, viewportInitialized, width, height, minZoom, refit, maxZoom, setViewport]);
  const viewport = {
    fitView: () => { moved.current = false; fit(); },
    zoomIn: () => { moved.current = true; void zoomIn(); },
    zoomOut: () => { moved.current = true; void zoomOut(); },
  };
  useImperativeHandle(viewportRef, () => viewport);
  useEffect(() => {
    if (lockAutoFit || !fitReady || !bounds || !boundsKey || !viewportInitialized || !width || !height || moved.current) return;
    // A region that shrinks can cut off content that fitted, so a view that refits checks again at each size.
    const key = refit === 'once' ? boundsKey : `${boundsKey}@${width}x${height}`;
    if (fitted.current === key || (fitted.current && refit === 'once')) return;
    const frame = requestAnimationFrame(() => {
      if (moved.current) return;
      fitted.current = key;
      const { x, y, zoom } = getViewport();
      const left = bounds.x * zoom + x, top = bounds.y * zoom + y;
      if (refit === 'always' || left < 0 || top < 0 || left + bounds.width * zoom > width ||
          top + bounds.height * zoom > height) fit();
    });
    return () => cancelAnimationFrame(frame);
  }, [fit, boundsKey, bounds, fitReady, viewportInitialized, width, height, refit, lockAutoFit, getViewport]);
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
    const direction = directions[event.key as keyof typeof directions];
    if (!direction || !event.target.matches('[data-canvas-focus]') || event.metaKey || event.ctrlKey || event.altKey) return;
    const next = adjacentControl(event.currentTarget, event.target, direction);
    event.preventDefault();
    next?.focus({ preventScroll: true });
  }
  function zoomWithoutMotion(event: MouseEvent<HTMLDivElement>) {
    if (!zoomOnDoubleClick || event.ctrlKey || !(event.target instanceof HTMLElement) ||
        !event.target.classList.contains('react-flow__pane') ||
        !window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    event.preventDefault();
    event.stopPropagation();
    const current = getViewport();
    const zoom = Math.max(minZoom, Math.min(maxZoom, current.zoom * (event.shiftKey ? 0.5 : 2)));
    const bounds = event.target.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    moved.current = true;
    void setViewport({
      x: x - (x - current.x) * zoom / current.zoom,
      y: y - (y - current.y) * zoom / current.zoom,
      zoom,
    });
  }
  return <div ref={root} className="relative h-full min-h-0 min-w-0 flex-1 overflow-hidden"
    onDoubleClickCapture={zoomWithoutMotion}
    onFocusCapture={navigateContent ? (event) => {
      const control = event.target;
      if (!control.closest('[data-canvas-item]')) return;
      if (revealFrame.current !== undefined) cancelAnimationFrame(revealFrame.current);
      revealFrame.current = requestAnimationFrame(() => {
        revealFrame.current = undefined;
        if (document.activeElement === control) reveal(control);
      });
    } : undefined}
    onKeyDown={navigateContent ? navigate : undefined}>
    <ReactFlow nodesDraggable={false} nodesConnectable={false} deleteKeyCode={null}
      id={flowId} {...flow} minZoom={minZoom} maxZoom={maxZoom} zoomOnDoubleClick={zoomOnDoubleClick}
      proOptions={{ hideAttribution: true }}
      onMove={(event, camera) => { if (event) moved.current = true; onMove?.(event, camera); }}>
      {children}
    </ReactFlow>
  </div>;
}
