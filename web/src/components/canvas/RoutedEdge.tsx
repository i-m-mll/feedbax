import {
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
  useReactFlow,
} from '@xyflow/react';
import { memo, useCallback } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import type { GraphEdgeData } from '@/types/graph';
import { routedEdgeStyle } from '@/components/ui/edgeStyle';

function buildPolylinePath(points: { x: number; y: number }[]) {
  if (points.length === 0) return '';
  const [first, ...rest] = points;
  return `M ${first.x},${first.y} ${rest.map((pt) => `L ${pt.x},${pt.y}`).join(' ')}`;
}

function offsetFromTangent(tangent: { x: number; y: number }, distance = 11) {
  const length = Math.hypot(tangent.x, tangent.y);
  if (length === 0) return { x: 0, y: -distance };
  return {
    x: (tangent.y / length) * distance,
    y: (-tangent.x / length) * distance,
  };
}

function polylineLabelPoint(points: { x: number; y: number }[]) {
  if (points.length === 0) return { x: 0, y: 0, tangent: { x: 1, y: 0 } };
  if (points.length === 1) return { ...points[0], tangent: { x: 1, y: 0 } };

  const segments = points.slice(1).map((point, index) => {
    const previous = points[index];
    const length = Math.hypot(point.x - previous.x, point.y - previous.y);
    return { from: previous, to: point, length };
  });
  const totalLength = segments.reduce((sum, segment) => sum + segment.length, 0);
  let distance = totalLength / 2;
  for (const segment of segments) {
    if (distance <= segment.length) {
      const ratio = segment.length === 0 ? 0 : distance / segment.length;
      return {
        x: segment.from.x + (segment.to.x - segment.from.x) * ratio,
        y: segment.from.y + (segment.to.y - segment.from.y) * ratio,
        tangent: {
          x: segment.to.x - segment.from.x,
          y: segment.to.y - segment.from.y,
        },
      };
    }
    distance -= segment.length;
  }
  const last = points[points.length - 1];
  const previous = points[points.length - 2];
  return {
    ...last,
    tangent: {
      x: last.x - previous.x,
      y: last.y - previous.y,
    },
  };
}

function RoutedEdgeComponent({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  selected,
  data,
}: EdgeProps) {
  const routing = (data as GraphEdgeData | undefined)?.routing;
  const schemaStatus = (data as GraphEdgeData | undefined)?.schema_status;
  const schemaMessage = (data as GraphEdgeData | undefined)?.schema_message;
  const temporality = (data as GraphEdgeData | undefined)?.temporality ?? 'instant';
  const isRecurrent = temporality === 'recurrent';
  const isElbow = routing?.style === 'elbow';
  const points = routing?.points ?? [];
  const { screenToFlowPosition } = useReactFlow();
  const addEdgePoint = useGraphStore((state) => state.addEdgePoint);
  const updateEdgePoint = useGraphStore((state) => state.updateEdgePoint);
  const removeEdgePoint = useGraphStore((state) => state.removeEdgePoint);
  const toggleEdgeStyleForEdge = useGraphStore((state) => state.toggleEdgeStyleForEdge);

  const autoElbowPoints =
    points.length === 0
      ? [
          { x: (sourceX + targetX) / 2, y: sourceY },
          { x: (sourceX + targetX) / 2, y: targetY },
        ]
      : points;

  const pathPoints = [
    { x: sourceX, y: sourceY },
    ...autoElbowPoints,
    { x: targetX, y: targetY },
  ];

  const elbowPath = buildPolylinePath(pathPoints);
  const [bezierPath, bezierLabelX, bezierLabelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const path = isElbow ? elbowPath : bezierPath;
  const labelPoint = isElbow
    ? polylineLabelPoint(pathPoints)
    : {
        x: bezierLabelX,
        y: bezierLabelY,
        tangent: { x: targetX - sourceX, y: targetY - sourceY },
      };
  const labelOffset = offsetFromTangent(labelPoint.tangent);

  const handlePathClick = useCallback(
    (event: React.MouseEvent<SVGPathElement>) => {
      if (!event.altKey && !event.shiftKey) return;
      event.stopPropagation();
      const point = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      addEdgePoint(id, point);
    },
    [addEdgePoint, id, screenToFlowPosition]
  );

  const handleDoubleClick = useCallback(
    (event: React.MouseEvent<SVGPathElement>) => {
      event.stopPropagation();
      toggleEdgeStyleForEdge(id);
    },
    [id, toggleEdgeStyleForEdge]
  );

  return (
    <>
      <path
        d={path}
        className="react-flow__edge-path"
        style={routedEdgeStyle({ selected, schemaStatus, recurrent: isRecurrent })}
        onDoubleClick={handleDoubleClick}
        aria-label={typeof schemaMessage === 'string' ? schemaMessage : undefined}
      />
      <path
        d={path}
        style={{ stroke: 'transparent', strokeWidth: 16, fill: 'none' }}
        onClick={handlePathClick}
        onDoubleClick={handleDoubleClick}
      />
      {isRecurrent && (
        <EdgeLabelRenderer>
          <div
            className="pointer-events-none text-[10px] font-medium leading-none text-slate-500"
            style={{
              position: 'absolute',
              transform: `translate(${labelPoint.x + labelOffset.x}px, ${labelPoint.y + labelOffset.y}px) translate(-50%, -50%)`,
            }}
          >
            t+1
          </div>
        </EdgeLabelRenderer>
      )}
      {isElbow &&
        points.map((point, index) => (
          <EdgeLabelRenderer key={`${id}-point-${index}`}>
            <div
              className="w-2.5 h-2.5 rounded-full bg-white border border-slate-300 shadow-soft cursor-move"
              style={{
                transform: `translate(-50%, -50%) translate(${point.x}px, ${point.y}px)`,
              }}
              onPointerDown={(event) => {
                event.stopPropagation();
                if (event.altKey) {
                  removeEdgePoint(id, index);
                  return;
                }
                let frame = 0;
                let pendingEvent: PointerEvent | null = null;
                const flushMove = () => {
                  frame = 0;
                  const moveEvent = pendingEvent;
                  if (!moveEvent) return;
                  pendingEvent = null;
                  const next = screenToFlowPosition({
                    x: moveEvent.clientX,
                    y: moveEvent.clientY,
                  });
                  updateEdgePoint(id, index, next);
                };
                const handleMove = (moveEvent: PointerEvent) => {
                  pendingEvent = moveEvent;
                  if (!frame) frame = requestAnimationFrame(flushMove);
                };
                const handleUp = () => {
                  if (frame) {
                    cancelAnimationFrame(frame);
                    flushMove();
                  }
                  window.removeEventListener('pointermove', handleMove);
                  window.removeEventListener('pointerup', handleUp);
                };
                window.addEventListener('pointermove', handleMove);
                window.addEventListener('pointerup', handleUp);
              }}
              title="Drag to route. Alt-click to remove."
            />
          </EdgeLabelRenderer>
        ))}
    </>
  );
}

function areRoutedEdgePropsEqual(previous: EdgeProps, next: EdgeProps) {
  return (
    previous.id === next.id &&
    previous.sourceX === next.sourceX &&
    previous.sourceY === next.sourceY &&
    previous.targetX === next.targetX &&
    previous.targetY === next.targetY &&
    previous.sourcePosition === next.sourcePosition &&
    previous.targetPosition === next.targetPosition &&
    previous.selected === next.selected &&
    previous.data === next.data
  );
}

export const RoutedEdge = memo(RoutedEdgeComponent, areRoutedEdgePropsEqual);
