import {
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
  useReactFlow,
} from '@xyflow/react';
import { useCallback } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import type { GraphEdgeData } from '@/types/graph';

function buildPolylinePath(points: { x: number; y: number }[]) {
  if (points.length === 0) return '';
  const [first, ...rest] = points;
  return `M ${first.x},${first.y} ${rest.map((pt) => `L ${pt.x},${pt.y}`).join(' ')}`;
}

export function RoutedEdge({
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
  const [bezierPath] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const path = isElbow ? elbowPath : bezierPath;

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
        style={{
          stroke: selected ? '#2563eb' : '#b8bcc6',
          strokeWidth: selected ? 2.5 : 1.5,
          fill: 'none',
        }}
        onDoubleClick={handleDoubleClick}
      />
      <path
        d={path}
        style={{ stroke: 'transparent', strokeWidth: 16, fill: 'none' }}
        onClick={handlePathClick}
        onDoubleClick={handleDoubleClick}
      />
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
                const handleMove = (moveEvent: PointerEvent) => {
                  const next = screenToFlowPosition({
                    x: moveEvent.clientX,
                    y: moveEvent.clientY,
                  });
                  updateEdgePoint(id, index, next);
                };
                const handleUp = () => {
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
