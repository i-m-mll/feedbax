import { getBezierPath, type EdgeProps } from '@xyflow/react';

export function StateFlowEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
}: EdgeProps) {
  const [path] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  return (
    <path
      d={path}
      className="react-flow__edge-path"
      style={{
        stroke: '#475569',
        strokeWidth: 3,
        opacity: 0.9,
        fill: 'none',
      }}
      pointerEvents="none"
    />
  );
}
