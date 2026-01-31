import { getBezierPath, type EdgeProps } from '@xyflow/react';

export function StateFlowEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const [path] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });
  const primary = data?.primary ?? true;
  const strokeWidth = primary ? 3 : 2.2;
  const stroke = primary ? '#475569' : '#94a3b8';
  const dash = primary ? 'none' : '6 6';

  return (
    <path
      d={path}
      className="react-flow__edge-path"
      style={{
        stroke,
        strokeWidth,
        strokeDasharray: dash,
        opacity: primary ? 0.9 : 0.75,
        fill: 'none',
      }}
      pointerEvents="stroke"
    />
  );
}
