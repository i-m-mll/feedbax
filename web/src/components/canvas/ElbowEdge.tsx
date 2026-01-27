import { BaseEdge, type EdgeProps, getSmoothStepPath } from '@xyflow/react';

export function ElbowEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  selected,
}: EdgeProps) {
  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 6,
  });

  return (
    <BaseEdge
      id={id}
      path={edgePath}
      style={{
        stroke: selected ? '#2563eb' : '#b8bcc6',
        strokeWidth: selected ? 2.5 : 1.5,
      }}
    />
  );
}
