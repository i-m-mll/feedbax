import { getBezierPath, type EdgeProps } from '@xyflow/react';
import { memo } from 'react';

function StateFlowEdgeComponent({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  selected,
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
  const strokeWidth = selected ? (primary ? 4.4 : 3.6) : primary ? 3 : 2.2;
  const stroke = selected ? '#2563eb' : primary ? '#475569' : '#94a3b8';
  const dash = primary ? 'none' : '6 6';

  return (
    <path
      d={path}
      className="react-flow__edge-path"
      style={{
        stroke,
        strokeWidth,
        strokeDasharray: dash,
        opacity: selected ? 1 : primary ? 0.9 : 0.75,
        fill: 'none',
      }}
      pointerEvents="stroke"
    />
  );
}

function areStateFlowEdgePropsEqual(previous: EdgeProps, next: EdgeProps) {
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

export const StateFlowEdge = memo(StateFlowEdgeComponent, areStateFlowEdgePropsEqual);
