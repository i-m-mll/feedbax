import { getBezierPath, type EdgeProps } from '@xyflow/react';
import { memo } from 'react';
import { stateFlowEdgeStyle } from '@/components/ui/edgeStyle';

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
  const primary = (data as { primary?: boolean } | undefined)?.primary ?? true;

  return (
    <path
      d={path}
      className="react-flow__edge-path"
      style={stateFlowEdgeStyle({ selected, primary })}
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
