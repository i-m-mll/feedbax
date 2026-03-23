/**
 * Edge types for the analysis DAG canvas.
 *
 * - AnalysisExplicitEdge: solid edge for explicit port wires
 * - AnalysisImplicitEdge: dashed/muted edge for implicit data dependencies
 */

import { getBezierPath, type EdgeProps } from '@xyflow/react';

/** Explicit port wire — solid, emerald-tinted. */
export function AnalysisExplicitEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
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

  return (
    <path
      d={path}
      className="react-flow__edge-path"
      style={{
        stroke: selected ? '#059669' : '#6ee7b7',
        strokeWidth: selected ? 2 : 1.5,
        fill: 'none',
      }}
    />
  );
}

/** Implicit data dependency — dashed, muted slate. */
export function AnalysisImplicitEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
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

  return (
    <path
      d={path}
      className="react-flow__edge-path"
      style={{
        stroke: selected ? '#64748b' : '#cbd5e1',
        strokeWidth: selected ? 1.5 : 1,
        strokeDasharray: '6 4',
        opacity: 0.7,
        fill: 'none',
      }}
    />
  );
}
