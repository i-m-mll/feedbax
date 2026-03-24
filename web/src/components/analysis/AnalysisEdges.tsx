/**
 * Edge types for the analysis DAG canvas.
 *
 * - AnalysisExplicitEdge: solid edge for explicit port wires
 * - AnalysisImplicitEdge: dashed/muted edge for implicit data dependencies
 *
 * Both edge types render a small field-path label when the wire carries a
 * specific state field path (e.g. ".net.hidden"), positioned at the midpoint.
 */

import { getBezierPath, type EdgeProps } from '@xyflow/react';
import type { AnalysisEdgeData } from '@/stores/analysisStore';

/**
 * Render a small label on the edge showing the field path suffix.
 * For "states.net.hidden" we show ".net.hidden" (drop the root segment).
 * For a top-level path like "states" we show nothing.
 */
function FieldPathLabel({
  fieldPath,
  labelX,
  labelY,
}: {
  fieldPath: string | undefined;
  labelX: number;
  labelY: number;
}) {
  if (!fieldPath) return null;

  // Show the suffix after the first segment, or the full path if it's a
  // single segment (top-level). For top-level, no label needed.
  const dotIndex = fieldPath.indexOf('.');
  if (dotIndex === -1) return null;

  const suffix = fieldPath.slice(dotIndex);

  return (
    <text
      x={labelX}
      y={labelY - 8}
      textAnchor="middle"
      dominantBaseline="auto"
      style={{
        fontSize: '9px',
        fill: '#94a3b8',
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        pointerEvents: 'none',
        userSelect: 'none',
      }}
    >
      {suffix}
    </text>
  );
}

/** Explicit port wire — solid, emerald-tinted. */
export function AnalysisExplicitEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  selected,
  data,
}: EdgeProps) {
  const edgeData = data as AnalysisEdgeData | undefined;
  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  return (
    <>
      <path
        d={path}
        className="react-flow__edge-path"
        style={{
          stroke: selected ? '#059669' : '#6ee7b7',
          strokeWidth: selected ? 2 : 1.5,
          fill: 'none',
        }}
      />
      <FieldPathLabel
        fieldPath={edgeData?.fieldPath}
        labelX={labelX}
        labelY={labelY}
      />
    </>
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
  data,
}: EdgeProps) {
  const edgeData = data as AnalysisEdgeData | undefined;
  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  return (
    <>
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
      <FieldPathLabel
        fieldPath={edgeData?.fieldPath}
        labelX={labelX}
        labelY={labelY}
      />
    </>
  );
}
