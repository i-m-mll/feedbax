import type { CSSProperties } from 'react';

export type EdgeSchemaStatus = 'warning' | 'blocked' | string | undefined;

export function routedEdgeStyle({
  selected,
  schemaStatus,
  recurrent,
}: {
  selected: boolean;
  schemaStatus: EdgeSchemaStatus;
  recurrent: boolean;
}): CSSProperties {
  const isWarning = schemaStatus === 'warning';
  const isBlocked = schemaStatus === 'blocked';
  return {
    stroke: isWarning ? '#f59e0b' : isBlocked ? '#ef4444' : selected ? '#2563eb' : '#b8bcc6',
    strokeWidth: selected || isWarning || isBlocked ? 2.5 : 1.5,
    strokeDasharray: recurrent ? '7 5' : undefined,
    fill: 'none',
  };
}

export function stateFlowEdgeStyle({
  selected,
  primary,
}: {
  selected: boolean;
  primary: boolean;
}): CSSProperties {
  return {
    stroke: selected ? '#2563eb' : primary ? '#475569' : '#94a3b8',
    strokeWidth: selected ? (primary ? 4.4 : 3.6) : primary ? 3 : 2.2,
    strokeDasharray: primary ? 'none' : '6 6',
    opacity: selected ? 1 : primary ? 0.9 : 0.75,
    fill: 'none',
  };
}
