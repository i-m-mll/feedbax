import type { NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import type { TapNodeData } from '@/types/graph';
import clsx from 'clsx';
import { memo } from 'react';
import { PortHandle } from '@/components/ui/NodePrimitives';
import { semanticTokens } from '@/components/ui/semanticTokens';

function TapNodeComponent({ data, selected }: NodeProps) {
  const tapData = data as TapNodeData;
  const tap = tapData.tap;
  const outputs = Object.keys(tap.paths ?? {});
  const isProbe = tap.type === 'probe';
  const size = 16;
  const half = size / 2;
  const spacing = 12;
  const total = Math.max(0, outputs.length - 1) * spacing;
  const start = half - total / 2;

  return (
    <div className="relative flex items-center justify-center">
      <div
        className={clsx(
          'flex items-center justify-center text-[9px] font-semibold text-white shadow-soft border',
          isProbe ? 'bg-sky-500 border-sky-200' : 'bg-amber-500 border-amber-200',
          selected ? clsx('ring-2', semanticTokens.selected.ring) : 'ring-0'
        )}
        style={{
          width: size,
          height: size,
          borderRadius: isProbe ? 999 : 4,
          transform: isProbe ? 'none' : 'rotate(45deg)',
        }}
        title={tap.type}
      >
        <span style={{ transform: isProbe ? 'none' : 'rotate(-45deg)' }}>
          {isProbe ? 'P' : 'I'}
        </span>
      </div>
      {outputs.map((name, index) => (
        <PortHandle
          key={`tap-out-${name}`}
          type="source"
          position={Position.Bottom}
          id={name}
          style={{
            left: start + index * spacing,
            bottom: -6,
            transform: 'translateX(-50%)',
          }}
          tone="state"
          size="md"
        />
      ))}
    </div>
  );
}

function areTapNodePropsEqual(previous: NodeProps, next: NodeProps) {
  return (
    previous.id === next.id &&
    previous.data === next.data &&
    previous.selected === next.selected &&
    previous.dragging === next.dragging &&
    previous.isConnectable === next.isConnectable
  );
}

export const TapNode = memo(TapNodeComponent, areTapNodePropsEqual);
