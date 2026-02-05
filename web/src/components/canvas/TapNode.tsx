import type { NodeProps } from '@xyflow/react';
import { Handle, Position } from '@xyflow/react';
import type { TapNodeData } from '@/types/graph';
import clsx from 'clsx';

export function TapNode({ data, selected }: NodeProps) {
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
          selected ? 'ring-2 ring-brand-400' : 'ring-0'
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
        <Handle
          key={`tap-out-${name}`}
          type="source"
          position={Position.Bottom}
          id={name}
          style={{
            left: start + index * spacing,
            bottom: -6,
            transform: 'translateX(-50%)',
          }}
          className="w-2.5 h-2.5 bg-slate-500 border border-white shadow-soft"
        />
      ))}
    </div>
  );
}
