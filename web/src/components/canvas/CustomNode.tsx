import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { GraphNodeData } from '@/types/graph';
import clsx from 'clsx';
import { useGraphStore } from '@/stores/graphStore';

const PORT_OFFSET = 64;
const PORT_GAP = 24;

export function CustomNode({ data, selected }: NodeProps<GraphNodeData>) {
  const { spec, label, collapsed } = data;
  const toggleNodeCollapse = useGraphStore((state) => state.toggleNodeCollapse);

  return (
    <div
      onDoubleClick={() => toggleNodeCollapse(label)}
      className={clsx(
        'rounded-xl border shadow-soft min-w-[180px] bg-white/90 backdrop-blur',
        selected ? 'border-brand-500 ring-1 ring-brand-500/40' : 'border-slate-200'
      )}
    >
      <div className="px-3 py-2 border-b border-slate-100 bg-slate-50/70 rounded-t-xl flex items-center justify-between">
        <div>
          <div className="text-sm font-medium text-slate-800">{label}</div>
        </div>
        <div className="text-[11px] text-slate-500 truncate max-w-[120px]">{spec.type}</div>
      </div>

      {collapsed ? (
        <div className="px-3 py-2 text-xs text-slate-500 flex items-center justify-between">
          <span>{spec.input_ports.length} inputs</span>
          <span>{spec.output_ports.length} outputs</span>
        </div>
      ) : (
        <div className="flex justify-between px-3 py-2 gap-6">
          <div className="space-y-2">
            {spec.input_ports.map((port, index) => (
              <div key={port} className="flex items-center gap-2 text-xs text-slate-600">
                <Handle
                  type="target"
                  position={Position.Left}
                  id={port}
                  style={{ top: PORT_OFFSET + index * PORT_GAP }}
                  className="w-3 h-3 bg-brand-500"
                />
                <span>{port}</span>
              </div>
            ))}
          </div>
          <div className="space-y-2 text-right">
            {spec.output_ports.map((port, index) => (
              <div key={port} className="flex items-center gap-2 text-xs text-slate-600 justify-end">
                <span>{port}</span>
                <Handle
                  type="source"
                  position={Position.Right}
                  id={port}
                  style={{ top: PORT_OFFSET + index * PORT_GAP }}
                  className="w-3 h-3 bg-mint-500"
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
