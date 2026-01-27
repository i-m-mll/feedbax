import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { GraphNodeData } from '@/types/graph';
import clsx from 'clsx';
import { useGraphStore } from '@/stores/graphStore';
import { useEffect, useState } from 'react';

const PORT_OFFSET = 64;
const PORT_GAP = 24;
const COLLAPSED_HANDLE_TOP = 38;

export function CustomNode({ data, selected }: NodeProps<GraphNodeData>) {
  const { spec, label, collapsed } = data;
  const toggleNodeCollapse = useGraphStore((state) => state.toggleNodeCollapse);
  const renameNode = useGraphStore((state) => state.renameNode);
  const [isEditing, setIsEditing] = useState(false);
  const [nameValue, setNameValue] = useState(label);

  useEffect(() => {
    if (!isEditing) {
      setNameValue(label);
    }
  }, [label, isEditing]);

  return (
    <div
      onDoubleClick={() => toggleNodeCollapse(label)}
      className={clsx(
        'rounded-xl border shadow-soft min-w-[180px] bg-white/90 backdrop-blur',
        selected ? 'border-brand-500 ring-1 ring-brand-500/40' : 'border-slate-200'
      )}
    >
      <div className="px-3 py-2 border-b border-slate-100 bg-slate-50/70 rounded-t-xl flex items-center justify-between gap-3">
        <div className="min-w-0">
          {isEditing ? (
            <input
              value={nameValue}
              onChange={(event) => setNameValue(event.target.value)}
              onBlur={() => {
                renameNode(label, nameValue);
                setIsEditing(false);
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  renameNode(label, nameValue);
                  setIsEditing(false);
                }
                if (event.key === 'Escape') {
                  setIsEditing(false);
                  setNameValue(label);
                }
              }}
              className="w-full bg-white/70 border border-slate-200 rounded-md px-2 py-1 text-sm text-slate-800"
              autoFocus
              onClick={(event) => event.stopPropagation()}
            />
          ) : (
            <button
              className="text-sm font-medium text-slate-800 hover:text-brand-600"
              onClick={(event) => {
                event.stopPropagation();
                setIsEditing(true);
              }}
              onDoubleClick={(event) => event.stopPropagation()}
            >
              {label}
            </button>
          )}
        </div>
        <div className="text-[11px] text-slate-500 truncate max-w-[120px]">{spec.type}</div>
      </div>

      <div className={collapsed ? 'px-3 py-2 text-xs text-slate-500 flex items-center justify-between min-h-[48px]' : 'flex justify-between px-3 py-2 gap-6'}>
        <div className={collapsed ? '' : 'space-y-2'}>
          {spec.input_ports.map((port, index) => (
            <div
              key={port}
              className={collapsed ? 'sr-only' : 'flex items-center gap-2 text-xs text-slate-600'}
            >
              <Handle
                type="target"
                position={Position.Left}
                id={port}
                style={{
                  top: collapsed ? COLLAPSED_HANDLE_TOP : PORT_OFFSET + index * PORT_GAP,
                }}
                className={collapsed ? 'w-3 h-3 bg-brand-500 opacity-0 pointer-events-none' : 'w-3 h-3 bg-brand-500'}
              />
              {!collapsed && <span>{port}</span>}
            </div>
          ))}
          {collapsed && <span>{spec.input_ports.length} inputs</span>}
        </div>
        <div className={collapsed ? 'text-right' : 'space-y-2 text-right'}>
          {spec.output_ports.map((port, index) => (
            <div
              key={port}
              className={collapsed ? 'sr-only' : 'flex items-center gap-2 text-xs text-slate-600 justify-end'}
            >
              {!collapsed && <span>{port}</span>}
              <Handle
                type="source"
                position={Position.Right}
                id={port}
                style={{
                  top: collapsed ? COLLAPSED_HANDLE_TOP : PORT_OFFSET + index * PORT_GAP,
                }}
                className={collapsed ? 'w-3 h-3 bg-mint-500 opacity-0 pointer-events-none' : 'w-3 h-3 bg-mint-500'}
              />
            </div>
          ))}
          {collapsed && <span>{spec.output_ports.length} outputs</span>}
        </div>
      </div>
    </div>
  );
}
