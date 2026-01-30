import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { GraphNodeData } from '@/types/graph';
import clsx from 'clsx';
import { useGraphStore } from '@/stores/graphStore';
import { useEffect, useState } from 'react';

const PORT_OFFSET = 64;
const PORT_GAP = 24;
const COLLAPSED_HANDLE_TOP = 38;
const COLLAPSED_HANDLE_GAP = 6;

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
        'relative rounded-xl border shadow-soft w-[220px] max-w-[240px] bg-white/90 backdrop-blur',
        selected ? 'border-brand-500 ring-1 ring-brand-500/40' : 'border-slate-200'
      )}
    >
      <div className="px-3 py-2 border-b border-slate-100 bg-slate-50/70 rounded-t-xl flex items-center justify-between gap-3 overflow-hidden">
        <div className="min-w-0 flex-1">
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
              className="text-sm font-medium text-slate-800 hover:text-brand-600 truncate w-full text-left"
              onClick={(event) => {
                event.stopPropagation();
                setIsEditing(true);
              }}
              onDoubleClick={(event) => event.stopPropagation()}
              title={label}
            >
              {label}
            </button>
          )}
        </div>
        <div className="text-[11px] text-slate-500 truncate max-w-[110px]" title={spec.type}>
          {spec.type}
        </div>
      </div>

      <div
        className={
          collapsed
            ? 'px-3 py-2 text-xs text-slate-500 flex items-center justify-between min-h-[48px]'
            : 'flex justify-between px-3 py-2 gap-6'
        }
      >
        <div className={collapsed ? '' : 'space-y-2'}>
          {collapsed ? (
            <>
              {spec.input_ports.map((port, index) => (
                <Handle
                  key={port}
                  type="target"
                  position={Position.Left}
                  id={port}
                  style={{
                    top: COLLAPSED_HANDLE_TOP + index * COLLAPSED_HANDLE_GAP,
                  }}
                  className="w-3 h-3 bg-brand-500 opacity-0 pointer-events-none"
                />
              ))}
              <span>{spec.input_ports.length} inputs</span>
            </>
          ) : (
            spec.input_ports.map((port, index) => (
              <div key={port} className="flex items-center gap-2 text-xs text-slate-600">
                <Handle
                  type="target"
                  position={Position.Left}
                  id={port}
                  style={{
                    top: PORT_OFFSET + index * PORT_GAP,
                  }}
                  className="w-3 h-3 bg-brand-500"
                />
                <span>{port}</span>
              </div>
            ))
          )}
        </div>
        <div className={collapsed ? 'text-right' : 'space-y-2 text-right'}>
          {collapsed ? (
            <>
              {spec.output_ports.map((port, index) => (
                <Handle
                  key={port}
                  type="source"
                  position={Position.Right}
                  id={port}
                  style={{
                    top: COLLAPSED_HANDLE_TOP + index * COLLAPSED_HANDLE_GAP,
                  }}
                  className="w-3 h-3 bg-mint-500 opacity-0 pointer-events-none"
                />
              ))}
              <span>{spec.output_ports.length} outputs</span>
            </>
          ) : (
            spec.output_ports.map((port, index) => (
              <div
                key={port}
                className="flex items-center gap-2 text-xs text-slate-600 justify-end"
              >
                <span>{port}</span>
                <Handle
                  type="source"
                  position={Position.Right}
                  id={port}
                  style={{
                    top: PORT_OFFSET + index * PORT_GAP,
                  }}
                  className="w-3 h-3 bg-mint-500"
                />
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
