import React from 'react';
import { useLayoutStore, DIVIDER_HEIGHT } from '@/stores/layoutStore';
import { ChevronUp, ChevronDown } from 'lucide-react';

interface DividerProps {
  availableHeight: number;
}

export function Divider({ availableHeight }: DividerProps) {
  const { topCollapsed, bottomCollapsed, toggleTop, toggleBottom, setBottomHeight, bottomHeight } =
    useLayoutStore();

  const isDraggable = !topCollapsed && !bottomCollapsed;

  const handleDragStart = (e: React.PointerEvent) => {
    if (!isDraggable) return;
    const startY = e.clientY;
    const startHeight = bottomHeight;
    const onMove = (ev: PointerEvent) => {
      const delta = startY - ev.clientY;
      setBottomHeight(startHeight + delta, availableHeight);
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  return (
    <div
      className="relative flex items-center border-t border-slate-200 bg-white z-10"
      style={{ height: DIVIDER_HEIGHT, overflow: 'visible' }}
    >
      {/* Drag pill — center, only shown when both expanded */}
      {isDraggable && (
        <div
          className="absolute left-1/2 -translate-x-1/2 w-12 h-full cursor-row-resize flex items-center justify-center"
          onPointerDown={handleDragStart}
        >
          <div className="w-10 h-1 rounded-full bg-slate-200 hover:bg-slate-300" />
        </div>
      )}

      {/* Collapse/expand buttons — right side, straddle the divider line */}
      <div className="absolute right-3 top-1/2 -translate-y-1/2" style={{ overflow: 'visible' }}>
        {/* ChevronUp — sits above the line; hidden when top pane is already collapsed */}
        {!topCollapsed && (
          <button
            className="absolute bottom-[10px] right-0 w-5 h-5 rounded-full bg-white border border-slate-200 shadow-sm flex items-center justify-center text-slate-400 hover:text-slate-600 hover:border-slate-300 transition-colors"
            title={bottomCollapsed ? 'Expand bottom pane' : 'Collapse top pane'}
            onClick={() => bottomCollapsed ? toggleBottom(availableHeight) : toggleTop(availableHeight)}
          >
            <ChevronUp className="w-3 h-3" />
          </button>
        )}
        {/* ChevronDown — sits below the line; hidden when bottom pane is already collapsed */}
        {!bottomCollapsed && (
          <button
            className="absolute top-[10px] right-0 w-5 h-5 rounded-full bg-white border border-slate-200 shadow-sm flex items-center justify-center text-slate-400 hover:text-slate-600 hover:border-slate-300 transition-colors"
            title={topCollapsed ? 'Expand top pane' : 'Collapse bottom pane'}
            onClick={() => topCollapsed ? toggleTop(availableHeight) : toggleBottom(availableHeight)}
          >
            <ChevronDown className="w-3 h-3" />
          </button>
        )}
      </div>
    </div>
  );
}
