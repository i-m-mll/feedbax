import React from 'react';
import { useLayoutStore, DIVIDER_HEIGHT } from '@/stores/layoutStore';
import { ChevronUp, ChevronDown, ChevronsUpDown } from 'lucide-react';

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
      className="relative flex items-center border-t border-slate-200 bg-white"
      style={{ height: DIVIDER_HEIGHT }}
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

      {/* Collapse/expand buttons — right side */}
      <div className="absolute right-2 flex flex-col items-center gap-0">
        {topCollapsed ? (
          /* Top is collapsed: show single restore button */
          <button
            className="p-0.5 text-slate-300 hover:text-slate-500"
            title="Expand top pane"
            onClick={() => toggleTop(availableHeight)}
          >
            <ChevronsUpDown className="w-3.5 h-3.5" />
          </button>
        ) : bottomCollapsed ? (
          /* Bottom is collapsed: show single restore button */
          <button
            className="p-0.5 text-slate-300 hover:text-slate-500"
            title="Expand bottom pane"
            onClick={() => toggleBottom(availableHeight)}
          >
            <ChevronsUpDown className="w-3.5 h-3.5" />
          </button>
        ) : (
          /* Both expanded: show ChevronUp (collapse top) and ChevronDown (collapse bottom) */
          <>
            <button
              className="p-0.5 text-slate-300 hover:text-slate-500"
              title="Collapse top pane"
              onClick={() => toggleTop(availableHeight)}
            >
              <ChevronUp className="w-3.5 h-3.5" />
            </button>
            <button
              className="p-0.5 text-slate-300 hover:text-slate-500"
              title="Collapse bottom pane"
              onClick={() => toggleBottom(availableHeight)}
            >
              <ChevronDown className="w-3.5 h-3.5" />
            </button>
          </>
        )}
      </div>
    </div>
  );
}
