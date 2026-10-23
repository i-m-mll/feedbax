import React from 'react';
import { useLayoutStore, DIVIDER_HEIGHT } from '@/stores/layoutStore';

interface DividerProps {
  availableHeight: number;
}

export function Divider({ availableHeight }: DividerProps) {
  const { topCollapsed, bottomCollapsed, setBottomHeight, bottomHeight } =
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
      className={isDraggable ? 'z-10 border-t border-slate-200 cursor-row-resize' : 'z-10 border-t border-slate-200'}
      style={{ height: DIVIDER_HEIGHT }}
      onPointerDown={handleDragStart}
      title={isDraggable ? 'Drag to resize panes' : undefined}
    />
  );
}
