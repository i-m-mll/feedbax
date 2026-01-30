import { useEffect, useMemo, useRef, useState } from 'react';
import { Header } from '@/components/layout/Header';
import { StatusBar } from '@/components/layout/StatusBar';
import { TopShelf } from '@/components/layout/TopShelf';
import { BottomShelf } from '@/components/layout/BottomShelf';
import { useAppShortcuts } from '@/hooks/useShortcuts';
import {
  useLayoutStore,
  BOTTOM_COLLAPSED_HEIGHT,
  MIN_BOTTOM_HEIGHT,
} from '@/stores/layoutStore';

export default function App() {
  useAppShortcuts();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [availableHeight, setAvailableHeight] = useState(0);
  const {
    topCollapsed,
    bottomCollapsed,
    bottomHeight,
    initializeBottomHeight,
  } = useLayoutStore();

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) return;
      const { height } = entries[0].contentRect;
      setAvailableHeight(height);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (availableHeight > 0) {
      initializeBottomHeight(Math.round(availableHeight * 0.5));
    }
  }, [availableHeight, initializeBottomHeight]);

  const { topHeight, bottomEffectiveHeight } = useMemo(() => {
    if (availableHeight <= 0) {
      return {
        topHeight: topCollapsed ? 0 : undefined,
        bottomEffectiveHeight: bottomCollapsed ? BOTTOM_COLLAPSED_HEIGHT : undefined,
      };
    }
    if (topCollapsed) {
      return {
        topHeight: 0,
        bottomEffectiveHeight: availableHeight,
      };
    }
    if (bottomCollapsed) {
      return {
        topHeight: Math.max(availableHeight - BOTTOM_COLLAPSED_HEIGHT, 0),
        bottomEffectiveHeight: BOTTOM_COLLAPSED_HEIGHT,
      };
    }
    const clampedBottom = Math.max(
      MIN_BOTTOM_HEIGHT,
      Math.min(availableHeight - MIN_BOTTOM_HEIGHT, bottomHeight)
    );
    return {
      topHeight: Math.max(availableHeight - clampedBottom, 0),
      bottomEffectiveHeight: clampedBottom,
    };
  }, [availableHeight, topCollapsed, bottomCollapsed, bottomHeight]);

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <div ref={containerRef} className="flex-1 min-h-0">
        <div
          className="grid h-full min-h-0"
          style={{
            gridTemplateRows:
              topHeight === undefined || bottomEffectiveHeight === undefined
                ? '1fr auto'
                : `${topHeight}px ${bottomEffectiveHeight}px`,
          }}
        >
          <div className={topCollapsed ? 'overflow-hidden' : 'min-h-0'}>
            {!topCollapsed && <TopShelf />}
          </div>
          <BottomShelf height={bottomEffectiveHeight ?? BOTTOM_COLLAPSED_HEIGHT} />
        </div>
      </div>
      <StatusBar />
    </div>
  );
}
