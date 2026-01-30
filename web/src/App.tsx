import { Header } from '@/components/layout/Header';
import { StatusBar } from '@/components/layout/StatusBar';
import { TopShelf } from '@/components/layout/TopShelf';
import { BottomShelf } from '@/components/layout/BottomShelf';
import { useAppShortcuts } from '@/hooks/useShortcuts';
import { useLayoutStore } from '@/stores/layoutStore';
import clsx from 'clsx';

export default function App() {
  useAppShortcuts();
  const topCollapsed = useLayoutStore((state) => state.topCollapsed);

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <div className="flex-1 flex flex-col min-h-0">
        <div
          className={clsx(
            'transition-all duration-200 ease-in-out',
            topCollapsed ? 'h-0 overflow-hidden' : 'flex-1 min-h-0'
          )}
        >
          {!topCollapsed && <TopShelf />}
        </div>
        <BottomShelf />
      </div>
      <StatusBar />
    </div>
  );
}
