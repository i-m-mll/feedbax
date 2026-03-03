import { Sidebar } from '@/components/layout/Sidebar';
import { RightPanel } from '@/components/panels/RightPanel';
import { Canvas } from '@/components/canvas/Canvas';
import { useLayoutStore } from '@/stores/layoutStore';

export function TopShelf({ height }: { height: number }) {
  const { topCollapsed } = useLayoutStore();
  return (
    <section
      className="flex flex-col h-full min-h-0 bg-white/80 backdrop-blur-sm"
      style={{ height }}
    >
      {!topCollapsed && (
        <div className="flex flex-1 min-h-0">
          <Sidebar />
          <main className="relative flex-1 min-h-0">
            <div className="absolute inset-0">
              <Canvas />
            </div>
          </main>
          <RightPanel />
        </div>
      )}
    </section>
  );
}
