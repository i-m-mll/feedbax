import { Sidebar } from '@/components/layout/Sidebar';
import { RightPanel } from '@/components/panels/RightPanel';
import { Canvas } from '@/components/canvas/Canvas';

export function TopShelf() {
  return (
    <div className="flex h-full min-h-0 border-b border-slate-100">
      <Sidebar />
      <main className="relative flex-1 min-h-0">
        <div className="absolute inset-0">
          <Canvas />
        </div>
      </main>
      <RightPanel />
    </div>
  );
}
