import { Header } from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { StatusBar } from '@/components/layout/StatusBar';
import { RightPanel } from '@/components/panels/RightPanel';
import { Canvas } from '@/components/canvas/Canvas';
import { useAppShortcuts } from '@/hooks/useShortcuts';

export default function App() {
  useAppShortcuts();

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <div className="flex flex-1 min-h-0">
        <Sidebar />
        <main className="relative flex-1 min-h-0">
          <div className="absolute inset-0">
            <Canvas />
          </div>
        </main>
        <RightPanel />
      </div>
      <StatusBar />
    </div>
  );
}
