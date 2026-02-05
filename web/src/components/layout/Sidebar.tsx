import { ComponentLibrary } from '@/components/panels/ComponentLibrary';

export function Sidebar() {
  return (
    <aside className="w-64 border-r border-slate-100 bg-white/90 backdrop-blur-sm flex flex-col">
      <div className="px-4 pt-4 pb-2">
        <h2 className="font-display text-xs uppercase tracking-[0.3em] text-slate-400">
          Components
        </h2>
      </div>
      <ComponentLibrary />
    </aside>
  );
}
