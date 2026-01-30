import { PropertiesPanel } from '@/components/panels/PropertiesPanel';

export function RightPanel() {
  return (
    <aside className="w-80 max-w-full border-l border-slate-100 bg-white/90 backdrop-blur-sm flex flex-col overflow-x-hidden">
      <div className="px-4 pt-4">
        <div className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">
          Properties
        </div>
      </div>
      <div className="flex-1 overflow-y-auto">
        <PropertiesPanel />
      </div>
    </aside>
  );
}
