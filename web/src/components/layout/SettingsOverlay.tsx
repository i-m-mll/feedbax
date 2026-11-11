import { useState } from 'react';
import { useSettingsStore } from '@/stores/settingsStore';
import { Dialog } from '@/components/ui/Dialog';
import { CloseButton } from '@/components/ui/PanelPrimitives';

type Section = 'general' | 'storage';

const NAV_ITEMS: { id: Section; label: string }[] = [
  { id: 'general', label: 'General' },
  { id: 'storage', label: 'Storage' },
];

export function SettingsOverlay({ onClose }: { onClose: () => void }) {
  const [activeSection, setActiveSection] = useState<Section>('general');

  const { showMinimap, toggleMinimap } = useSettingsStore();

  return (
    <Dialog
      onClose={onClose}
      ariaLabel="Settings"
      className="fixed inset-x-0 bottom-0 top-12 z-50 flex items-stretch bg-white"
      panelClassName="flex h-full w-full bg-white"
    >
        {/* Left sidebar */}
        <nav className="flex-none w-44 border-r border-slate-100 pt-14 pb-6 px-4 flex flex-col gap-0.5">
          <div className="text-[10px] uppercase tracking-widest text-slate-400 px-2 pb-2">
            Settings
          </div>
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveSection(item.id)}
              className={[
                'text-left text-xs font-medium px-2 py-1.5 rounded-lg transition-colors',
                activeSection === item.id
                  ? 'text-slate-800 bg-slate-100'
                  : 'text-slate-400 hover:text-slate-600 hover:bg-slate-50',
              ].join(' ')}
            >
              {item.label}
            </button>
          ))}
        </nav>

        {/* Content area */}
        <div className="flex-1 relative overflow-y-auto">
          {/* Close button */}
          <CloseButton
            onClick={onClose}
            title="Close settings"
            className="absolute top-4 right-4 p-1.5 rounded-lg"
          />

          <div className="p-8 max-w-xl">
            {activeSection === 'general' && <GeneralSection
              showMinimap={showMinimap}
              onToggleMinimap={toggleMinimap}
            />}
            {activeSection === 'storage' && <StorageSection />}
          </div>
        </div>
    </Dialog>
  );
}

/* -------------------------------------------------------------------------- */
/* Section: General                                                             */
/* -------------------------------------------------------------------------- */

interface GeneralSectionProps {
  showMinimap: boolean;
  onToggleMinimap: () => void;
}

function GeneralSection({ showMinimap, onToggleMinimap }: GeneralSectionProps) {
  return (
    <div>
      <SectionHeading>General</SectionHeading>
      <div className="divide-y divide-slate-100">
        <SettingRow label="Show minimap">
          <Toggle checked={showMinimap} onChange={onToggleMinimap} />
        </SettingRow>
      </div>
      <div className="mt-8">
        <SectionHeading>Shortcuts</SectionHeading>
        <div className="divide-y divide-slate-100 rounded-lg border border-slate-100">
          {[
            ['Save project', 'Cmd/Ctrl S'],
            ['Undo', 'Cmd/Ctrl Z'],
            ['Redo', 'Cmd/Ctrl Shift Z or Cmd/Ctrl Y'],
            ['Duplicate selection', 'Cmd/Ctrl D'],
            ['Select all', 'Cmd/Ctrl A'],
            ['Deselect', 'Esc'],
            ['Delete selection', 'Delete or Backspace'],
            ['Zoom to fit', 'Cmd/Ctrl 0'],
          ].map(([label, shortcut]) => (
            <div key={label} className="flex items-center justify-between gap-4 px-3 py-2">
              <span className="text-xs font-medium text-slate-600">{label}</span>
              <kbd className="rounded border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] font-medium text-slate-500">
                {shortcut}
              </kbd>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Section: Storage                                                             */
/* -------------------------------------------------------------------------- */

function StorageSection() {
  return (
    <div>
      <SectionHeading>Storage</SectionHeading>
      <div className="mb-6">
        <p className="text-sm font-medium text-slate-700 mb-1">Data directory</p>
        <p className="text-xs text-slate-400 font-mono bg-slate-50 border border-slate-100 rounded-lg px-3 py-2 select-all">
          ~/.feedbax/web/graphs
        </p>
        <p className="mt-2 text-xs text-slate-400">
          Configured via the <code className="font-mono text-slate-500">FEEDBAX_WEB_DATA</code> environment
          variable. Default: <code className="font-mono text-slate-500">~/.feedbax/web/graphs</code>
        </p>
      </div>
      <div className="text-xs text-slate-400 space-y-2 border-t border-slate-100 pt-4">
        <p>
          Graph files are stored as JSON in this directory. To change the location, set{' '}
          <code className="font-mono text-slate-500">FEEDBAX_WEB_DATA</code> before starting the server.
        </p>
        <p>
          For cloud deployment, a database backend is recommended.
        </p>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Shared primitives                                                            */
/* -------------------------------------------------------------------------- */

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-base font-semibold text-slate-800 mb-4">{children}</h2>
  );
}

function SettingRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between py-3 gap-4">
      <div className="min-w-0">
        <div className="text-sm font-medium text-slate-700">{label}</div>
      </div>
      <div className="flex-none">{children}</div>
    </div>
  );
}

function Toggle({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={[
        'relative inline-flex h-5 w-9 flex-none items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-brand-500/30',
        checked ? 'bg-brand-500' : 'bg-slate-200',
      ].join(' ')}
    >
      <span
        className={[
          'inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform',
          checked ? 'translate-x-4' : 'translate-x-0.5',
        ].join(' ')}
      />
    </button>
  );
}
