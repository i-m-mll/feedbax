import { X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useSettingsStore } from '@/stores/settingsStore';

type Section = 'general' | 'canvas' | 'storage';

const NAV_ITEMS: { id: Section; label: string }[] = [
  { id: 'general', label: 'General' },
  { id: 'canvas', label: 'Canvas' },
  { id: 'storage', label: 'Storage' },
];

export function SettingsOverlay({ onClose }: { onClose: () => void }) {
  const [activeSection, setActiveSection] = useState<Section>('general');

  const {
    showMinimap,
    toggleMinimap,
    snapToGrid,
    setSnapToGrid,
    snapGridSize,
    setSnapGridSize,
    showGridBackground,
    setShowGridBackground,
    reduceAnimations,
    setReduceAnimations,
    defaultEdgeStyle,
    setDefaultEdgeStyle,
  } = useSettingsStore();

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-stretch justify-end bg-black/40"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      {/* Panel */}
      <div className="flex h-full w-full max-w-3xl bg-white shadow-xl">
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
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
            title="Close settings"
          >
            <X className="w-4 h-4" />
          </button>

          <div className="p-8 max-w-xl">
            {activeSection === 'general' && <GeneralSection
              showMinimap={showMinimap}
              onToggleMinimap={toggleMinimap}
              snapToGrid={snapToGrid}
              onSetSnapToGrid={setSnapToGrid}
              snapGridSize={snapGridSize}
              onSetSnapGridSize={setSnapGridSize}
              showGridBackground={showGridBackground}
              onSetShowGridBackground={setShowGridBackground}
              reduceAnimations={reduceAnimations}
              onSetReduceAnimations={setReduceAnimations}
            />}
            {activeSection === 'canvas' && <CanvasSection
              defaultEdgeStyle={defaultEdgeStyle}
              onSetDefaultEdgeStyle={setDefaultEdgeStyle}
            />}
            {activeSection === 'storage' && <StorageSection />}
          </div>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Section: General                                                             */
/* -------------------------------------------------------------------------- */

interface GeneralSectionProps {
  showMinimap: boolean;
  onToggleMinimap: () => void;
  snapToGrid: boolean;
  onSetSnapToGrid: (v: boolean) => void;
  snapGridSize: number;
  onSetSnapGridSize: (v: number) => void;
  showGridBackground: boolean;
  onSetShowGridBackground: (v: boolean) => void;
  reduceAnimations: boolean;
  onSetReduceAnimations: (v: boolean) => void;
}

function GeneralSection({
  showMinimap,
  onToggleMinimap,
  snapToGrid,
  onSetSnapToGrid,
  snapGridSize,
  onSetSnapGridSize,
  showGridBackground,
  onSetShowGridBackground,
  reduceAnimations,
  onSetReduceAnimations,
}: GeneralSectionProps) {
  return (
    <div>
      <SectionHeading>General</SectionHeading>
      <div className="divide-y divide-slate-100">
        <SettingRow label="Show minimap" description="Display a miniature overview of the canvas">
          <Toggle checked={showMinimap} onChange={onToggleMinimap} />
        </SettingRow>
        <SettingRow label="Snap to grid" description="Snap nodes to a fixed grid when dragging">
          <Toggle checked={snapToGrid} onChange={(checked) => onSetSnapToGrid(checked)} />
        </SettingRow>
        {snapToGrid && (
          <SettingRow label="Grid size" description="Size of each grid cell in pixels">
            <input
              type="number"
              min={5}
              max={100}
              step={5}
              value={snapGridSize}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                if (!isNaN(val) && val > 0) onSetSnapGridSize(val);
              }}
              className="w-20 text-sm text-right border border-slate-200 rounded-lg px-2 py-1 text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-500/30 focus:border-brand-500"
            />
          </SettingRow>
        )}
        <SettingRow label="Show grid background" description="Render a dot grid on the canvas">
          <Toggle
            checked={showGridBackground}
            onChange={(checked) => onSetShowGridBackground(checked)}
          />
        </SettingRow>
        <SettingRow label="Reduce animations" description="Minimize motion for performance or accessibility">
          <Toggle
            checked={reduceAnimations}
            onChange={(checked) => onSetReduceAnimations(checked)}
          />
        </SettingRow>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Section: Canvas                                                              */
/* -------------------------------------------------------------------------- */

interface CanvasSectionProps {
  defaultEdgeStyle: 'default' | 'straight' | 'step';
  onSetDefaultEdgeStyle: (v: 'default' | 'straight' | 'step') => void;
}

function CanvasSection({ defaultEdgeStyle, onSetDefaultEdgeStyle }: CanvasSectionProps) {
  const options: { value: 'default' | 'straight' | 'step'; label: string; description: string }[] = [
    { value: 'default', label: 'Default (curved)', description: 'Smooth bezier curves' },
    { value: 'straight', label: 'Straight', description: 'Direct point-to-point lines' },
    { value: 'step', label: 'Step', description: 'Right-angle routing' },
  ];

  return (
    <div>
      <SectionHeading>Canvas</SectionHeading>
      <div className="mb-1">
        <p className="text-sm font-medium text-slate-700 mb-3">Default edge style</p>
        <div className="flex flex-col gap-2">
          {options.map((opt) => (
            <label
              key={opt.value}
              className="flex items-center gap-3 p-3 rounded-xl border border-slate-100 hover:border-slate-200 hover:bg-slate-50 cursor-pointer transition-colors"
            >
              <input
                type="radio"
                name="defaultEdgeStyle"
                value={opt.value}
                checked={defaultEdgeStyle === opt.value}
                onChange={() => onSetDefaultEdgeStyle(opt.value)}
                className="accent-brand-500"
              />
              <div>
                <div className="text-sm font-medium text-slate-700">{opt.label}</div>
                <div className="text-xs text-slate-400">{opt.description}</div>
              </div>
            </label>
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
  description,
  children,
}: {
  label: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between py-3 gap-4">
      <div className="min-w-0">
        <div className="text-sm font-medium text-slate-700">{label}</div>
        {description && (
          <div className="text-xs text-slate-400 mt-0.5">{description}</div>
        )}
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
