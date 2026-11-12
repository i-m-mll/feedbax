import {
  ArrowLeft, ArrowRight, BarChart3, Columns2, Crosshair, FileText, GitBranch, Image, List,
  Map as SceneIcon, PanelBottom, PanelLeft, PanelRight, PanelTop, Rows2, Settings2, Target, Workflow, X,
  type LucideIcon,
} from 'lucide-react';
import {
  useId, useLayoutEffect, useRef, useState,
  type CSSProperties, type Dispatch, type ReactNode, type SetStateAction,
} from 'react';
import { appTip, installTooltips } from '@window-shell/browser/tooltip';
import { ResizeSeparator } from '@/components/layout/ResizeSeparator';
import { studioName, type StudioNameId } from '@/components/activity/studioVocabulary';
import {
  compactMapHeight, fitProjectWindow, projectRailWidth, type PreferredWindowSize, type ProjectWindowSizePreview,
} from './projectWindowLayout';
import {
  projectWindowSlotView, projectWindowViews, showProjectWindowView,
  type ProjectWindowMode, type ProjectWindowSlot, type ProjectWindowState, type ProjectWindowView,
} from './projectWindowState';
import './projectWindow.css';

export interface ProjectWindowViewRegion {
  body: ReactNode;
  header?: ReactNode;
  drawer?: ReactNode;
  inspector?: ReactNode;
}

export interface ProjectWindowProps {
  state: ProjectWindowState;
  onStateChange: Dispatch<SetStateAction<ProjectWindowState>>;
  projectName: string;
  saveStatus?: ReactNode;
  availableViews: readonly ProjectWindowView[];
  renderView: (view: ProjectWindowView, slot: ProjectWindowSlot) => ProjectWindowViewRegion;
  map: ReactNode;
  mapDrawer: ReactNode;
  compactMap: ReactNode;
  onBack?: () => void;
  onForward?: () => void;
  viewShortcuts?: Partial<Record<ProjectWindowView, string>>;
  variant?: ReactNode;
  commandPalette?: ReactNode;
  clock?: ReactNode;
  failure?: ReactNode;
  problems?: ReactNode;
  activity?: ReactNode;
}

const views = {
  runs: List,
  model: GitBranch,
  task: Crosshair,
  objective: Target,
  training: Settings2,
  scene: SceneIcon,
  results: BarChart3,
  analysis: Workflow,
  figures: Image,
  document: FileText,
} satisfies Record<ProjectWindowView, LucideIcon>;

const modes = [
  { mode: 'map', name: 'mapOnly', icon: PanelTop },
  { mode: 'both', name: 'both', icon: Rows2 },
  { mode: 'details', name: 'detailsOnly', icon: PanelBottom },
] satisfies Array<{ mode: ProjectWindowMode; name: StudioNameId; icon: LucideIcon }>;

const slots = ['primary', 'secondary'] as const;

function WindowButton({
  name, icon: Icon, onClick, pressed, controls, placement = 'bottom', shortcut,
}: {
  name: StudioNameId;
  icon: LucideIcon;
  onClick?: () => void;
  pressed?: boolean;
  controls?: string;
  placement?: 'bottom' | 'right';
  shortcut?: string;
}) {
  const label = studioName(name);
  return (
    <button type="button" aria-label={studioName(name)} aria-pressed={pressed} aria-controls={controls}
      disabled={!onClick} onClick={onClick} className="studio-window-control"
      {...appTip(shortcut ? `${label} ${shortcut}` : label, placement)}>
      <Icon aria-hidden="true" className="size-4" />
    </button>
  );
}

function usePanelSize() {
  const panel = useRef<HTMLElement | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const element = panel.current;
    if (!element) return;
    function measure(width: number, height: number) {
      const next = { width: Math.round(width), height: Math.round(height) };
      setSize((previous) => next.width === previous.width && next.height === previous.height ? previous : next);
    }
    measure(element.clientWidth, element.clientHeight);
    const observer = new ResizeObserver(([entry]) => {
      if (entry) measure(entry.contentRect.width, entry.contentRect.height);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return { panel, ...size };
}

export function ProjectWindow({
  state, onStateChange, projectName, saveStatus, availableViews, renderView, map, mapDrawer, compactMap,
  onBack, onForward, viewShortcuts, variant, commandPalette, clock, failure, problems, activity,
}: ProjectWindowProps) {
  const id = useId();
  const ids = {
    map: `${id}-map`,
    mapDrawer: `${id}-map-drawer`,
    details: `${id}-details`,
    detailsDrawer: `${id}-details-drawer`,
    inspector: `${id}-inspector`,
    primary: `${id}-primary`,
    secondary: `${id}-secondary`,
  };
  const { panel, width, height } = usePanelSize();
  const [preview, setPreview] = useState<ProjectWindowSizePreview>({});
  const slotElements = useRef<Record<ProjectWindowSlot, HTMLElement | null>>({ primary: null, secondary: null });
  const pendingFocus = useRef<ProjectWindowSlot | null>(null);
  useLayoutEffect(() => installTooltips(), []);

  const primaryView = projectWindowSlotView(state, 'primary', availableViews);
  const secondaryView = projectWindowSlotView(state, 'secondary', availableViews);
  const focusedSlot = secondaryView && state.focusedSlot === 'secondary' ? 'secondary' : 'primary';
  const shownViews = { primary: primaryView, secondary: secondaryView };
  const regions = {
    primary: primaryView ? renderView(primaryView, 'primary') : null,
    secondary: secondaryView ? renderView(secondaryView, 'secondary') : null,
  };
  const focusedRegion = regions[focusedSlot];
  const fit = fitProjectWindow(width, height, secondaryView ? state : { ...state, secondaryView: null }, preview);
  const detailsVisible = state.mode !== 'map';
  const fullMapVisible = state.mode === 'map' || (state.mode === 'both' && fit.mapHeight.open);
  const mapDrawerVisible = fullMapVisible && fit.mapDrawer.open;
  const detailsDrawerVisible = detailsVisible && fit.detailsDrawer.open;
  const inspectorVisible = detailsVisible && fit.inspector.open;
  const split = secondaryView !== null;
  const folded = split && !fit.secondary.open;
  const slotWidth = (slot: ProjectWindowSlot) => folded
    ? (slot === focusedSlot ? fit.primarySize : fit.secondary.size)
    : slot === 'primary' ? fit.primarySize : fit.secondary.size;

  useLayoutEffect(() => {
    if (!pendingFocus.current || !detailsVisible) return;
    const element = slotElements.current[pendingFocus.current];
    if (!element) return;
    pendingFocus.current = null;
    element.focus();
  }, [detailsVisible, focusedSlot, split]);

  function focusSlot(slot: ProjectWindowSlot, moveFocus = false) {
    if (moveFocus) pendingFocus.current = slot;
    onStateChange((previous) => previous.focusedSlot === slot ? previous : { ...previous, focusedSlot: slot });
  }

  function showSecondary() {
    pendingFocus.current = 'secondary';
    onStateChange((previous) => showProjectWindowView(previous, 'scene', 'secondary'));
  }

  function hideSecondary() {
    pendingFocus.current = 'primary';
    onStateChange((previous) => ({ ...previous, secondaryView: null, focusedSlot: 'primary' }));
  }

  function previewSize(key: PreferredWindowSize, value: number | null) {
    setPreview((previous) => {
      if (value !== null) return { ...previous, [key]: value };
      const next = { ...previous };
      delete next[key];
      return next;
    });
  }

  function resize(key: PreferredWindowSize, size: number) {
    onStateChange((previous) => ({ ...previous, [key]: size }));
  }

  function toggleMapDrawer() {
    onStateChange((previous) => previous.mode === 'details'
      ? { ...previous, mode: 'both', mapDrawerOpen: true }
      : { ...previous, mapDrawerOpen: !previous.mapDrawerOpen });
  }

  function toggleDetailsSide(key: 'detailsDrawerOpen' | 'inspectorOpen') {
    onStateChange((previous) => previous.mode === 'map'
      ? { ...previous, mode: 'both', [key]: true }
      : { ...previous, [key]: !previous[key] });
  }

  const frameStyle = {
    '--shell-rail-width': `${projectRailWidth}px`,
    '--studio-map-drawer-edge': `${projectRailWidth + (mapDrawerVisible ? fit.mapDrawer.size : 0)}px`,
  } as CSSProperties;
  const panelRows = state.mode === 'map'
    ? 'minmax(0, 1fr) 0px 0px'
    : state.mode === 'details'
      ? `${compactMapHeight}px 0px minmax(0, 1fr)`
      : `${fit.mapHeight.size}px 1px minmax(0, 1fr)`;
  const resizeProps = (key: PreferredWindowSize) => ({
    onPreview: (size: number | null) => previewSize(key, size),
    onResize: (size: number) => resize(key, size),
  });

  return (
    <div className="shell-frame studio-project-window" style={frameStyle}>
      <header className="shell-frame__bar studio-window-bar" data-tauri-drag-region>
        <div className="shell-frame__cell studio-window-bar__controls" data-tauri-drag-region>
          {modes.map(({ mode, name, icon }) => (
            <WindowButton key={mode} name={name} icon={icon} pressed={state.mode === mode}
              onClick={() => onStateChange((previous) => ({ ...previous, mode }))} />
          ))}
          <WindowButton name="back" icon={ArrowLeft} onClick={onBack} />
          <WindowButton name="forward" icon={ArrowRight} onClick={onForward} />
          <WindowButton name="mapDrawer" icon={PanelLeft} pressed={mapDrawerVisible} controls={ids.mapDrawer} onClick={toggleMapDrawer} />
        </div>
        <div className="shell-frame__cell studio-window-bar__project" data-separated={mapDrawerVisible ? '' : undefined} data-tauri-drag-region>
          <h1 className="min-w-0 truncate text-sm font-semibold" data-tauri-drag-region>{projectName}</h1>
          {saveStatus}
          {variant}
        </div>
        <div className="studio-window-bar__commands">{commandPalette}</div>
      </header>
      <nav className="shell-frame__rail shell-scroll-rail studio-window-rail" data-axis="y" aria-label={studioName('details')}>
        {projectWindowViews.map((view) => availableViews.includes(view) ? (
          <WindowButton key={view} name={view} icon={views[view]} placement="right" shortcut={viewShortcuts?.[view]}
            pressed={detailsVisible && shownViews[focusedSlot] === view} controls={ids[focusedSlot]}
            onClick={() => onStateChange((previous) => showProjectWindowView(previous, view, focusedSlot))} />
        ) : <span key={view} className="studio-window-control" aria-hidden="true" />)}
      </nav>
      <main ref={panel} className="shell-frame__content studio-window-panel" style={{ gridTemplateRows: panelRows }}>
        <section id={ids.map} className="studio-window-map" aria-label={studioName('pipelineMap')}>
          <div className="studio-window-map__full" hidden={!fullMapVisible}>
            <aside id={ids.mapDrawer} aria-label={studioName('mapDrawer')} className="studio-window-side"
              hidden={!mapDrawerVisible} style={{ width: fit.mapDrawer.size }}>
              <div className="studio-window-side__contents">{mapDrawer}</div>
              {mapDrawerVisible && (
                <ResizeSeparator label={studioName('mapDrawer')} controls={ids.mapDrawer} axis="x" direction={1}
                  layout={fit.mapDrawer} className="studio-window-resize studio-window-resize--right" {...resizeProps('mapDrawerWidth')} />
              )}
            </aside>
            <div className="min-h-0 min-w-0 flex-1 overflow-auto">{map}</div>
          </div>
          {!fullMapVisible && <div className="studio-window-map__compact">{compactMap}</div>}
        </section>
        <div className="studio-window-panel__divider">
          {state.mode === 'both' && fit.mapHeight.open && (
            <ResizeSeparator label={studioName('panelHeight')} controls={ids.map} axis="y" direction={1}
              layout={fit.mapHeight} className="studio-window-resize studio-window-resize--row" {...resizeProps('mapHeight')} />
          )}
        </div>
        <section id={ids.details} className="studio-window-details" hidden={!detailsVisible} aria-label={studioName('details')}>
          <aside id={ids.detailsDrawer} aria-label={studioName('detailsDrawer')} className="studio-window-side"
            hidden={!detailsDrawerVisible} style={{ width: fit.detailsDrawer.size }}>
            <div className="studio-window-side__contents">{focusedRegion?.drawer}</div>
            {detailsDrawerVisible && (
              <ResizeSeparator label={studioName('detailsDrawer')} controls={ids.detailsDrawer} axis="x" direction={1}
                layout={fit.detailsDrawer} className="studio-window-resize studio-window-resize--right" {...resizeProps('detailsDrawerWidth')} />
            )}
          </aside>
          {slots.map((slot) => {
            const view = shownViews[slot];
            const region = regions[slot];
            if (!view || !region) {
              return slot === 'primary' ? <div key={slot} id={ids[slot]} className="min-w-0 flex-1" /> : null;
            }
            const isFolded = folded && slot !== focusedSlot;
            return (
              <div key={slot} className="studio-window-slot" data-slot={slot} style={{ width: slotWidth(slot) }}>
                {isFolded && (
                  <button type="button" className="studio-window-slot__strip" aria-label={studioName(view)}
                    onClick={() => focusSlot(slot, true)} {...appTip(studioName(view), 'right')}>
                    {studioName(view)}
                  </button>
                )}
                <section id={ids[slot]} ref={(element) => { slotElements.current[slot] = element; }}
                  aria-label={studioName(view)} className="studio-window-slot__view" tabIndex={0}
                  data-focused={focusedSlot === slot ? '' : undefined} hidden={isFolded}
                  onFocusCapture={() => focusSlot(slot)} onPointerDownCapture={() => focusSlot(slot)}>
                  <header className="studio-window-slot__header">
                    <h2 className="min-w-0 truncate text-xs font-semibold">{studioName(view)}</h2>
                    <div className="studio-window-slot__controls">{region.header}</div>
                    {slot === 'primary' && !split && (
                      <WindowButton name="splitView" icon={Columns2} onClick={showSecondary} />
                    )}
                    {slot === 'secondary' && (
                      <WindowButton name="closeSlot" icon={X} onClick={hideSecondary} />
                    )}
                  </header>
                  <div key={view} className="studio-window-slot__body">{region.body}</div>
                </section>
                {slot === 'secondary' && fit.secondary.open && (
                  <ResizeSeparator label={studioName('slotWidth')} controls={ids.secondary} axis="x" direction={-1}
                    layout={fit.secondary} className="studio-window-resize studio-window-resize--left" {...resizeProps('secondaryWidth')} />
                )}
              </div>
            );
          })}
          <aside id={ids.inspector} aria-label={studioName('windowInspector')} className="studio-window-side studio-window-side--inspector"
            hidden={!inspectorVisible} style={{ width: fit.inspector.size }}>
            <div className="studio-window-side__contents">{focusedRegion?.inspector}</div>
            {inspectorVisible && (
              <ResizeSeparator label={studioName('windowInspector')} controls={ids.inspector} axis="x" direction={-1}
                layout={fit.inspector} className="studio-window-resize studio-window-resize--left" {...resizeProps('inspectorWidth')} />
            )}
          </aside>
        </section>
      </main>
      <footer className="shell-frame__footer studio-window-bottom">
        <div className="studio-window-bottom__left">
          <WindowButton name="detailsDrawer" icon={PanelLeft} pressed={detailsDrawerVisible}
            controls={ids.detailsDrawer} onClick={() => toggleDetailsSide('detailsDrawerOpen')} />
        </div>
        <div className="studio-window-bottom__clock">{clock}</div>
        <div className="studio-window-bottom__right">
          {failure}
          {problems}
          {activity}
          <WindowButton name="windowInspector" icon={PanelRight} pressed={inspectorVisible}
            controls={ids.inspector} onClick={() => toggleDetailsSide('inspectorOpen')} />
        </div>
      </footer>
    </div>
  );
}
