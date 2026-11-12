import {
  ArrowLeft, ArrowRight, BarChart3, Columns2, Command, Crosshair, FileText, GitBranch, Image, List,
  Map as SceneIcon, PanelBottom, PanelLeft, PanelRight, PanelTop, Rows2, Settings2, Target, Workflow, X,
  type LucideIcon,
} from 'lucide-react';
import {
  useEffect, useId, useLayoutEffect, useRef, useState,
  type CSSProperties, type Dispatch, type ReactNode, type SetStateAction,
} from 'react';
import { appTip, installTooltips } from '@window-shell/browser/tooltip';
import { ResizeSeparator } from '@/components/layout/ResizeSeparator';
import { studioName } from '@/components/activity/studioVocabulary';
import { useElementSize } from '@/hooks/useElementSize';
import {
  compactMapHeight, fitProjectWindow, projectRailWidth, type PreferredWindowSize, type ProjectWindowSizePreview,
} from './projectWindowLayout';
import {
  projectWindowSlotView, projectWindowViews, showProjectWindowView,
  type ProjectWindowSlot, type ProjectWindowState, type ProjectWindowView,
} from './projectWindowState';
import { StudioCommandPalette } from './StudioCommandPalette';
import { bindStudioCommand, commandAttributes, commandInContext, matchesShortcut, studioCommandContext, type StudioCommand } from './studioCommands';
import { nativeOwnsShortcut, useStudioNativeMenu } from './studioNativeMenu';
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
  commands: readonly StudioCommand[];
  mapCommands?: readonly StudioCommand[];
  focusRequest?: number;
  variant?: ReactNode;
  overlay?: ReactNode;
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
  { mode: 'map', id: 'showMap', icon: PanelTop },
  { mode: 'both', id: 'showBoth', icon: Rows2 },
  { mode: 'details', id: 'showDetails', icon: PanelBottom },
] as const;

const slots = ['primary', 'secondary'] as const;
const regionOrder = ['top', 'map', 'steps', 'rail', 'drawer', ...slots, 'inspector', 'bottom'] as const;
type WindowRegion = typeof regionOrder[number];

function WindowButton({ command, icon: Icon, pressed, controls, tab }: {
  command: StudioCommand;
  icon: LucideIcon;
  pressed?: boolean;
  controls?: string;
  tab?: { id: string; tabbable: boolean };
}) {
  return (
    <button type="button" {...commandAttributes(command, tab ? 'right' : 'bottom')}
      role={tab ? 'tab' : undefined} id={tab?.id} tabIndex={tab ? (tab.tabbable ? 0 : -1) : undefined}
      aria-selected={tab ? pressed : undefined} aria-pressed={!tab ? pressed : undefined}
      aria-controls={controls} disabled={!command.enabled} onClick={command.invoke} className="studio-window-control">
      <Icon aria-hidden="true" className="size-4" />
    </button>
  );
}

function visibleRegion(root: HTMLElement | null, region: WindowRegion) {
  return Array.from(root?.querySelectorAll<HTMLElement>(`[data-studio-focus-region="${region}"]`) ?? [])
    .find((element) => !element.closest('[hidden], [inert]') && element.getClientRects().length > 0);
}

function focusRegion(root: HTMLElement | null, region: WindowRegion) {
  const target = visibleRegion(root, region);
  (target?.querySelector<HTMLElement>('[role="tab"][tabindex="0"]') ?? target)?.focus();
  return !!target;
}

export function ProjectWindow({
  state, onStateChange, projectName, saveStatus, availableViews, renderView, map, mapDrawer, compactMap,
  onBack, onForward, commands, mapCommands, focusRequest, variant, overlay, clock, failure, problems, activity,
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
  const root = useRef<HTMLDivElement>(null);
  const { ref: panel, width, height } = useElementSize<HTMLElement>();
  const [preview, setPreview] = useState<ProjectWindowSizePreview>({});
  const [requestedFocus, setRequestedFocus] = useState<{ region: WindowRegion }>();
  const [paletteScope, setPaletteScope] = useState<'project' | 'activity' | null>(null);
  const [railFocus, setRailFocus] = useState<ProjectWindowView | null>(null);
  const handledFocusRequest = useRef(focusRequest);
  const moveFocus = (region: WindowRegion) => setRequestedFocus({ region });
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
  const fit = fitProjectWindow(width, height, secondaryView ? state : { ...state, secondaryView: null }, preview, {
    drawer: Boolean(focusedRegion?.drawer), inspector: Boolean(focusedRegion?.inspector),
  });
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
    if (requestedFocus && !focusRegion(root.current, requestedFocus.region)) {
      focusRegion(root.current, detailsVisible ? focusedSlot : 'map');
    }
  }, [requestedFocus]);

  useLayoutEffect(() => {
    if (focusRequest === handledFocusRequest.current) return;
    handledFocusRequest.current = focusRequest;
    focusRegion(root.current, focusedSlot);
  }, [focusRequest, focusedSlot]);

  function focusSlot(slot: ProjectWindowSlot) {
    onStateChange((previous) => previous.focusedSlot === slot ? previous : { ...previous, focusedSlot: slot });
  }

  function activateView(view: ProjectWindowView) {
    onStateChange((previous) => showProjectWindowView(previous, view, focusedSlot));
    moveFocus(focusedSlot);
  }

  function switchSlot() {
    const other = focusedSlot === 'primary' ? 'secondary' : 'primary';
    onStateChange((previous) => ({ ...previous, mode: previous.mode === 'map' ? 'both' : previous.mode, focusedSlot: other }));
    moveFocus(other);
  }

  function splitDetails() {
    onStateChange((previous) => split
      ? { ...previous, secondaryView: null, focusedSlot: 'primary' }
      : showProjectWindowView(previous, 'scene', 'secondary'));
    moveFocus(split ? 'primary' : 'secondary');
  }

  function cycleRegion(direction: number) {
    const visible = regionOrder.filter((region) => visibleRegion(root.current, region));
    const current = (document.activeElement?.closest('[data-studio-focus-region]') as HTMLElement | null)?.dataset.studioFocusRegion;
    const index = visible.findIndex((region) => region === current);
    if (!visible.length) return;
    const next = index < 0 ? (direction > 0 ? 0 : visible.length - 1) : (index + direction + visible.length) % visible.length;
    focusRegion(root.current, visible[next]);
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

  const layoutModes = modes.map(({ mode, id, icon }) => ({
    mode, icon, command: bindStudioCommand(id, () => {
      onStateChange((previous) => ({ ...previous, mode }));
      moveFocus(mode === 'details' ? focusedSlot : 'map');
    }),
  }));
  const back = bindStudioCommand('back', () => onBack?.(), { enabled: !!onBack });
  const forward = bindStudioCommand('forward', () => onForward?.(), { enabled: !!onForward });
  const steps = bindStudioCommand('showSteps', () => {
    onStateChange((previous) => ({
      ...previous, mode: previous.mode === 'details' ? 'both' : previous.mode, mapDrawerOpen: !mapDrawerVisible,
    }));
    moveFocus(mapDrawerVisible ? 'map' : 'steps');
  }, { name: mapDrawerVisible ? 'hideSteps' : 'showSteps' });
  const side = (id: 'showDrawer' | 'showInspector', key: 'detailsDrawerOpen' | 'inspectorOpen',
    region: 'drawer' | 'inspector', visible: boolean) => bindStudioCommand(id, () => {
    onStateChange((previous) => ({
      ...previous, mode: previous.mode === 'map' ? 'both' : previous.mode, [key]: !visible,
    }));
    moveFocus(visible ? focusedSlot : region);
  }, { name: visible ? (id === 'showDrawer' ? 'hideDrawer' : 'hideInspector') : id });
  const drawer = side('showDrawer', 'detailsDrawerOpen', 'drawer', detailsDrawerVisible);
  const inspector = side('showInspector', 'inspectorOpen', 'inspector', inspectorVisible);
  const splitCommand = bindStudioCommand('splitDetails', splitDetails,
    { name: split ? 'closeSlot' : 'splitDetails', enabled: !!availableViews.length });
  const switchCommand = bindStudioCommand('switchSlot', switchSlot, { enabled: split });
  const tabCommands = availableViews.map((view) => bindStudioCommand(view, () => activateView(view)));
  const tabView = railFocus && availableViews.includes(railFocus) ? railFocus : shownViews[focusedSlot] ?? availableViews[0];
  const tabId = (view: ProjectWindowView) => `${id}-tab-${view}`;
  const palette = bindStudioCommand('commandPalette', () => {
    const scope = document.activeElement?.closest('[data-studio-activity-panel]') ? 'activity' : 'project';
    setPaletteScope((open) => open ? null : scope);
  });
  const paletteCommands = [
    ...commands, ...(fullMapVisible ? mapCommands ?? [] : []),
    ...layoutModes.map(({ command }) => command), steps, drawer, inspector, splitCommand, switchCommand,
    back, forward, ...tabCommands,
    bindStudioCommand('nextRegion', () => cycleRegion(1)),
    bindStudioCommand('previousRegion', () => cycleRegion(-1)),
  ];
  const shortcutCommands = useRef({ palette, paletteCommands, paletteScope });
  shortcutCommands.current = { palette, paletteCommands, paletteScope };
  useStudioNativeMenu([palette, ...paletteCommands], paletteScope !== null);
  useEffect(() => {
    function handleKey(event: KeyboardEvent) {
      if (event.defaultPrevented || event.isComposing || event.repeat) return;
      const { palette, paletteCommands, paletteScope } = shortcutCommands.current;
      const context = studioCommandContext(event.target, paletteScope !== null);
      const command = matchesShortcut(event, palette.shortcut) ? palette
        : paletteCommands.find((item) => matchesShortcut(event, item.shortcut));
      if (!command) return;
      if (command.id === 'addComponent' && !(event.target instanceof Element &&
          event.target.closest('[role="tabpanel"], [data-studio-focus-region="drawer"]'))) return;
      if (!commandInContext(command, { ...context, activity: false })) return;
      event.preventDefault();
      if (!commandInContext(command, context) || nativeOwnsShortcut(command, event)) return;
      command.invoke();
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

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
    <div ref={root} className="shell-frame studio-project-window" style={frameStyle}>
      <header className="shell-frame__bar studio-window-bar" data-studio-focus-region="top" tabIndex={-1} data-tauri-drag-region>
        <div className="shell-frame__cell studio-window-bar__controls" data-tauri-drag-region>
          {layoutModes.map(({ mode, icon, command }) => (
            <WindowButton key={mode} command={command} icon={icon} pressed={state.mode === mode} />
          ))}
          <WindowButton command={back} icon={ArrowLeft} />
          <WindowButton command={forward} icon={ArrowRight} />
          <WindowButton command={steps} icon={PanelLeft} pressed={mapDrawerVisible} controls={ids.mapDrawer} />
        </div>
        <div className="shell-frame__cell studio-window-bar__project" data-separated="" data-tauri-drag-region>
          <h1 className="min-w-0 truncate text-sm font-semibold" data-tauri-drag-region>{projectName}</h1>
          {saveStatus}
          {variant}
        </div>
        <div className="studio-window-bar__commands">
          <WindowButton command={palette} icon={Command} />
          {paletteScope && <StudioCommandPalette
            commands={paletteScope === 'activity' ? paletteCommands.filter((command) => !command.selectionScoped) : paletteCommands}
            close={() => setPaletteScope(null)} />}
          {overlay}
        </div>
      </header>
      <main ref={panel} className="shell-frame__content studio-window-panel" style={{ gridTemplateRows: panelRows }}>
        <section id={ids.map} className="studio-window-map" aria-label={studioName('pipelineMap')}>
          <div className="studio-window-map__full" hidden={!fullMapVisible}>
            <div data-studio-focus-region="map" tabIndex={-1} className="min-h-0 min-w-0 flex-1 overflow-hidden">{map}</div>
            <aside id={ids.mapDrawer} data-studio-focus-region="steps" tabIndex={-1} aria-label={studioName('mapDrawer')} className="studio-window-side"
              hidden={!mapDrawerVisible} style={{ width: fit.mapDrawer.size }}>
              <div className="studio-window-side__contents">{mapDrawer}</div>
              {mapDrawerVisible && (
                <ResizeSeparator label={studioName('mapDrawer')} controls={ids.mapDrawer} axis="x" direction={1}
                  layout={fit.mapDrawer} className="studio-window-resize studio-window-resize--right" {...resizeProps('mapDrawerWidth')} />
              )}
            </aside>
          </div>
          {!fullMapVisible && <div className="studio-window-map__compact" data-studio-focus-region="map" tabIndex={-1}>{compactMap}</div>}
        </section>
        <div className="studio-window-panel__divider">
          {state.mode === 'both' && fit.mapHeight.open && (
            <ResizeSeparator label={studioName('panelHeight')} controls={ids.map} axis="y" direction={1}
              layout={fit.mapHeight} className="studio-window-resize studio-window-resize--row" {...resizeProps('mapHeight')} />
          )}
        </div>
        <nav className="shell-scroll-rail studio-window-rail" data-axis="y" aria-label={studioName('details')}
          data-studio-focus-region="rail" tabIndex={-1}>
          <div role="tablist" aria-label={studioName('details')} aria-orientation="vertical" className="studio-window-tabs"
            onFocusCapture={(event) => {
              const id = event.target.dataset.commandId;
              const view = availableViews.find((item) => item === id);
              if (view) setRailFocus(view);
            }}
            onBlurCapture={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget)) setRailFocus(null);
            }}
            onKeyDown={(event) => {
              const tabs = [...event.currentTarget.querySelectorAll<HTMLElement>('[role="tab"]')];
              const index = tabs.findIndex((tab) => tab === event.target);
              if (index < 0) return;
              const destination = { ArrowUp: (index - 1 + tabs.length) % tabs.length,
                ArrowDown: (index + 1) % tabs.length, Home: 0, End: tabs.length - 1 }[event.key];
              if (destination === undefined) return;
              event.preventDefault();
              tabs[destination]?.focus();
            }}>
            {projectWindowViews.map((view) => {
              const command = tabCommands.find(({ id }) => id === view);
              return command ? (
                <WindowButton key={view} command={command} icon={views[view]} controls={ids[focusedSlot]}
                  pressed={detailsVisible && shownViews[focusedSlot] === view} tab={{ id: tabId(view), tabbable: tabView === view }} />
              ) : <span key={view} className="studio-window-control" aria-hidden="true" />;
            })}
          </div>
        </nav>
        <section id={ids.details} className="studio-window-details" hidden={!detailsVisible} aria-label={studioName('details')}>
          <aside id={ids.detailsDrawer} data-studio-focus-region="drawer" tabIndex={-1} aria-label={studioName('detailsDrawer')} className="studio-window-side"
            hidden={!detailsDrawerVisible} style={{ width: fit.detailsDrawer.size }}>
            <div className="studio-window-side__contents" data-populated={focusedRegion?.drawer ? '' : undefined}>{focusedRegion?.drawer}</div>
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
                    onClick={() => { focusSlot(slot); moveFocus(slot); }} {...appTip(studioName(view), 'right')}>
                    {studioName(view)}
                  </button>
                )}
                <section id={ids[slot]} role="tabpanel" aria-labelledby={tabId(view)} data-studio-focus-region={slot} className="studio-window-slot__view" tabIndex={0}
                  data-focused={focusedSlot === slot ? '' : undefined} hidden={isFolded}
                  onFocusCapture={() => focusSlot(slot)} onPointerDownCapture={() => focusSlot(slot)}>
                  <header className="studio-window-slot__header">
                    <h2 className="studio-window-slot__name">{studioName(view)}</h2>
                    <div className="studio-window-slot__controls">{region.header}</div>
                    {slot === 'primary' && !split && (
                      <WindowButton command={splitCommand} icon={Columns2} />
                    )}
                    {slot === 'secondary' && (
                      <WindowButton command={splitCommand} icon={X} />
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
          <aside id={ids.inspector} data-studio-focus-region="inspector" tabIndex={-1} aria-label={studioName('windowInspector')} className="studio-window-side studio-window-side--inspector"
            hidden={!inspectorVisible} style={{ width: fit.inspector.size }}>
            <div className="studio-window-side__contents" data-populated={focusedRegion?.inspector ? '' : undefined}>{focusedRegion?.inspector}</div>
            {inspectorVisible && (
              <ResizeSeparator label={studioName('windowInspector')} controls={ids.inspector} axis="x" direction={-1}
                layout={fit.inspector} className="studio-window-resize studio-window-resize--left" {...resizeProps('inspectorWidth')} />
            )}
          </aside>
        </section>
      </main>
      <footer className="shell-frame__footer studio-window-bottom" data-studio-focus-region="bottom" tabIndex={-1}>
        <div className="studio-window-bottom__left">
          <WindowButton command={drawer} icon={PanelLeft} pressed={detailsDrawerVisible} controls={ids.detailsDrawer} />
        </div>
        <div className="studio-window-bottom__clock">{clock}</div>
        <div className="studio-window-bottom__right">
          {failure}
          {problems}
          {activity}
          <WindowButton command={inspector} icon={PanelRight} pressed={inspectorVisible} controls={ids.inspector} />
        </div>
      </footer>
    </div>
  );
}
