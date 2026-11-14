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
import { StudioRegion, StudioRegionBoundary, type StudioRegionState } from '@/components/activity/StudioRegion';
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
import { useStudioTitleBar } from './studioTitleBar';
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
  serviceFailure?: Extract<StudioRegionState, { kind: 'failed' }>;
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
  onBack, onForward, commands, mapCommands, focusRequest, variant, overlay, clock, failure, serviceFailure, problems, activity,
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
  const [mapHasFocus, setMapHasFocus] = useState(false);
  const pointerTransition = useRef<{ slot: ProjectWindowSlot }>();
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
  const sideRegionsAvailable = shownViews[focusedSlot] !== 'results';
  const fit = fitProjectWindow(width, height, {
    ...state,
    secondaryView,
    detailsDrawerOpen: sideRegionsAvailable && state.detailsDrawerOpen,
    inspectorOpen: sideRegionsAvailable && state.inspectorOpen,
  }, preview, {
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

  useEffect(() => {
    let frame = 0;
    function finishPointer() {
      const pending = pointerTransition.current;
      if (!pending) return;
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        if (pointerTransition.current !== pending) return;
        pointerTransition.current = undefined;
        const region = document.activeElement?.closest<HTMLElement>('[data-studio-focus-region]');
        if (region?.dataset.studioFocusRegion !== pending.slot) return;
        onStateChange((state) => state.focusedSlot === pending.slot ? state : { ...state, focusedSlot: pending.slot });
      });
    }
    window.addEventListener('pointerup', finishPointer, true);
    window.addEventListener('pointercancel', finishPointer, true);
    return () => {
      window.removeEventListener('pointerup', finishPointer, true);
      window.removeEventListener('pointercancel', finishPointer, true);
      cancelAnimationFrame(frame);
    };
  }, [onStateChange]);

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
  }, { name: visible ? (id === 'showDrawer' ? 'hideDrawer' : 'hideInspector') : id, enabled: sideRegionsAvailable });
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
  const viewportIds = new Set(mapCommands?.map(({ id }) => id));
  const viewportCommands = fullMapVisible && (mapHasFocus || !commands.some(({ id }) => viewportIds.has(id)))
    ? mapCommands ?? [] : [];
  const paletteCommands = [
    ...commands.filter(({ id }) => viewportCommands.length === 0 || !viewportIds.has(id)), ...viewportCommands,
    ...layoutModes.map(({ command }) => command), steps, drawer, inspector, splitCommand, switchCommand,
    back, forward, ...tabCommands,
    bindStudioCommand('nextRegion', () => cycleRegion(1)),
    bindStudioCommand('previousRegion', () => cycleRegion(-1)),
  ];
  const shortcutCommands = useRef({ palette, paletteCommands, paletteScope });
  shortcutCommands.current = { palette, paletteCommands, paletteScope };
  useStudioNativeMenu([palette, ...paletteCommands], paletteScope !== null);
  const titleBar = useStudioTitleBar(projectName);
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
      // WebKit must leave the event unhandled for the native menu's key equivalent.
      if (nativeOwnsShortcut(command, event)) return;
      event.preventDefault();
      if (!commandInContext(command, context)) return;
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
  const offline = serviceFailure && <StudioRegion state={serviceFailure} />;
  const connectedBody = (body: ReactNode) => (
    <>
      <div className="studio-window-region-content" hidden={!!offline}>{body}</div>
      {offline}
    </>
  );

  return (
    <div ref={root} className="shell-frame studio-project-window" style={frameStyle}
      onFocusCapture={(event) => {
        if (!event.target.closest('[role="dialog"]')) {
          setMapHasFocus(event.target.closest('[data-studio-focus-region="map"]') !== null);
        }
      }}>
      <header className="shell-frame__bar studio-window-bar" data-studio-focus-region="top" tabIndex={-1} data-studio-drag-region {...titleBar}>
        <div className="shell-frame__cell studio-window-bar__controls" data-studio-drag-region>
          {layoutModes.map(({ mode, icon, command }) => (
            <WindowButton key={mode} command={command} icon={icon} pressed={state.mode === mode} />
          ))}
          <WindowButton command={back} icon={ArrowLeft} />
          <WindowButton command={forward} icon={ArrowRight} />
          <WindowButton command={steps} icon={PanelLeft} pressed={mapDrawerVisible} controls={ids.mapDrawer} />
        </div>
        <div className="shell-frame__cell studio-window-bar__project" data-separated="" data-studio-drag-region>
          <h1 className="min-w-0 truncate text-sm font-semibold" data-studio-drag-region>{projectName}</h1>
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
            <div data-studio-focus-region="map" tabIndex={-1} className="min-h-0 min-w-0 flex-1 overflow-hidden">
              <StudioRegionBoundary name="pipelineMap">{connectedBody(map)}</StudioRegionBoundary>
            </div>
            <aside id={ids.mapDrawer} data-studio-focus-region="steps" tabIndex={-1} aria-label={studioName('mapDrawer')} className="studio-window-side"
              hidden={!mapDrawerVisible} style={{ width: fit.mapDrawer.size }}>
              <div className="studio-window-side__contents">
                <StudioRegionBoundary name="mapDrawer">{connectedBody(mapDrawer)}</StudioRegionBoundary>
              </div>
              {mapDrawerVisible && (
                <ResizeSeparator label={studioName('mapDrawer')} controls={ids.mapDrawer} axis="x" direction={1}
                  layout={fit.mapDrawer} className="studio-window-resize studio-window-resize--right" {...resizeProps('mapDrawerWidth')} />
              )}
            </aside>
          </div>
          {!fullMapVisible && <div className="studio-window-map__compact" data-studio-focus-region="map" tabIndex={-1}>
            <StudioRegionBoundary name="pipelineMap">{connectedBody(compactMap)}</StudioRegionBoundary>
          </div>}
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
            <div className="studio-window-side__contents" data-populated={focusedRegion?.drawer ? '' : undefined}>
              <StudioRegionBoundary name="detailsDrawer">
                {focusedRegion?.drawer && connectedBody(focusedRegion.drawer)}
              </StudioRegionBoundary>
            </div>
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
                  onFocusCapture={() => { if (pointerTransition.current?.slot !== slot) focusSlot(slot); }}
                  onPointerDownCapture={() => {
                    const crossesResults = slot !== focusedSlot && (view === 'results' || !sideRegionsAvailable);
                    // Keep the clicked control under the pointer until its click is dispatched.
                    pointerTransition.current = crossesResults ? { slot } : undefined;
                    if (!crossesResults) focusSlot(slot);
                  }}
                  onClickCapture={() => {
                    pointerTransition.current = undefined;
                    focusSlot(slot);
                  }}>
                  <header className="studio-window-slot__header">
                    <h2 className="studio-window-slot__name">{studioName(view)}</h2>
                    <div className="studio-window-slot__controls" hidden={!!offline && view !== 'runs'}>
                      <StudioRegionBoundary key={view} name={view}>{region.header}</StudioRegionBoundary>
                    </div>
                    {slot === 'primary' && !split && (
                      <WindowButton command={splitCommand} icon={Columns2} />
                    )}
                    {slot === 'secondary' && (
                      <WindowButton command={splitCommand} icon={X} />
                    )}
                  </header>
                  <div key={view} className="studio-window-slot__body">
                    <StudioRegionBoundary name={view}>{connectedBody(region.body)}</StudioRegionBoundary>
                  </div>
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
            <div className="studio-window-side__contents" data-populated={focusedRegion?.inspector ? '' : undefined}>
              <StudioRegionBoundary name="windowInspector">
                {focusedRegion?.inspector && connectedBody(focusedRegion.inspector)}
              </StudioRegionBoundary>
            </div>
            {inspectorVisible && (
              <ResizeSeparator label={studioName('windowInspector')} controls={ids.inspector} axis="x" direction={-1}
                layout={fit.inspector} className="studio-window-resize studio-window-resize--left" {...resizeProps('inspectorWidth')} />
            )}
          </aside>
        </section>
      </main>
      <footer className="shell-frame__footer studio-window-bottom" data-studio-focus-region="bottom" tabIndex={-1}>
        <StudioRegionBoundary name="bottomBar">
          <div className="studio-window-bottom__left">
            <WindowButton command={drawer} icon={PanelLeft} pressed={detailsDrawerVisible} controls={ids.detailsDrawer} />
          </div>
          <div className="studio-window-bottom__clock" hidden={!!offline}>{clock}</div>
          <div className="studio-window-bottom__right">
            {failure}
            {problems}
            {activity}
            <WindowButton command={inspector} icon={PanelRight} pressed={inspectorVisible} controls={ids.inspector} />
          </div>
        </StudioRegionBoundary>
      </footer>
    </div>
  );
}
