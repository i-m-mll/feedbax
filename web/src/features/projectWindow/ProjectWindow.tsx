import {
  BarChart3, Columns2, Crosshair, FileText, GitBranch, Image, List,
  Map as SceneIcon, PanelBottom, PanelTop, Rows2, Settings2, Target, Workflow, X,
  type LucideIcon,
} from 'lucide-react';
import { StudioLogo } from '../../components/StudioLogo';
import {
  useEffect, useId, useLayoutEffect, useRef, useState,
  type CSSProperties, type Dispatch, type ReactNode, type SetStateAction,
} from 'react';
import { appTip, installTooltips } from '@window-shell/browser/tooltip';
import { canvasViewportIcons } from '@/components/canvas/CanvasStrip';
import type { CanvasViewportCommand } from '@/components/canvas/StudioCanvas';
import { ResizeSeparator } from '@/components/layout/ResizeSeparator';
import { Button, IconButton } from '@/components/ui/Button';
import { CommandBar } from '@/components/ui/CommandBar';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { StudioRegion, StudioRegionBoundary } from '@/components/activity/StudioRegion';
import { studioName } from '@/components/activity/studioVocabulary';
import { useElementSize } from '@/hooks/useElementSize';
import {
  compactMapHeight, fitProjectWindow, projectRailWidth, type PreferredWindowSize, type ProjectWindowSizePreview,
} from './projectWindowLayout';
import {
  closeProjectWindowSlot, defaultProjectWindowState, projectWindowSlotView, projectWindowViews, showProjectWindowView,
  type ProjectWindowMode, type ProjectWindowSlot, type ProjectWindowState, type ProjectWindowView,
} from './projectWindowState';
import type { DocumentPickerModel } from './DocumentPicker';
import { SlotHeader } from './SlotHeader';
import { StudioCommandPalette } from './StudioCommandPalette';
import { StudioConnectionStatus } from './StudioConnectionStatus';
import { StudioSettings } from './StudioSettings';
import { bindStudioCommand, commandAttributes, commandInContext, matchesShortcut, studioCommandContext, type StudioCommand } from './studioCommands';
import { nativeOwnsShortcut, useStudioNativeMenu } from './studioNativeMenu';
import { useStudioTitleBar } from './studioTitleBar';
import { appearanceCommands } from '@/appearance/appearanceCommands';
import './projectWindow.css';

export interface ProjectWindowViewRegion {
  body: ReactNode;
  picker?: DocumentPickerModel;
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
  mapCommands?: readonly CanvasViewportCommand[];
  focusRequest?: number;
  variant?: ReactNode;
  overlay?: ReactNode;
  clock?: ReactNode;
  failure?: ReactNode;
  problems?: ReactNode;
  activity?: ReactNode;
}

/** The glyph of each view, and of the document kind a view shows. */
export const viewGlyphs = {
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

function WindowButton({ command, icon: Icon = command.glyph!, pressed, controls, tab, placement = tab ? 'right' : 'bottom' }: {
  command: StudioCommand;
  icon?: LucideIcon;
  pressed?: boolean;
  controls?: string;
  tab?: { id: string; tabbable: boolean };
  placement?: 'bottom' | 'right';
}) {
  return (
    <IconButton label={command.label} icon={<Icon aria-hidden="true" />} {...commandAttributes(command, placement)}
      role={tab ? 'tab' : undefined} id={tab?.id} tabIndex={tab ? (tab.tabbable ? 0 : -1) : undefined}
      aria-selected={tab ? pressed : undefined} pressed={!tab ? pressed : undefined}
      aria-controls={controls} disabled={!command.enabled} onClick={command.invoke} className="studio-window-control" />
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
  const [settingsOpen, setSettingsOpen] = useState(false);
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
  const sides = (region: ProjectWindowViewRegion | null) => ({ drawer: !!region?.drawer, inspector: !!region?.inspector });
  const focusedSides = sides(focusedRegion);
  const mode = availableViews.length ? state.mode : 'map';
  const fit = fitProjectWindow(width, height, {
    ...state,
    mode,
    secondaryView,
    detailsDrawerOpen: focusedSides.drawer && state.detailsDrawerOpen,
    inspectorOpen: focusedSides.inspector && state.inspectorOpen,
  }, preview);
  const detailsVisible = mode !== 'map';
  const fullMapVisible = mode === 'map' || (mode === 'both' && fit.mapHeight.open);
  const mapDrawerVisible = fullMapVisible && fit.mapDrawer.open;
  // Before the panel is measured, its sides follow the saved state rather than a zero-width fit.
  const measured = width > 0;
  const detailsDrawerVisible = detailsVisible && (measured ? fit.detailsDrawer.open : focusedSides.drawer && state.detailsDrawerOpen);
  const inspectorVisible = detailsVisible && (measured ? fit.inspector.open : focusedSides.inspector && state.inspectorOpen);
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

  function closeSlot(slot: ProjectWindowSlot) {
    onStateChange((previous) => closeProjectWindowSlot(previous, slot));
    moveFocus('primary');
  }

  function splitDetails() {
    if (split) return closeSlot(focusedSlot);
    onStateChange((previous) => showProjectWindowView(previous, 'scene', 'secondary'));
    moveFocus('secondary');
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

  const layoutModes = modes.map(({ mode: target, id, icon }) => ({
    mode: target, icon, command: bindStudioCommand(id, () => {
      onStateChange((previous) => ({ ...previous, mode: target }));
      moveFocus(target === 'details' ? focusedSlot : 'map');
    }, { enabled: target === 'map' || availableViews.length > 0 }),
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
  }, { name: visible ? (id === 'showDrawer' ? 'hideDrawer' : 'hideInspector') : id, enabled: focusedSides[region === 'drawer' ? 'drawer' : 'inspector'] });
  const drawer = side('showDrawer', 'detailsDrawerOpen', 'drawer', detailsDrawerVisible);
  const inspector = side('showInspector', 'inspectorOpen', 'inspector', inspectorVisible);
  const splitCommand = bindStudioCommand('splitDetails', splitDetails,
    { name: split ? 'closeSlot' : 'splitDetails', enabled: !!availableViews.length });
  const closeCommand = (slot: ProjectWindowSlot) => bindStudioCommand('splitDetails', () => closeSlot(slot),
    { name: 'closeSlot', keyboard: false });
  const switchCommand = bindStudioCommand('switchSlot', switchSlot, { enabled: split });
  const tabCommands = projectWindowViews.map((view) => bindStudioCommand(view, () => activateView(view),
    { enabled: availableViews.includes(view) }));
  const tabView = railFocus ?? shownViews[focusedSlot] ?? availableViews[0];
  const tabId = (view: ProjectWindowView) => `${id}-tab-${view}`;
  const palette = bindStudioCommand('commandPalette', () => {
    const scope = document.activeElement?.closest('[data-studio-activity-panel]') ? 'activity' : 'project';
    setPaletteScope((open) => open ? null : scope);
  });
  const settings = bindStudioCommand('settings', () => setSettingsOpen((open) => !open), { name: 'openSettings' });
  const viewportIds = new Set<StudioCommand['id']>(mapCommands?.map(({ id }) => id));
  const separator = mode === 'map' ? '100%' : `${mode === 'both' ? fit.mapHeight.size : compactMapHeight}px`;
  const viewportCommands = fullMapVisible && (mapHasFocus || !commands.some(({ id }) => viewportIds.has(id)))
    ? mapCommands ?? [] : [];
  const paletteCommands = [
    ...commands.filter(({ id }) => viewportCommands.length === 0 || !viewportIds.has(id)), ...viewportCommands,
    ...layoutModes.map(({ command }) => command), steps, drawer, inspector, splitCommand, switchCommand,
    back, forward, ...tabCommands,
    bindStudioCommand('nextRegion', () => cycleRegion(1)),
    bindStudioCommand('previousRegion', () => cycleRegion(-1)),
    settings, ...appearanceCommands(),
  ];
  const shortcutCommands = useRef({ palette, paletteCommands, paletteScope });
  shortcutCommands.current = { palette, paletteCommands, paletteScope };
  useStudioNativeMenu([palette, ...paletteCommands], paletteScope !== null);
  const bar = useRef<HTMLElement>(null);
  useStudioTitleBar(bar, projectName);
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
  const panelRows = mode === 'map'
    ? 'minmax(0, 1fr) 0px 0px'
    : mode === 'details'
      ? `${compactMapHeight}px 0px minmax(0, 1fr)`
      : `${fit.mapHeight.size}px 1px minmax(0, 1fr)`;
  const resizeProps = (key: PreferredWindowSize) => ({
    onPreview: (size: number | null) => previewSize(key, size),
    onResize: (size: number) => resize(key, size),
  });
  const focusedOwner = `${focusedSlot}:${shownViews[focusedSlot]}`;
  const regionBody = (name: StudioCommand['name'], body: ReactNode, owner?: string) => (
    <div className="studio-window-region-content">
      <StudioRegionBoundary key={owner} name={name}>{body}</StudioRegionBoundary>
    </div>
  );

  return (
    <div ref={root} className="shell-frame studio-project-window" style={frameStyle}
      onFocusCapture={(event) => {
        if (!event.target.closest('[role="dialog"]')) {
          setMapHasFocus(event.target.closest('[data-studio-focus-region="map"]') !== null);
        }
      }}>
      <header ref={bar} className="shell-frame__bar studio-window-bar" data-studio-focus-region="top" tabIndex={-1} data-shell-drag-region>
        <div className="shell-frame__cell shell-titlebar__leading studio-window-bar__controls">
          <span className="shell-titlebar__logo"><StudioLogo /></span>
          <SegmentedControl<ProjectWindowMode> aria-label={studioName('layoutMode')} value={mode}
            onValueChange={(target) => layoutModes.find((item) => item.mode === target)?.command.invoke()}
            items={layoutModes.map(({ mode: target, icon: Icon, command }) => ({
              value: target, label: command.label, icon: <Icon aria-hidden="true" />, disabled: !command.enabled,
              attributes: commandAttributes(command),
            }))} />
          <CommandBar commands={[steps, back, forward]} moreLabel={studioName('more')} itemClassName="studio-window-control"
            itemProps={(command) => ({
              ...commandAttributes(command),
              ...(command === steps ? { 'aria-pressed': mapDrawerVisible, 'aria-controls': ids.mapDrawer } : {}),
            })} />
        </div>
        <div className="shell-frame__cell studio-window-bar__project" data-separated="">
          <h1 className="min-w-0 truncate text-sm font-semibold">{projectName}</h1>
          {saveStatus}
          {variant}
        </div>
        <div className="studio-window-bar__commands">
          <WindowButton command={palette} />
          {paletteScope && <StudioCommandPalette
            commands={paletteScope === 'activity' ? paletteCommands.filter((command) => !command.selectionScoped) : paletteCommands}
            close={() => setPaletteScope(null)} />}
          {settingsOpen && <StudioSettings close={() => setSettingsOpen(false)} />}
          {overlay}
        </div>
      </header>
      <main ref={panel} className="shell-frame__content studio-window-panel" style={{ gridTemplateRows: panelRows }}>
        <section id={ids.map} className="studio-window-map" aria-label={studioName('pipelineMap')}>
          <div className="studio-window-map__full" hidden={!fullMapVisible}>
            <div data-studio-focus-region="map" tabIndex={-1} className="min-h-0 min-w-0 flex-1 overflow-hidden">
              {regionBody('pipelineMap', map)}
            </div>
            <aside id={ids.mapDrawer} data-studio-focus-region="steps" tabIndex={-1} aria-label={studioName('mapDrawer')} className="studio-window-side"
              hidden={!mapDrawerVisible} style={{ width: fit.mapDrawer.size }}>
              <div className="studio-window-side__contents">
                {regionBody('mapDrawer', mapDrawer)}
              </div>
              {mapDrawerVisible && (
                <ResizeSeparator label={studioName('mapDrawer')} controls={ids.mapDrawer} axis="x" direction={1}
                  layout={fit.mapDrawer} className="studio-window-resize studio-window-resize--right" {...resizeProps('mapDrawerWidth')} />
              )}
            </aside>
          </div>
          {!fullMapVisible && <div className="studio-window-map__compact" data-studio-focus-region="map" tabIndex={-1}>
            {regionBody('pipelineMap', compactMap)}
          </div>}
        </section>
        <div className="studio-window-panel__divider">
          {mode === 'both' && fit.mapHeight.open && (
            <ResizeSeparator label={studioName('panelHeight')} controls={ids.map} axis="y" direction={1}
              layout={fit.mapHeight} className="studio-window-resize studio-window-resize--row" {...resizeProps('mapHeight')} />
          )}
        </div>
        <nav className="shell-rail studio-window-rail" data-separated="" aria-label={studioName('details')}
          data-studio-focus-region="rail" tabIndex={-1} style={{ '--shell-rail-separator': separator } as CSSProperties}>
          {fullMapVisible && mapCommands && (
            <div role="toolbar" aria-label={studioName('pipelineMap')} aria-orientation="vertical" className="studio-window-rail__map">
              {mapCommands.map((command) => (
                <WindowButton key={command.id} command={command} icon={canvasViewportIcons[command.id]} placement="right" />
              ))}
            </div>
          )}
          {detailsVisible && <div role="tablist" aria-label={studioName('details')} aria-orientation="vertical"
            className="shell-scroll-rail studio-window-tabs" data-axis="y"
            onFocusCapture={(event) => {
              const id = event.target.dataset.commandId;
              const view = projectWindowViews.find((item) => item === id);
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
            {tabCommands.map((command) => {
              const view = command.id;
              const selected = shownViews[focusedSlot] === view;
              return (
                <WindowButton key={view} command={command} icon={viewGlyphs[view]} controls={selected ? ids[focusedSlot] : undefined}
                  pressed={selected} tab={{ id: tabId(view), tabbable: tabView === view }} />
              );
            })}
          </div>}
        </nav>
        <section id={ids.details} className="studio-window-details" hidden={!detailsVisible} aria-label={studioName('details')}>
          <aside id={ids.detailsDrawer} data-studio-focus-region="drawer" tabIndex={-1} aria-label={studioName('detailsDrawer')} className="studio-window-side"
            hidden={!detailsDrawerVisible} style={{ width: fit.detailsDrawer.size }}>
            <div className="studio-window-side__contents" data-populated={focusedRegion?.drawer ? '' : undefined}>
              {regionBody('detailsDrawer', focusedRegion?.drawer, focusedOwner)}
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
                  <Button variant="ghost" className="studio-window-slot__strip" aria-label={studioName(view)}
                    onClick={() => { focusSlot(slot); moveFocus(slot); }} {...appTip(studioName(view), 'right')}>
                    {studioName(view)}
                  </Button>
                )}
                <section id={ids[slot]} role="tabpanel" aria-labelledby={tabId(view)} data-studio-focus-region={slot} className="studio-window-slot__view" tabIndex={0}
                  data-focused={focusedSlot === slot ? '' : undefined} hidden={isFolded}
                  onFocusCapture={() => { if (pointerTransition.current?.slot !== slot) focusSlot(slot); }}
                  onPointerDownCapture={() => {
                    const own = sides(region);
                    const relayout = slot !== focusedSlot &&
                      (own.drawer !== focusedSides.drawer || own.inspector !== focusedSides.inspector);
                    // Keep the clicked control under the pointer until its click is dispatched.
                    pointerTransition.current = relayout ? { slot } : undefined;
                    if (!relayout) focusSlot(slot);
                  }}
                  onClickCapture={() => {
                    pointerTransition.current = undefined;
                    focusSlot(slot);
                  }}>
                  <SlotHeader view={view} icon={viewGlyphs[view]} picker={region.picker}
                    trailing={split ? <WindowButton command={closeCommand(slot)} icon={X} />
                      : <WindowButton command={splitCommand} icon={Columns2} />}>
                    {region.header}
                  </SlotHeader>
                  <div key={view} className="studio-window-slot__body">
                    {regionBody(view, region.body)}
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
              {regionBody('windowInspector', focusedRegion?.inspector, focusedOwner)}
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
            <WindowButton command={drawer} pressed={detailsDrawerVisible} controls={ids.detailsDrawer} />
          </div>
          <div className="studio-window-bottom__clock">{clock}</div>
          <div className="studio-window-bottom__right">
            <StudioConnectionStatus />
            {failure}
            {problems}
            {activity}
            <WindowButton command={inspector} pressed={inspectorVisible} controls={ids.inspector} />
          </div>
        </StudioRegionBoundary>
      </footer>
    </div>
  );
}

const loadingRegion = <StudioRegion state={{ kind: 'loading' }} />;

/** The window before its project has loaded: its regions loading, its bottom bar reporting the connection. */
export function PendingProjectWindow() {
  const [state, setState] = useState<ProjectWindowState>(defaultProjectWindowState);
  return (
    <ProjectWindow state={state} onStateChange={setState} projectName="" availableViews={[]}
      renderView={() => ({ body: loadingRegion })} map={loadingRegion} mapDrawer={loadingRegion}
      compactMap={loadingRegion} commands={[]} />
  );
}
