import {
  BarChart3, Columns2, Crosshair, FileText, GitBranch, Image, List,
  Map as SceneIcon, PanelBottom, PanelTop, Pin, Rows2, Settings, Settings2, Target, Workflow, X,
  type LucideIcon,
} from 'lucide-react';
import { StudioLogo } from '../../components/StudioLogo';
import {
  createContext, useContext, useEffect, useId, useLayoutEffect, useRef, useState,
  type CSSProperties, type Dispatch, type ReactNode, type SetStateAction,
} from 'react';
import { appTip, installTooltips } from '@window-shell/browser/tooltip';
import { shellHost } from '@window-shell/browser/titlebar';
import { CanvasStrip } from '@/components/canvas/CanvasStrip';
import type { CanvasViewportCommand } from '@/components/canvas/StudioCanvas';
import { ResizeSeparator } from '@/components/layout/ResizeSeparator';
import { Button, IconButton } from '@/components/ui/Button';
import { CommandBar, withGlyph } from '@/components/ui/CommandBar';
import { ScrollFade } from '@/components/ui/ScrollFade';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { StudioRegion, StudioRegionBoundary } from '@/components/activity/StudioRegion';
import { studioName, studioReasonedName } from '@/components/activity/studioVocabulary';
import { FieldGroups } from '@/components/values/fieldGroups';
import { useElementSize } from '@/hooks/useElementSize';
import {
  compactMapHeight, fitProjectWindow, projectRailWidth, type PreferredWindowSize, type ProjectWindowSizePreview,
} from './projectWindowLayout';
import {
  closeProjectWindowSlot, defaultProjectWindowState, noViews, projectWindowSlotView, projectWindowViews, sectionOpen,
  showProjectWindowView, withSectionOpen, type ProjectWindowMode, type ViewOffer, type ProjectWindowSlot, type ProjectWindowState, type ProjectWindowView,
} from './projectWindowState';
import type { DocumentPickerModel } from './DocumentPicker';
import { SlotHeader } from './SlotHeader';
import { StudioCommandPalette } from './StudioCommandPalette';
import { StudioConnectionStatus } from './StudioConnectionStatus';
import { StudioSettings } from './StudioSettings';
import { useWindowCommands } from './heldGestures';
import { bindStudioCommand, commandAttributes, commandInContext, commandTip, keyHandled, matchesShortcut, studioCommandContext, type StudioCommand } from './studioCommands';
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
  /** The body is a form or report, drawn at the slot's inset rather than to its edges. */
  padded?: boolean;
}

/** A slot's pin: its command and, while pinned, its step's name and, when that is not the selected step, its views. */
export interface SlotPinning {
  readonly command: StudioCommand;
  readonly pinned?: { readonly label: string; readonly views?: ViewOffer };
}

export interface ProjectWindowProps {
  state: ProjectWindowState;
  onStateChange: Dispatch<SetStateAction<ProjectWindowState>>;
  projectName: string;
  saveStatus?: ReactNode;
  views: ViewOffer;
  /** Each slot's pin, in split view. */
  pins?: Readonly<Partial<Record<ProjectWindowSlot, SlotPinning>>>;
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
  /** The rows that open above the clock, which take their height from the details. */
  shelf?: ReactNode;
  failure?: ReactNode;
  problems?: ReactNode;
  activity?: ReactNode;
  /** The open project's rows in Settings. */
  projectSettings?: ReactNode;
}

/** The glyph of each view, and of the document kind a view shows. */
export const viewGlyphs = {
  runs: List,
  model: GitBranch,
  task: Crosshair,
  objective: Target,
  scene: SceneIcon,
  results: BarChart3,
  analysis: Workflow,
  figures: Image,
  document: FileText,
  training: Settings2,
} satisfies Record<ProjectWindowView, LucideIcon>;

const modes = [
  { mode: 'map', id: 'showMap', icon: PanelTop },
  { mode: 'both', id: 'showBoth', icon: Rows2 },
  { mode: 'details', id: 'showDetails', icon: PanelBottom },
] as const;

const slots = ['primary', 'secondary'] as const;
const regionOrder = ['top', 'map', 'steps', 'rail', 'drawer', ...slots, 'inspector', 'settings', 'bottom'] as const;
type WindowRegion = typeof regionOrder[number];

function WindowButton({ command, icon: Icon = command.glyph!, pressed, controls, tab, placement = tab ? 'right' : 'bottom', reason }: {
  command: StudioCommand;
  icon?: LucideIcon;
  pressed?: boolean;
  controls?: string;
  tab?: { id: string; tabbable: boolean };
  placement?: 'bottom' | 'right';
  /** Why the command does not apply, which its tip gives after its name. */
  reason?: string;
}) {
  return (
    <IconButton label={command.label} icon={<Icon aria-hidden="true" />} {...commandAttributes(command, placement)}
      {...reason !== undefined && { ...appTip(studioReasonedName(command.name, reason), placement), 'aria-description': reason }}
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

type PaletteScope = 'project' | 'activity';

const PaletteState = createContext<[PaletteScope | null, Dispatch<SetStateAction<PaletteScope | null>>] | null>(null);

/**
 * Whether the command palette is open, and for which scope. It is held above the windows of a page, so a palette opened
 * in the window drawn while the project loads stays open in the window that replaces it.
 */
export function ProjectWindowPalette({ children }: { children: ReactNode }) {
  return <PaletteState.Provider value={useState<PaletteScope | null>(null)}>{children}</PaletteState.Provider>;
}

function usePaletteState() {
  const state = useContext(PaletteState);
  if (!state) throw new Error('A ProjectWindow is drawn outside ProjectWindowPalette.');
  return state;
}

export function ProjectWindow({
  state, onStateChange, projectName, saveStatus, views, pins, renderView, map, mapDrawer, compactMap,
  onBack, onForward, commands, mapCommands, focusRequest, variant, overlay, clock, shelf, failure, problems, activity,
  projectSettings,
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
  const { ref: shelfBox, height: shelfHeight } = useElementSize<HTMLDivElement>();
  const [preview, setPreview] = useState<ProjectWindowSizePreview>({});
  const [requestedFocus, setRequestedFocus] = useState<{ region: WindowRegion }>();
  const [paletteScope, setPaletteScope] = usePaletteState();
  const settingsRegion = useRef<HTMLElement>(null);
  const mapSection = useRef<HTMLElement>(null);
  const divider = useRef<HTMLDivElement>(null);
  const detailsSection = useRef<HTMLElement>(null);
  const [railFocus, setRailFocus] = useState<ProjectWindowView | null>(null);
  const [mapHasFocus, setMapHasFocus] = useState(false);
  const pointerTransition = useRef<{ slot: ProjectWindowSlot }>();
  const handledFocusRequest = useRef(focusRequest);
  const moveFocus = (region: WindowRegion) => setRequestedFocus({ region });
  useLayoutEffect(() => installTooltips(), []);

  const offers = {
    primary: pins?.primary?.pinned?.views ?? views,
    secondary: pins?.secondary?.pinned?.views ?? views,
  };
  const primaryView = projectWindowSlotView(state, 'primary', offers.primary.available);
  const secondaryView = projectWindowSlotView(state, 'secondary', offers.secondary.available);
  const focusedSlot = secondaryView && state.focusedSlot === 'secondary' ? 'secondary' : 'primary';
  /** The views the rail offers: those of the focused slot's step. */
  const focusedOffer = offers[focusedSlot];
  const availableViews = focusedOffer.available;
  const shownViews = { primary: primaryView, secondary: secondaryView };
  const regions = {
    primary: primaryView ? renderView(primaryView, 'primary') : null,
    secondary: secondaryView ? renderView(secondaryView, 'secondary') : null,
  };
  const focusedRegion = regions[focusedSlot];
  const sides = (region: ProjectWindowViewRegion | null) => ({ drawer: !!region?.drawer, inspector: !!region?.inspector });
  const focusedSides = sides(focusedRegion);
  const mode = availableViews.length ? state.mode : 'map';
  const fit = fitProjectWindow(width, height - shelfHeight, {
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

  // Settings covers what the main area showed when it opened, and closes once a gesture shows something else there.
  const shown = JSON.stringify([mode, primaryView, secondaryView, focusedSlot,
    state.mapDrawerOpen, state.detailsDrawerOpen, state.inspectorOpen]);
  const [settingsOver, setSettingsOver] = useState<string | null>(null);
  if (settingsOver !== null && settingsOver !== shown) setSettingsOver(null);
  const settingsOpen = settingsOver !== null;

  useLayoutEffect(() => {
    [mapSection, divider, detailsSection].forEach(({ current }) => { current!.inert = settingsOpen; });
  }, [settingsOpen]);

  function toggleSettings() {
    if (!settingsOpen) {
      setSettingsOver(shown);
      moveFocus('settings');
      return;
    }
    closeSettings();
  }

  function closeSettings() {
    if (settingsRegion.current?.contains(document.activeElement)) moveFocus(detailsVisible ? focusedSlot : 'map');
    setSettingsOver(null);
  }

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
    setSettingsOver(null);
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
    const view = availableViews.includes('scene') ? 'scene'
      : availableViews.find((candidate) => candidate !== primaryView) ?? availableViews[0];
    onStateChange((previous) => showProjectWindowView(previous, view, 'secondary'));
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
  const back = bindStudioCommand('back', () => { setSettingsOver(null); onBack?.(); }, { enabled: !!onBack });
  const forward = bindStudioCommand('forward', () => { setSettingsOver(null); onForward?.(); }, { enabled: !!onForward });
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
    { name: 'closeSlot', keyboard: false, instance: slot });
  const switchCommand = bindStudioCommand('switchSlot', switchSlot, { enabled: split });
  const tabCommands = projectWindowViews.map((view) => bindStudioCommand(view, () => activateView(view),
    { enabled: availableViews.includes(view) }));
  const tabView = railFocus ?? shownViews[focusedSlot] ?? availableViews[0];
  const tabId = (view: ProjectWindowView) => `${id}-tab-${view}`;
  const palette = bindStudioCommand('commandPalette', () => {
    const scope = document.activeElement?.closest('[data-studio-activity-panel]') ? 'activity' : 'project';
    setPaletteScope((open) => open ? null : scope);
  });
  const settings = bindStudioCommand('settings', toggleSettings);
  /** In a browser the start page is the project list; the desktop app opens projects from its own start window. */
  const projects = projectName && shellHost() !== 'native'
    ? bindStudioCommand('projects', () => window.location.assign(window.location.pathname)) : undefined;
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
    settings, ...projects ? [projects] : [], ...appearanceCommands(),
  ];
  const shortcutCommands = useRef({ palette, paletteCommands, paletteScope });
  shortcutCommands.current = { palette, paletteCommands, paletteScope };
  const windowCommands = [palette, ...paletteCommands];
  useWindowCommands(windowCommands);
  useStudioNativeMenu(windowCommands, paletteScope !== null);
  const bar = useRef<HTMLElement>(null);
  useStudioTitleBar(bar, projectName);
  useEffect(() => {
    function handleKey(event: KeyboardEvent) {
      if (keyHandled(event) || event.isComposing || event.repeat) return;
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
  const panelRows = `${mode === 'map'
    ? 'minmax(0, 1fr) 0px 0px'
    : mode === 'details'
      ? `${compactMapHeight}px 0px minmax(0, 1fr)`
      : `${fit.mapHeight.size}px 1px minmax(0, 1fr)`} auto`;
  const resizeProps = (key: PreferredWindowSize) => ({
    onPreview: (size: number | null) => previewSize(key, size),
    onResize: (size: number) => resize(key, size),
  });
  const focusedOwner = `${focusedSlot}:${shownViews[focusedSlot]}`;
  const regionBody = (name: StudioCommand['name'], body: ReactNode, owner?: string, view?: ProjectWindowView | null) => (
    <div className="studio-window-region-content">
      <StudioRegionBoundary key={owner} name={name}>
        {view ? <FieldGroups.Provider value={{
          isOpen: (field) => sectionOpen(state, `${view}.field${field}`),
          setOpen: (field, open) => onStateChange((current) => withSectionOpen(current, `${view}.field${field}`, open)),
        }}>{body}</FieldGroups.Provider> : body}
      </StudioRegionBoundary>
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
          <CommandBar entries={[steps, back, forward].map((command) => ({ command: withGlyph(command), priority: 'normal' }))} moreLabel={studioName('more')} itemClassName="studio-window-control"
            itemProps={(command) => ({
              ...commandAttributes(command),
              ...(command === steps ? { 'aria-pressed': mapDrawerVisible, 'aria-controls': ids.mapDrawer } : {}),
            })} />
        </div>
        <div className="shell-frame__cell studio-window-bar__project" data-separated="">
          <h1 className="min-w-0 truncate text-base font-normal">
            {projects
              ? <Button variant="ghost" className="max-w-full px-1 text-base" {...commandTip(projects)} onClick={projects.invoke}>
                <span className="truncate">{projectName}</span>
              </Button>
              : projectName}
          </h1>
          {saveStatus}
          {variant}
        </div>
        <div className="studio-window-bar__commands">
          <WindowButton command={palette} />
          {paletteScope && <StudioCommandPalette
            commands={paletteScope === 'activity' ? paletteCommands.filter((command) => !command.selectionScoped) : paletteCommands}
            close={() => setPaletteScope(null)} />}
          {overlay}
        </div>
      </header>
      <main ref={panel} className="shell-frame__content studio-window-panel" style={{ gridTemplateRows: panelRows }}>
        <section ref={mapSection} id={ids.map} className="studio-window-map" aria-label={studioName('pipelineMap')}>
          <div className="studio-window-map__full" hidden={!fullMapVisible}>
            <div data-studio-focus-region="map" tabIndex={-1} className="min-h-0 min-w-0 flex-1 overflow-hidden">
              {mapCommands ? <CanvasStrip viewport={mapCommands} label={studioName('pipelineMap')}>{regionBody('pipelineMap', map)}</CanvasStrip>
                : regionBody('pipelineMap', map)}
            </div>
            <aside id={ids.mapDrawer} data-studio-focus-region="steps" tabIndex={-1} aria-label={studioName('mapDrawer')} className="studio-window-side"
              hidden={!mapDrawerVisible} style={{ width: fit.mapDrawer.size }}>
              <ScrollFade className="studio-window-side__contents">
                {regionBody('mapDrawer', mapDrawer)}
              </ScrollFade>
              {mapDrawerVisible && (
                <ResizeSeparator label={studioName('mapDrawer')} controls={ids.mapDrawer} axis="x" direction={1}
                  layout={fit.mapDrawer} className="studio-window-resize studio-window-resize--right" {...resizeProps('mapDrawerWidth')} />
              )}
            </aside>
          </div>
          {!fullMapVisible && <ScrollFade axis="x" className="studio-window-map__compact" data-studio-focus-region="map" tabIndex={-1}>
            {regionBody('pipelineMap', compactMap)}
          </ScrollFade>}
        </section>
        <div ref={divider} className="studio-window-panel__divider">
          {mode === 'both' && fit.mapHeight.open && (
            <ResizeSeparator label={studioName('panelHeight')} controls={ids.map} axis="y" direction={1}
              layout={fit.mapHeight} className="studio-window-resize studio-window-resize--row" {...resizeProps('mapHeight')} />
          )}
        </div>
        <nav className="shell-rail studio-window-rail" data-separated="" aria-label={studioName('details')}
          data-studio-focus-region="rail" tabIndex={-1} style={{ '--shell-rail-separator': separator } as CSSProperties}>
          {detailsVisible && <div role="tablist" aria-label={studioName('details')} aria-orientation="vertical"
            className="shell-scroll-rail studio-window-tabs" data-axis="y"
            onFocusCapture={(event) => {
              const id = event.target.dataset.commandId;
              const view = projectWindowViews.find((item) => item === id);
              if (view && event.target.getAttribute('aria-disabled') !== 'true') setRailFocus(view);
            }}
            onBlurCapture={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget)) setRailFocus(null);
            }}
            onKeyDown={(event) => {
              const tabs = [...event.currentTarget.querySelectorAll<HTMLElement>('[role="tab"]:not([aria-disabled="true"])')];
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
              if (!command.enabled) {
                return <WindowButton key={view} command={command} icon={viewGlyphs[view]} tab={{ id: tabId(view), tabbable: false }}
                  reason={focusedOffer.absent[view]} />;
              }
              const selected = !settingsOpen && shownViews[focusedSlot] === view;
              return (
                <WindowButton key={view} command={command} icon={viewGlyphs[view]} controls={selected ? ids[focusedSlot] : undefined}
                  pressed={selected} tab={{ id: tabId(view), tabbable: tabView === view }} />
              );
            })}
          </div>}
        </nav>
        <section ref={detailsSection} id={ids.details} className="studio-window-details" hidden={!detailsVisible} aria-label={studioName('details')}>
          <aside id={ids.detailsDrawer} data-studio-focus-region="drawer" tabIndex={-1} aria-label={studioName('detailsDrawer')} className="studio-window-side"
            hidden={!detailsDrawerVisible} style={{ width: fit.detailsDrawer.size }}>
            <ScrollFade className="studio-window-side__contents" data-populated={focusedRegion?.drawer ? '' : undefined}>
              {regionBody('detailsDrawer', focusedRegion?.drawer, focusedOwner, shownViews[focusedSlot])}
            </ScrollFade>
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
                  <SlotHeader view={view} icon={split ? viewGlyphs[view] : undefined} picker={region.picker}
                    step={split ? pins?.[slot]?.pinned?.label : undefined}
                    trailing={split ? <>
                      {pins?.[slot] && <WindowButton command={pins[slot].command} icon={Pin}
                        pressed={!!pins[slot].pinned} />}
                      <WindowButton command={closeCommand(slot)} icon={X} />
                    </> : <WindowButton command={splitCommand} icon={Columns2} />}>
                    {region.header}
                  </SlotHeader>
                  <ScrollFade key={view} axis="both" className="studio-window-slot__body" data-padded={region.padded ? '' : undefined}>
                    {regionBody(view, region.body, undefined, view)}
                  </ScrollFade>
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
            <ScrollFade className="studio-window-side__contents" data-populated={focusedRegion?.inspector ? '' : undefined}>
              {regionBody('windowInspector', focusedRegion?.inspector, focusedOwner, shownViews[focusedSlot])}
            </ScrollFade>
            {inspectorVisible && (
              <ResizeSeparator label={studioName('windowInspector')} controls={ids.inspector} axis="x" direction={-1}
                layout={fit.inspector} className="studio-window-resize studio-window-resize--left" {...resizeProps('inspectorWidth')} />
            )}
          </aside>
        </section>
        <div ref={shelfBox} className="studio-window-shelf" hidden={!shelf}>{shelf}</div>
        {settingsOpen && (
          <section ref={settingsRegion} className="studio-window-settings" aria-label={studioName('settings')}
            data-studio-focus-region="settings" tabIndex={-1}
            onKeyDown={(event) => {
              if (event.key !== 'Escape' || event.defaultPrevented) return;
              event.preventDefault();
              closeSettings();
            }}>
            <h2 className="sr-only">{studioName('settings')}</h2>
            <StudioSettings project={projectSettings} />
          </section>
        )}
      </main>
      <footer className="shell-frame__footer studio-window-bottom" data-studio-focus-region="bottom" tabIndex={-1}>
        <StudioRegionBoundary name="bottomBar">
          <div className="studio-window-bottom__corner">
            <WindowButton command={settings} icon={Settings} pressed={settingsOpen} placement="right" />
          </div>
          <div className="studio-window-bottom__left">
            <WindowButton command={drawer} pressed={detailsDrawerVisible} controls={ids.detailsDrawer} />
          </div>
          <div className="studio-window-bottom__clock">{clock}</div>
          <div className="studio-window-bottom__notice">
            <StudioConnectionStatus />
            {failure}
          </div>
          <div className="studio-window-bottom__right">
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

/**
 * The window with no project: its regions loading while one is `pending`, else empty, and its bottom bar reporting
 * the connection.
 */
export function ProjectlessWindow({ pending }: { pending: boolean }) {
  const [state, setState] = useState<ProjectWindowState>(defaultProjectWindowState);
  const region = pending ? loadingRegion : null;
  return (
    <ProjectWindow state={state} onStateChange={setState} projectName="" views={noViews}
      renderView={() => ({ body: region })} map={region} mapDrawer={region}
      compactMap={region} commands={[]} />
  );
}
