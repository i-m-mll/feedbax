import { fitPanes } from '@window-shell/browser/layout';
import type { ProjectWindowState } from './projectWindowState';

export const projectRailWidth = 42;
export const compactMapHeight = 44;

const minSlotWidth = 260;
const minDetailsDrawerWidth = 150;
const minInspectorWidth = 172;
const minMapHeight = 150;
const minDetailsHeight = 180;
const foldWidth = 36;

export type PreferredWindowSize =
  | 'mapHeight'
  | 'mapDrawerWidth'
  | 'detailsDrawerWidth'
  | 'inspectorWidth'
  | 'secondaryWidth';

export type ProjectWindowSizePreview = Partial<Record<PreferredWindowSize, number>>;

export function fitProjectWindow(
  width: number,
  height: number,
  state: ProjectWindowState,
  preview: ProjectWindowSizePreview = {},
) {
  const size = (key: PreferredWindowSize) => preview[key] ?? state[key];
  const map = fitPanes(width, minSlotWidth, [{
    open: state.mapDrawerOpen && state.mode !== 'details',
    preferredSize: size('mapDrawerWidth'),
    minSize: 218,
  }]);
  const mapHeight = fitPanes(Math.max(0, height - 1), minDetailsHeight, [{
    open: true,
    preferredSize: size('mapHeight'),
    minSize: minMapHeight,
    collapsedSize: compactMapHeight,
  }]).panes[0]!;

  const detailsDrawerWidth = size('detailsDrawerWidth');
  const splitMinimum = 2 * minSlotWidth
    + (state.detailsDrawerOpen ? Math.max(minDetailsDrawerWidth, detailsDrawerWidth) : 0)
    + (state.inspectorOpen ? minInspectorWidth : 0);
  const details = fitPanes(width, minSlotWidth, [
    { open: state.detailsDrawerOpen, preferredSize: detailsDrawerWidth, minSize: minDetailsDrawerWidth },
    { open: state.inspectorOpen, preferredSize: size('inspectorWidth'), minSize: minInspectorWidth },
    { open: state.secondaryView !== null && width >= splitMinimum,
      preferredSize: size('secondaryWidth'), minSize: minSlotWidth,
      collapsedSize: state.secondaryView === null ? 0 : foldWidth },
  ]);
  const [detailsDrawer, inspector, secondary] = details.panes;
  const [mapDrawer] = map.panes;
  return {
    mapDrawer: mapDrawer!,
    mapHeight,
    detailsDrawer: detailsDrawer!,
    inspector: inspector!,
    secondary: secondary!,
    primarySize: details.mainSize,
  };
}
