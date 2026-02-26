import { useCallback, useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import { ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import { useTrajectoryStore } from '@/stores/trajectoryStore';
import { TASK_TYPE_LABELS } from '@/types/trajectory';
import { Scene } from '@/components/viewer/Scene';
import { PlaybackControls } from '@/components/viewer/PlaybackControls';
import { useTrainingStore } from '@/stores/trainingStore';
import type { TrajectorySnapshot } from '@/stores/trainingStore';

export function TrajectoryPanel() {
  const {
    datasets,
    activeDataset,
    metadata,
    filteredIndices,
    activeIndex,
    trajectoryData,
    loading,
    error,
    playback,
    filterBodyIdx,
    filterTaskType,
    loadDatasets,
    selectDataset,
    applyFilter,
    selectTrajectory,
  } = useTrajectoryStore();

  const latestTrajectory = useTrainingStore((state) => state.latestTrajectory);
  const [liveMode, setLiveMode] = useState(false);

  // Load datasets on mount
  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  // Unique body indices and task types from metadata
  const bodyIndices = useMemo(() => metadata?.body_indices ?? [], [metadata]);
  const taskTypes = useMemo(() => metadata?.task_types ?? [], [metadata]);

  // Navigate trajectory index within filtered list
  const currentFilteredPos = useMemo(() => {
    if (filteredIndices === null || activeIndex === null) return -1;
    return filteredIndices.indexOf(activeIndex);
  }, [filteredIndices, activeIndex]);

  const handlePrev = useCallback(() => {
    if (!filteredIndices || currentFilteredPos <= 0) return;
    selectTrajectory(filteredIndices[currentFilteredPos - 1]);
  }, [filteredIndices, currentFilteredPos, selectTrajectory]);

  const handleNext = useCallback(() => {
    if (!filteredIndices || currentFilteredPos >= filteredIndices.length - 1) return;
    selectTrajectory(filteredIndices[currentFilteredPos + 1]);
  }, [filteredIndices, currentFilteredPos, selectTrajectory]);

  const handleDatasetChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value;
      if (value) selectDataset(value);
    },
    [selectDataset],
  );

  const handleBodyFilter = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value;
      applyFilter(value === '' ? null : Number(value), filterTaskType);
    },
    [applyFilter, filterTaskType],
  );

  const handleTaskFilter = useCallback(
    (taskType: number) => {
      applyFilter(filterBodyIdx, filterTaskType === taskType ? null : taskType);
    },
    [applyFilter, filterBodyIdx, filterTaskType],
  );

  return (
    <div className="flex flex-col h-full">
      {/* Top bar: filters and navigation */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-100 bg-white/80 flex-shrink-0 flex-wrap">
        {/* Dataset selector */}
        <select
          value={activeDataset ?? ''}
          onChange={handleDatasetChange}
          className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700 bg-white min-w-[140px]"
        >
          <option value="">Select dataset...</option>
          {datasets.map((ds) => (
            <option key={ds.name} value={ds.name}>
              {ds.name}
            </option>
          ))}
        </select>

        {/* Body index filter */}
        {metadata && bodyIndices.length > 0 && (
          <select
            value={filterBodyIdx ?? ''}
            onChange={handleBodyFilter}
            className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700 bg-white"
          >
            <option value="">All bodies</option>
            {bodyIndices.map((idx) => (
              <option key={idx} value={idx}>
                Body {idx}
              </option>
            ))}
          </select>
        )}

        {/* Task type filter buttons */}
        {metadata && taskTypes.length > 0 && (
          <div className="flex items-center gap-1">
            <span className="text-xs text-slate-400 mr-1">Task:</span>
            {taskTypes.map((tt) => (
              <button
                key={tt}
                onClick={() => handleTaskFilter(tt)}
                className={clsx(
                  'text-[10px] font-semibold px-2 py-0.5 rounded-full border',
                  filterTaskType === tt
                    ? 'border-brand-500 text-brand-600 bg-brand-500/10'
                    : 'border-slate-200 text-slate-400 hover:text-slate-600 hover:bg-slate-50',
                )}
              >
                {TASK_TYPE_LABELS[tt] ?? `Type ${tt}`}
              </button>
            ))}
          </div>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Live mode toggle */}
        <button
          type="button"
          onClick={() => setLiveMode((v) => !v)}
          className={clsx(
            'rounded-full border px-2.5 py-0.5 text-[10px] font-semibold transition-colors',
            liveMode
              ? 'border-brand-400 bg-brand-500/10 text-brand-600'
              : 'border-slate-200 text-slate-400 hover:text-slate-600 hover:bg-slate-50',
            !latestTrajectory && 'opacity-50 cursor-not-allowed',
          )}
          disabled={!latestTrajectory}
          title={latestTrajectory ? 'Toggle live trajectory overlay' : 'No live trajectory available yet'}
        >
          Live
        </button>

        {/* Trajectory index navigator */}
        {filteredIndices && filteredIndices.length > 0 && (
          <div className="flex items-center gap-1">
            <button
              onClick={handlePrev}
              disabled={currentFilteredPos <= 0}
              className="p-0.5 rounded text-slate-500 hover:text-slate-700 hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed"
              title="Previous trajectory"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-xs text-slate-500 tabular-nums min-w-[60px] text-center">
              {currentFilteredPos >= 0 ? currentFilteredPos + 1 : '-'} / {filteredIndices.length}
            </span>
            <button
              onClick={handleNext}
              disabled={currentFilteredPos >= filteredIndices.length - 1}
              className="p-0.5 rounded text-slate-500 hover:text-slate-700 hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed"
              title="Next trajectory"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* Main canvas area */}
      <div className="flex-1 relative min-h-0">
        {/* Error banner */}
        {error && (
          <div className="absolute top-2 left-2 right-2 z-10 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-xs text-red-600">
            {error}
          </div>
        )}

        {/* Loading spinner overlay */}
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/60">
            <Loader2 className="w-6 h-6 text-brand-500 animate-spin" />
          </div>
        )}

        {/* Empty state */}
        {!activeDataset && !loading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-sm text-slate-400">Select a dataset to begin</div>
          </div>
        )}

        {/* No results state */}
        {activeDataset && !loading && filteredIndices?.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-sm text-slate-400">No trajectories match the current filters</div>
          </div>
        )}

        {/* Three.js scene */}
        <Scene
          trajectoryData={trajectoryData}
          frame={playback.frame}
        />

        {/* Live trajectory overlay */}
        {liveMode && latestTrajectory && (
          <div className="absolute inset-0 pointer-events-none">
            <LiveTrajectoryView snapshot={latestTrajectory} />
          </div>
        )}
      </div>

      {/* Playback controls */}
      {trajectoryData && <PlaybackControls />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Live trajectory viewer (2D SVG overlay, no joint angles required)
// ---------------------------------------------------------------------------

const LIVE_PADDING = 24; // px padding inside the SVG viewport
const LIVE_SIZE = 200;   // SVG viewport size in px

/**
 * Simple 2D SVG overlay that draws the latest streamed effector path and
 * target position.  Coordinates are normalised to a unit square and mapped
 * into the SVG viewport with a small padding margin.
 */
function LiveTrajectoryView({ snapshot }: { snapshot: TrajectorySnapshot }) {
  const { effector, target } = snapshot;

  // Determine data bounds so we can normalise to the viewport.
  const allX = effector.map(([x]) => x).concat([target[0]]);
  const allY = effector.map(([, y]) => y).concat([target[1]]);
  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const minY = Math.min(...allY);
  const maxY = Math.max(...allY);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const range = Math.max(rangeX, rangeY);

  const inner = LIVE_SIZE - LIVE_PADDING * 2;

  const toSvg = (x: number, y: number): [number, number] => [
    LIVE_PADDING + ((x - minX) / range) * inner,
    // Flip Y so positive Y points up
    LIVE_SIZE - LIVE_PADDING - ((y - minY) / range) * inner,
  ];

  const pathD = effector
    .map(([x, y], i) => {
      const [sx, sy] = toSvg(x, y);
      return `${i === 0 ? 'M' : 'L'}${sx.toFixed(1)},${sy.toFixed(1)}`;
    })
    .join(' ');

  const [tx, ty] = toSvg(target[0], target[1]);
  const [ox, oy] = toSvg(effector[0]?.[0] ?? 0, effector[0]?.[1] ?? 0);
  const [ex, ey] = toSvg(
    effector[effector.length - 1]?.[0] ?? 0,
    effector[effector.length - 1]?.[1] ?? 0,
  );

  return (
    <div
      className="absolute bottom-4 right-4 rounded-xl border border-brand-200 bg-white/90 shadow-soft"
      style={{ width: LIVE_SIZE, height: LIVE_SIZE }}
    >
      <div className="absolute top-1.5 left-2 text-[9px] font-semibold uppercase tracking-widest text-brand-500">
        Live · batch {snapshot.batch}
      </div>
      <svg width={LIVE_SIZE} height={LIVE_SIZE} className="block">
        {/* Effector trace */}
        <path
          d={pathD}
          fill="none"
          stroke="#6366f1"
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
          opacity={0.85}
        />
        {/* Origin dot */}
        <circle cx={ox} cy={oy} r={3} fill="#94a3b8" />
        {/* End-effector dot */}
        <circle cx={ex} cy={ey} r={4} fill="#6366f1" />
        {/* Target marker */}
        <circle
          cx={tx}
          cy={ty}
          r={6}
          fill="none"
          stroke="#10b981"
          strokeWidth={1.5}
          opacity={0.9}
        />
        <circle cx={tx} cy={ty} r={2} fill="#10b981" opacity={0.9} />
      </svg>
    </div>
  );
}
