import { useTrainingStore } from '@/stores/trainingStore';

const WORKSPACE = { xMin: -0.05, xMax: 0.55, yMin: -0.05, yMax: 0.55 };
const SVG_SIZE = 180;

function toSvg(x: number, y: number): [number, number] {
  const sx = ((x - WORKSPACE.xMin) / (WORKSPACE.xMax - WORKSPACE.xMin)) * SVG_SIZE;
  // flip y: SVG y=0 is top, physical y=0 is bottom
  const sy = SVG_SIZE - ((y - WORKSPACE.yMin) / (WORKSPACE.yMax - WORKSPACE.yMin)) * SVG_SIZE;
  return [sx, sy];
}

export function TrajectoryViewer() {
  const latestTrajectory = useTrainingStore((s) => s.latestTrajectory);

  if (!latestTrajectory) {
    return (
      <div className="flex items-center justify-center h-[190px] text-xs text-slate-400">
        Trajectory will appear after first snapshot
      </div>
    );
  }

  const { effector, target, batch } = latestTrajectory;

  // Build SVG path string
  const points = effector.map(([x, y]) => toSvg(x, y));
  const pathD = points.map(([sx, sy], i) => `${i === 0 ? 'M' : 'L'} ${sx.toFixed(1)} ${sy.toFixed(1)}`).join(' ');

  const [tx, ty] = toSvg(target[0], target[1]);

  return (
    <div className="flex flex-col gap-1">
      <div className="text-[10px] text-slate-400 flex justify-between px-1">
        <span>Trajectory</span>
        <span>batch {batch}</span>
      </div>
      <svg
        width={SVG_SIZE}
        height={SVG_SIZE}
        className="rounded bg-slate-50 border border-slate-100 block mx-auto"
        viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`}
      >
        {/* Workspace border */}
        <rect x={0} y={0} width={SVG_SIZE} height={SVG_SIZE} fill="none" stroke="#e2e8f0" strokeWidth={1} />
        {/* Origin cross */}
        {(() => { const [ox, oy] = toSvg(0, 0); return <circle cx={ox} cy={oy} r={2} fill="#cbd5e1" />; })()}
        {/* Trajectory path */}
        <path d={pathD} fill="none" stroke="#6366f1" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" opacity={0.8} />
        {/* Start point */}
        <circle cx={points[0][0]} cy={points[0][1]} r={3} fill="#94a3b8" />
        {/* End point */}
        <circle cx={points[points.length-1][0]} cy={points[points.length-1][1]} r={3} fill="#6366f1" />
        {/* Target */}
        <circle cx={tx} cy={ty} r={5} fill="none" stroke="#f59e0b" strokeWidth={2} />
        <circle cx={tx} cy={ty} r={2} fill="#f59e0b" />
      </svg>
    </div>
  );
}
