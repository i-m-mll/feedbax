type Point = [number, number];

export const plotWidth = 240;
export const plotHeight = 126;
const plotInset = 8;

const conciseLoss = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 4 });
const scientificLoss = new Intl.NumberFormat(undefined, { notation: 'scientific', maximumSignificantDigits: 4 });

export function formatTrainingLoss(value: number) {
  const magnitude = Math.abs(value);
  return (magnitude > 0 && (magnitude < 0.001 || magnitude >= 10_000) ? scientificLoss : conciseLoss).format(value);
}

export function projectSeries(series: readonly Point[][], preserveAspect = false) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const track of series) {
    for (const [x, y] of track) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  if (!Number.isFinite(minX)) return series;
  const width = plotWidth - 2 * plotInset;
  const height = plotHeight - 2 * plotInset;
  const spanX = maxX - minX;
  const spanY = maxY - minY;
  let scaleX = spanX > 0 ? width / spanX : Infinity;
  let scaleY = spanY > 0 ? height / spanY : Infinity;
  if (preserveAspect) scaleX = scaleY = Math.min(scaleX, scaleY);
  if (!Number.isFinite(scaleX)) scaleX = 1;
  if (!Number.isFinite(scaleY)) scaleY = 1;
  return series.map((track) => track.map(([x, y]): Point => [
    plotWidth / 2 + (x - minX - spanX / 2) * scaleX,
    plotHeight / 2 - (y - minY - spanY / 2) * scaleY,
  ]));
}

export function PlotTrack({ points, dashed }: { points: Point[]; dashed?: boolean }) {
  const last = points[points.length - 1];
  if (!last) return null;
  return (
    <g className="text-shell-foreground">
      <polyline
        points={points.map(([x, y]) => `${x},${y}`).join(' ')}
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        strokeDasharray={dashed ? '4 3' : undefined}
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
      <circle cx={last[0]} cy={last[1]} r={3} fill="currentColor" />
    </g>
  );
}
