export type Point = { x: number; y: number };
export type Box = { x: number; y: number; width: number; height: number };

const overlaps = (a: Box, b: Box, gap: number) =>
  a.x < b.x + b.width + gap && b.x < a.x + a.width + gap && a.y < b.y + b.height + gap && b.y < a.y + a.height + gap;

/** Where `box` goes among `placed` boxes: where it is, unless that is within `gap` of one, else the nearest clear height in its column. */
export function clearSpot(box: Box, placed: readonly Box[], gap: number): Point {
  const clear = (y: number) => !placed.some((other) => overlaps({ ...box, y }, other, gap));
  const heights = [box.y, ...placed.flatMap((other) => [other.y + other.height + gap, other.y - gap - box.height])];
  const y = heights.filter(clear).reduce((best, next) => Math.abs(next - box.y) < Math.abs(best - box.y) ? next : best, Infinity);
  return { x: box.x, y };
}

/** Each box's position in order, each clear of those before it. */
export function separate(boxes: readonly Box[], gap: number): Point[] {
  const placed: Box[] = [];
  return boxes.map((box) => {
    const at = clearSpot(box, placed, gap);
    placed.push({ ...box, ...at });
    return at;
  });
}

/** An SVG path through `points` with its corners rounded. */
export function roundedPath(points: readonly Point[], radius = 6) {
  let path = `M ${points[0].x},${points[0].y}`;
  for (let index = 1; index < points.length - 1; index += 1) {
    const a = points[index - 1], b = points[index], c = points[index + 1];
    const before = Math.min(radius, Math.hypot(b.x - a.x, b.y - a.y) / 2);
    const after = Math.min(radius, Math.hypot(c.x - b.x, c.y - b.y) / 2);
    path += ` L ${b.x + Math.sign(a.x - b.x) * before},${b.y + Math.sign(a.y - b.y) * before}`;
    path += ` Q ${b.x},${b.y} ${b.x + Math.sign(c.x - b.x) * after},${b.y + Math.sign(c.y - b.y) * after}`;
  }
  const last = points.at(-1)!;
  return `${path} L ${last.x},${last.y}`;
}

type Route = { clearance: number; bend: number };

/**
 * The orthogonal wire from `from`, leaving rightward, to `to`, entering rightward, that keeps `clearance` from every
 * card and costs least, counting each bend as `bend` of length; undefined when the cards leave no such wire.
 */
export function route(cards: readonly Box[], from: Point, to: Point, { clearance, bend }: Route): Point[] | undefined {
  const start = { x: from.x + clearance, y: from.y }, end = { x: to.x - clearance, y: to.y };
  const walls = cards.map((card) => ({
    left: card.x - clearance, right: card.x + card.width + clearance, top: card.y - clearance, bottom: card.y + card.height + clearance,
  }));
  const inside = ({ x, y }: Point) => walls.some((wall) => x > wall.left && x < wall.right && y > wall.top && y < wall.bottom);
  if (inside(start) || inside(end)) return undefined;
  const lines = (values: number[]) => [...new Set(values)].sort((a, b) => a - b);
  const xs = lines([start.x, end.x, ...walls.flatMap((wall) => [wall.left, wall.right])]);
  const ys = lines([start.y, end.y, ...walls.flatMap((wall) => [wall.top, wall.bottom])]);
  const rows = ys.length;
  /** Whether the step from grid point (i, j) to the next along x (axis 0) or y (axis 1) runs through no card. */
  const open = (i: number, j: number, axis: 0 | 1) => {
    const [x0, x1, y0, y1] = axis === 0 ? [xs[i], xs[i + 1], ys[j], ys[j]] : [xs[i], xs[i], ys[j], ys[j + 1]];
    return !walls.some((wall) => (axis === 0
      ? wall.top < y0 && y0 < wall.bottom && wall.left < x1 && x0 < wall.right
      : wall.left < x0 && x0 < wall.right && wall.top < y1 && y0 < wall.bottom));
  };
  /** A grid point with the axis a wire there is heading along: it turns in place, at the cost of a bend. */
  const state = (i: number, j: number, axis: number) => (i * rows + j) * 2 + axis;
  const cost = new Float64Array(xs.length * rows * 2).fill(Infinity);
  const prior = new Int32Array(cost.length).fill(-1);
  const origin = state(xs.indexOf(start.x), ys.indexOf(start.y), 0);
  const goal = state(xs.indexOf(end.x), ys.indexOf(end.y), 0);
  cost[origin] = 0;
  const queue = new MinQueue();
  queue.push(origin, 0);
  for (let current = queue.pop(); current !== undefined; current = queue.pop()) {
    if (current === goal) break;
    const axis = current % 2, cell = current >> 1, i = Math.floor(cell / rows), j = cell % rows;
    const steps: [number, number, number, number][] = axis === 0
      ? [[i + 1, j, 0, i + 1 < xs.length && open(i, j, 0) ? xs[i + 1] - xs[i] : -1], [i - 1, j, 0, i > 0 && open(i - 1, j, 0) ? xs[i] - xs[i - 1] : -1]]
      : [[i, j + 1, 1, j + 1 < rows && open(i, j, 1) ? ys[j + 1] - ys[j] : -1], [i, j - 1, 1, j > 0 && open(i, j - 1, 1) ? ys[j] - ys[j - 1] : -1]];
    steps.push([i, j, 1 - axis, bend]);
    for (const [ni, nj, heading, length] of steps) {
      if (length < 0) continue;
      const next = state(ni, nj, heading);
      const total = cost[current] + length;
      if (total < cost[next]) {
        cost[next] = total;
        prior[next] = current;
        queue.push(next, total);
      }
    }
  }
  if (cost[goal] === Infinity) return undefined;
  const corners: Point[] = [to, end];
  for (let at = goal; prior[at] !== -1; at = prior[at]) {
    if (prior[at] >> 1 === at >> 1) corners.push({ x: xs[Math.floor((at >> 1) / rows)], y: ys[(at >> 1) % rows] });
  }
  corners.push(start);
  corners.push(from);
  const distinct = corners.reverse().filter((point, index, all) => index === 0 || point.x !== all[index - 1].x || point.y !== all[index - 1].y);
  return distinct.filter((point, index, all) => index === 0 || index === all.length - 1
    || (point.x !== all[index - 1].x || point.x !== all[index + 1].x) && (point.y !== all[index - 1].y || point.y !== all[index + 1].y));
}

class MinQueue {
  private items: number[] = [];
  private keys: number[] = [];
  push(item: number, key: number) {
    let index = this.items.length;
    this.items.push(item);
    this.keys.push(key);
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (this.keys[parent] <= key) break;
      this.swap(index, parent);
      index = parent;
    }
  }
  pop() {
    if (!this.items.length) return undefined;
    const top = this.items[0];
    const lastItem = this.items.pop()!, lastKey = this.keys.pop()!;
    if (this.items.length) {
      this.items[0] = lastItem;
      this.keys[0] = lastKey;
      for (let index = 0; ;) {
        const left = index * 2 + 1, right = left + 1;
        let least = index;
        if (left < this.keys.length && this.keys[left] < this.keys[least]) least = left;
        if (right < this.keys.length && this.keys[right] < this.keys[least]) least = right;
        if (least === index) break;
        this.swap(index, least);
        index = least;
      }
    }
    return top;
  }
  private swap(a: number, b: number) {
    [this.items[a], this.items[b]] = [this.items[b], this.items[a]];
    [this.keys[a], this.keys[b]] = [this.keys[b], this.keys[a]];
  }
}
