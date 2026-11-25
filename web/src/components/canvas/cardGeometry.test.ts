import { describe, expect, it } from 'vitest';
import { route, separate, type Box, type Point } from './cardGeometry';

const options = { clearance: 10, bend: 40 };
const right = (box: Box): Point => ({ x: box.x + box.width, y: box.y + box.height / 2 });
const left = (box: Box): Point => ({ x: box.x, y: box.y + box.height / 2 });

/** Whether the polyline passes through the inside of `box`. */
function crosses(points: readonly Point[], box: Box) {
  return points.slice(1).some((b, index) => {
    const a = points[index];
    const [x0, x1, y0, y1] = [Math.min(a.x, b.x), Math.max(a.x, b.x), Math.min(a.y, b.y), Math.max(a.y, b.y)];
    return x0 < box.x + box.width && x1 > box.x && y0 < box.y + box.height && y1 > box.y;
  });
}

describe('route', () => {
  it('runs straight between cards in a row', () => {
    const a = { x: 0, y: 0, width: 100, height: 30 }, b = { x: 200, y: 0, width: 100, height: 30 };
    expect(route([a, b], right(a), left(b), options)).toEqual([right(a), left(b)]);
  });

  it('goes around a card standing between source and target instead of through it', () => {
    const a = { x: 0, y: 0, width: 100, height: 30 }, wall = { x: 150, y: -40, width: 100, height: 110 };
    const b = { x: 300, y: 0, width: 100, height: 30 };
    const points = route([a, wall, b], right(a), left(b), options)!;
    expect(crosses(points, wall)).toBe(false);
    expect(points[1].x).toBeGreaterThan(points[0].x);
    expect(points.at(-2)!.x).toBeLessThan(points.at(-1)!.x);
  });

  it('loops a wire to a card on its left around both cards, leaving right and entering left', () => {
    const a = { x: 300, y: 0, width: 100, height: 30 }, b = { x: 0, y: 0, width: 100, height: 30 };
    const points = route([a, b], right(a), left(b), options)!;
    expect(crosses(points, a) || crosses(points.slice(1, -1), b)).toBe(false);
    expect(points[1].x).toBeGreaterThan(points[0].x);
    expect(points.at(-2)!.x).toBeLessThan(points.at(-1)!.x);
  });
});

describe('separate', () => {
  it('moves a card off one placed before it and leaves clear cards where they are', () => {
    const boxes = [{ x: 0, y: 0, width: 100, height: 30 }, { x: 20, y: 10, width: 100, height: 30 }, { x: 400, y: 0, width: 100, height: 30 }];
    const [first, second, third] = separate(boxes, 20);
    expect(first).toEqual({ x: 0, y: 0 });
    expect(second).toEqual({ x: 20, y: 50 });
    expect(third).toEqual({ x: 400, y: 0 });
  });
});
