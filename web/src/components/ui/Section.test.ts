import { expect, it } from 'vitest';
import { shareHeight } from './Section';

const plain = (...contents: number[]) => contents.map((content) => ({ content }));
/** A list of `rows` rows of 29 px between 4 px of padding above and below. */
const list = (rows: number) => ({
  content: 8 + rows * 29 - 1,
  stops: [...Array.from({ length: rows - 1 }, (_, index) => 4 + (index + 1) * 29 - 1), 8 + rows * 29 - 1],
});

it('gives what no section needs to the last open section', () => {
  expect(shareHeight(plain(100, 50), 400)).toEqual({ heights: [100, 300], rest: 0 });
});

it('gives a short section its content and the rest to the one that overflows', () => {
  expect(shareHeight(plain(1000, 120), 400)).toEqual({ heights: [280, 120], rest: 0 });
});

it('shares equally between sections that both overflow', () => {
  expect(shareHeight(plain(1000, 600, 50), 450)).toEqual({ heights: [200, 200, 50], rest: 0 });
});

it('gives nothing when the headers take all the height', () => {
  expect(shareHeight(plain(100, 100), -20)).toEqual({ heights: [0, 0], rest: 0 });
});

// Catches a list cut through a row, as a three-row list a few pixels short of its content that scrolls a little and
// fades its last row.
it('ends an overflowing list on a row and leaves the rest as a gap below it', () => {
  expect(shareHeight([list(3)], 90)).toEqual({ heights: [61], rest: 29 });
});

// Catches the space a cut row would have taken going nowhere, to a list that already shows more, or to another row
// of a long list while a short one is left scrolling by its last row.
it('gives what ending on rows frees first to a list it completes, then to the list showing least, a row at a time', () => {
  const { heights, rest } = shareHeight([list(10), list(10), plain(40)[0]], 160);
  expect(heights).toEqual([61, 32, 67]);
  expect(rest).toBe(0);
  expect(shareHeight([list(10), list(2)], 120).heights).toEqual([32, 88]);
  expect(shareHeight([list(3), list(10)], 190)).toEqual({ heights: [94, 90], rest: 6 });
});
