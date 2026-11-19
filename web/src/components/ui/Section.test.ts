import { expect, it } from 'vitest';
import { shareHeight } from './Section';

it('gives what no section needs to the last open section', () => {
  expect(shareHeight([100, 50], 400)).toEqual([100, 300]);
});

it('gives a short section its content and the rest to the one that overflows', () => {
  expect(shareHeight([1000, 120], 400)).toEqual([280, 120]);
});

it('shares equally between sections that both overflow', () => {
  expect(shareHeight([1000, 600, 50], 450)).toEqual([200, 200, 50]);
});

it('gives nothing when the headers take all the height', () => {
  expect(shareHeight([100, 100], -20)).toEqual([0, 0]);
});
