import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { DataError, parseCatalog, parseMetrics, parseNeighbours } from './data';

const movies = {
  genres: ['Crime', 'Drama'],
  id: [1, 2],
  title: ['A', 'B'],
  year: [1994, null],
  genre: [[0], [0, 1]],
  tags: [['heist'], []],
  ratings: [3, 0],
};

const neighbours = {
  scale: 1000,
  hybrid: { beta: 1, gamma: 1 },
  content: [
    [1, 500],
    [0, 500],
  ],
  collab: [[], []],
};

describe('parseCatalog', () => {
  it('accepts well-formed data and builds the id map', () => {
    const catalog = parseCatalog(movies);
    expect(catalog.rowOf.get(2)).toBe(1);
  });

  it.each([
    ['a non-object', null],
    ['mismatched columns', { ...movies, title: ['A'] }],
    ['duplicate ids', { ...movies, id: [1, 1] }],
    ['a genre index out of range', { ...movies, genre: [[0], [7]] }],
    ['a non-integer year', { ...movies, year: ['1994', null] }],
    ['no movies', { ...movies, id: [], title: [], year: [], genre: [], tags: [], ratings: [] }],
  ])('rejects %s', (_label, raw) => {
    expect(() => parseCatalog(raw)).toThrow(DataError);
  });
});

describe('parseNeighbours', () => {
  it('accepts valid lists', () => {
    expect(parseNeighbours(neighbours, 2).content[0]).toEqual([1, 500]);
  });

  it.each([
    ['a self reference', { ...neighbours, content: [[0, 500], []] }],
    ['an out-of-range row', { ...neighbours, content: [[5, 500], []] }],
    ['an odd-length list', { ...neighbours, content: [[1], []] }],
    ['a score above the scale', { ...neighbours, content: [[1, 5000], []] }],
    ['the wrong number of lists', { ...neighbours, collab: [[]] }],
    ['a negative beta', { ...neighbours, hybrid: { beta: -1, gamma: 1 } }],
  ])('rejects %s', (_label, raw) => {
    expect(() => parseNeighbours(raw, 2)).toThrow(DataError);
  });
});

describe('parseMetrics', () => {
  it('accepts the committed metrics file', () => {
    const raw: unknown = JSON.parse(
      readFileSync(new URL('../../public/data/metrics.json', import.meta.url), 'utf8'),
    );
    expect(parseMetrics(raw).k).toBe(10);
  });

  it('rejects a metric that is not a number', () => {
    const bad = {
      users: 1,
      k: 10,
      settings: {},
      query_coverage: {},
      profiles: { p: { m: { hit_rate: 'x' } } },
    };
    expect(() => parseMetrics(bad)).toThrow(DataError);
  });
});
