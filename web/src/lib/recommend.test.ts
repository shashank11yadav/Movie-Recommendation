import { describe, expect, it } from 'vitest';
import type { Catalog, ModelName, Neighbours } from './data';
import { realData } from './test-fixtures';
import golden from './golden.test-data.json';
import { coldLiked, recommend, sourceWeights } from './recommend';

function tinyCatalog(): Catalog {
  // Rows: 0 Heat, 1 Ronin, 2 Toy Story, 3 Obscure film
  return {
    genres: ['Action', 'Animation', 'Crime', 'Drama'],
    id: [6, 10, 1, 99],
    title: ['Heat', 'Ronin', 'Toy Story', 'Obscure film'],
    year: [1995, 1998, 1995, null],
    genre: [[0, 2], [0, 2], [1], [3]],
    tags: [['heist', 'la'], ['heist'], ['pixar'], []],
    ratings: [100, 50, 200, 1],
    rowOf: new Map([
      [6, 0],
      [10, 1],
      [1, 2],
      [99, 3],
    ]),
  };
}

function tinyNeighbours(): Neighbours {
  return {
    scale: 1000,
    hybrid: { beta: 1, gamma: 1 },
    content: [
      [1, 900, 3, 400],
      [0, 900],
      [3, 300],
      [0, 500, 2, 500],
    ],
    collab: [[1, 600, 2, 200], [0, 600], [0, 200], []],
  };
}

describe('sourceWeights', () => {
  it('moves weight to content for rarely rated movies', () => {
    const hybrid = { beta: 4, gamma: 0.5 };
    expect(sourceWeights('hybrid', 0, hybrid)).toEqual([0, 0.5]);
    expect(sourceWeights('hybrid', 4, hybrid)).toEqual([0.5, 0.25]);
    expect(sourceWeights('content', 1000, hybrid)).toEqual([0, 1]);
    expect(sourceWeights('collab', 0, hybrid)).toEqual([1, 0]);
  });
});

describe('recommend', () => {
  const catalog = tinyCatalog();
  const neighbours = tinyNeighbours();

  it('sums neighbour scores across liked movies and never returns a liked movie', () => {
    const recs = recommend(catalog, neighbours, [0, 1], 'content');
    const rows = recs.map((r) => r.row);
    expect(rows).toEqual([3]);
    expect(recs[0]?.score).toBeCloseTo(0.4);
  });

  it('explains each result with the strongest liked movie and shared metadata', () => {
    const [top] = recommend(catalog, neighbours, [0], 'collab');
    expect(top).toMatchObject({
      row: 1,
      because: 0,
      collabShare: 1,
      sharedGenres: ['Action', 'Crime'],
      sharedTags: ['heist'],
    });
  });

  it('breaks score ties by rating count', () => {
    const recs = recommend(catalog, neighbours, [3], 'content');
    expect(recs.map((r) => r.row)).toEqual([2, 0]); // both 0.5, Toy Story has more ratings
  });

  it('falls back to content for a movie without collaborative neighbours', () => {
    expect(recommend(catalog, neighbours, [3], 'collab')).toEqual([]);
    expect(coldLiked(neighbours, [0, 3])).toEqual([3]);
    const hybrid = recommend(catalog, neighbours, [3], 'hybrid');
    expect(hybrid.length).toBeGreaterThan(0);
    expect(hybrid.every((r) => r.collabShare === 0)).toBe(true);
  });

  it('treats a repeated like as one', () => {
    expect(recommend(catalog, neighbours, [0, 0], 'hybrid')).toEqual(
      recommend(catalog, neighbours, [0], 'hybrid'),
    );
  });

  it('returns nothing for an empty selection', () => {
    expect(recommend(catalog, neighbours, [], 'hybrid')).toEqual([]);
  });
});

describe('parity with the Python pipeline', () => {
  const { catalog, neighbours } = realData();

  it.each(golden.cases.map((c) => [c.model, c.liked.join('+'), c] as const))(
    '%s for %s matches the golden file',
    (_model, _liked, testCase) => {
      const liked = testCase.liked.map((id) => {
        const row = catalog.rowOf.get(id);
        if (row === undefined) throw new Error(`unknown id ${id}`);
        return row;
      });
      const recs = recommend(catalog, neighbours, liked, testCase.model as ModelName);
      expect(recs.map((r) => catalog.id[r.row])).toEqual(testCase.results.map((r) => r.id));
      recs.forEach((rec, i) => {
        const expected = testCase.results[i];
        if (!expected) throw new Error('missing golden row');
        expect(rec.score).toBeCloseTo(expected.score, 9);
        expect(rec.collabShare).toBeCloseTo(expected.collabShare, 6);
        expect(catalog.id[rec.because]).toBe(expected.because);
        expect(rec.sharedGenres).toEqual(expected.sharedGenres);
        expect(rec.sharedTags).toEqual(expected.sharedTags);
      });
    },
  );

  it('has golden cases for every model', () => {
    expect(new Set(golden.cases.map((c) => c.model))).toEqual(
      new Set(['content', 'collab', 'hybrid']),
    );
  });
});
