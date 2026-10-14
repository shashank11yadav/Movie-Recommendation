// Client-side scoring. This mirrors `recommend` in pipeline/src/movierec/scoring.py
// step for step; golden.test-data.json (exported by the pipeline) keeps them in sync.

import type { Catalog, ModelName, Neighbours } from './data';

export interface Recommendation {
  readonly row: number;
  readonly score: number;
  /** Fraction of the score that came from co-rating (collaborative) neighbours. */
  readonly collabShare: number;
  /** The liked movie that contributed most to this result. */
  readonly because: number;
  readonly sharedGenres: readonly string[];
  readonly sharedTags: readonly string[];
}

// Rounding before ranking stops float summation order from reordering near-ties
// differently here and in numpy.
const SCORE_DECIMALS = 1e9;

function round(score: number): number {
  return Math.round(score * SCORE_DECIMALS) / SCORE_DECIMALS;
}

/** Weights on (collaborative, content) lists for one liked movie. */
export function sourceWeights(
  model: ModelName,
  ratingCount: number,
  hybrid: Neighbours['hybrid'],
): [number, number] {
  switch (model) {
    case 'content':
      return [0, 1];
    case 'collab':
      return [1, 0];
    case 'hybrid': {
      const w = ratingCount / (ratingCount + hybrid.beta);
      return [w, (1 - w) * hybrid.gamma];
    }
  }
}

export function recommend(
  catalog: Catalog,
  neighbours: Neighbours,
  liked: readonly number[],
  model: ModelName,
  limit = 10,
): Recommendation[] {
  const sources = [...new Set(liked)];
  const perSource = sources.map(() => new Map<number, number>());
  const fromCollab = new Map<number, number>();

  sources.forEach((src, i) => {
    const contributions = perSource[i];
    if (!contributions) return;
    // A film without a collaborative list counts as unrated, so its whole
    // hybrid weight goes to content rather than to an empty list.
    const hasCollab = (neighbours.collab[src]?.length ?? 0) > 0;
    const ratingCount = hasCollab ? (catalog.ratings[src] ?? 0) : 0;
    const [wCollab, wContent] = sourceWeights(model, ratingCount, neighbours.hybrid);
    const add = (list: readonly number[] | undefined, weight: number, collab: boolean) => {
      if (!list || weight === 0) return;
      for (let j = 0; j < list.length; j += 2) {
        const target = list[j] ?? -1;
        const part = (weight * (list[j + 1] ?? 0)) / neighbours.scale;
        contributions.set(target, (contributions.get(target) ?? 0) + part);
        if (collab) fromCollab.set(target, (fromCollab.get(target) ?? 0) + part);
      }
    };
    add(neighbours.collab[src], wCollab, true);
    add(neighbours.content[src], wContent, false);
  });

  const excluded = new Set(sources);
  const totals = new Map<number, number>();
  for (const contributions of perSource) {
    for (const [target, part] of contributions) {
      totals.set(target, (totals.get(target) ?? 0) + part);
    }
  }

  const ranked = [...totals]
    .map(([row, total]) => ({ row, total, rounded: round(total) }))
    .filter(({ row, rounded }) => rounded > 0 && !excluded.has(row))
    .sort(
      (a, b) =>
        b.rounded - a.rounded ||
        (catalog.ratings[b.row] ?? 0) - (catalog.ratings[a.row] ?? 0) ||
        a.row - b.row,
    )
    .slice(0, limit);

  return ranked.map(({ row, total, rounded }) => {
    let because = sources[0] ?? row;
    let best = -Infinity;
    sources.forEach((src, i) => {
      const part = perSource[i]?.get(row) ?? 0;
      if (part > best) {
        best = part;
        because = src;
      }
    });
    const likedGenres = new Set(catalog.genre[because]);
    const likedTags = new Set(catalog.tags[because]);
    return {
      row,
      score: rounded,
      collabShare: (fromCollab.get(row) ?? 0) / total,
      because,
      sharedGenres: (catalog.genre[row] ?? [])
        .filter((g) => likedGenres.has(g))
        .map((g) => catalog.genres[g] ?? '')
        .sort(),
      sharedTags: (catalog.tags[row] ?? []).filter((t) => likedTags.has(t)),
    };
  });
}

/** Liked movies that have no collaborative neighbours (too few ratings). */
export function coldLiked(neighbours: Neighbours, liked: readonly number[]): number[] {
  return liked.filter((row) => (neighbours.collab[row]?.length ?? 0) === 0);
}
