// Fuzzy title search over the catalog with MiniSearch. Everything runs in the
// browser; the index is built once from movies.json (~10k titles, a few ms).

import MiniSearch from 'minisearch';
import type { Catalog } from './data';

export const MAX_QUERY_LENGTH = 100;

/** Lowercase and strip accents so "amelie" finds "Amélie". */
export function fold(text: string): string {
  return text.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase();
}

interface Doc {
  id: number;
  title: string;
  year: string;
}

export type SearchFn = (query: string, limit?: number) => number[];

export function createSearch(catalog: Catalog): SearchFn {
  const index = new MiniSearch<Doc>({
    fields: ['title', 'year'],
    processTerm: (term) => fold(term),
    searchOptions: {
      boost: { title: 2 },
      prefix: true,
      // Typos are only forgiven on longer words, otherwise "up" matches everything.
      fuzzy: (term) => (term.length > 3 ? 0.2 : false),
      combineWith: 'AND',
      // Among equally good text matches, prefer films people actually rated.
      boostDocument: (row: number) => 1 + Math.log10(1 + (catalog.ratings[row] ?? 0)) / 2,
    },
  });
  index.addAll(
    catalog.title.map((title, row) => ({
      id: row,
      title,
      year: String(catalog.year[row] ?? ''),
    })),
  );

  return (query, limit = 8) => {
    const cleaned = query.slice(0, MAX_QUERY_LENGTH).trim();
    if (cleaned.length === 0) return [];
    return index
      .search(cleaned)
      .slice(0, limit)
      .map((hit) => hit.id as number);
  };
}

export function movieLabel(catalog: Catalog, row: number): string {
  const year = catalog.year[row];
  const title = catalog.title[row] ?? 'Unknown title';
  return year === null || year === undefined ? title : `${title} (${year})`;
}
