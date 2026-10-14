// Shareable state in the URL fragment: #liked=296,1&model=hybrid
// The fragment is user-controlled input, so it is parsed strictly and anything
// unexpected is dropped rather than trusted.

import { MODEL_NAMES, type Catalog, type ModelName } from './data';

export const MAX_LIKED = 10;

export interface UrlState {
  liked: number[];
  model: ModelName;
}

const DEFAULT_MODEL: ModelName = 'hybrid';

function isModel(value: string | null): value is ModelName {
  return value !== null && (MODEL_NAMES as readonly string[]).includes(value);
}

/** Returns catalog rows for valid, known movie ids, deduplicated and capped. */
export function parseHash(hash: string, catalog: Catalog): UrlState {
  const params = new URLSearchParams(hash.replace(/^#/, '').slice(0, 512));
  const model = params.get('model');
  const rows: number[] = [];
  for (const part of (params.get('liked') ?? '').split(',')) {
    if (!/^\d{1,7}$/.test(part)) continue;
    const row = catalog.rowOf.get(Number(part));
    if (row !== undefined && !rows.includes(row)) rows.push(row);
    if (rows.length === MAX_LIKED) break;
  }
  return { liked: rows, model: isModel(model) ? model : DEFAULT_MODEL };
}

export function toHash(state: UrlState, catalog: Catalog): string {
  const ids = state.liked.map((row) => catalog.id[row]).filter((id) => id !== undefined);
  const params = new URLSearchParams();
  if (ids.length > 0) params.set('liked', ids.join(','));
  if (state.model !== DEFAULT_MODEL) params.set('model', state.model);
  const text = params.toString().replaceAll('%2C', ',');
  return text ? `#${text}` : '';
}
