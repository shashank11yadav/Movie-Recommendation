// Loading and validating the JSON files produced by the pipeline (pipeline/src/movierec/export.py).
// They are same-origin static files, but they are still parsed as untrusted input:
// a stale or truncated deploy should fail with a clear message, not render nonsense.

export type ModelName = 'content' | 'collab' | 'hybrid';
export const MODEL_NAMES: readonly ModelName[] = ['hybrid', 'collab', 'content'];

/** Flat [row, score, row, score, ...] per movie. */
export type NeighbourLists = readonly (readonly number[])[];

export interface Catalog {
  readonly genres: readonly string[];
  readonly id: readonly number[];
  readonly title: readonly string[];
  readonly year: readonly (number | null)[];
  readonly genre: readonly (readonly number[])[];
  readonly tags: readonly (readonly string[])[];
  readonly ratings: readonly number[];
  readonly rowOf: ReadonlyMap<number, number>;
}

export interface Neighbours {
  readonly scale: number;
  readonly hybrid: { readonly beta: number; readonly gamma: number };
  readonly content: NeighbourLists;
  readonly collab: NeighbourLists;
}

export interface ModelMetrics {
  readonly hit_rate: number;
  readonly hit_rate_ci: readonly [number, number];
  readonly ndcg: number;
  readonly coverage: number;
  readonly avg_popularity: number;
  readonly long_tail_share: number;
}

export interface Metrics {
  readonly users: number;
  readonly catalog_size: number;
  readonly k: number;
  readonly neighbors_per_movie: number;
  readonly settings: Readonly<Record<string, string | number>>;
  readonly profiles: Readonly<Record<string, Readonly<Record<string, ModelMetrics>>>>;
  readonly query_coverage: Readonly<Record<string, number>>;
}

export class DataError extends Error {
  override name = 'DataError';
}

type Json = unknown;

function fail(message: string): never {
  throw new DataError(message);
}

function record(value: Json, what: string): Record<string, Json> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    fail(`${what} is not an object`);
  }
  return value as Record<string, Json>;
}

function array(value: Json, what: string): Json[] {
  if (!Array.isArray(value)) fail(`${what} is not an array`);
  return value;
}

function isInt(value: Json): value is number {
  return typeof value === 'number' && Number.isInteger(value);
}

function finite(value: Json, what: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail(`${what} is not a number`);
  return value;
}

function ints(value: Json, what: string): number[] {
  const list = array(value, what);
  if (!list.every(isInt)) fail(`${what} must contain integers`);
  return list;
}

function strings(value: Json, what: string): string[] {
  const list = array(value, what);
  if (!list.every((v) => typeof v === 'string')) fail(`${what} must contain strings`);
  return list;
}

export function parseCatalog(raw: Json): Catalog {
  const obj = record(raw, 'movies.json');
  const genres = strings(obj['genres'], 'genres');
  const id = ints(obj['id'], 'id');
  const n = id.length;
  if (n === 0) fail('movies.json has no movies');

  const title = strings(obj['title'], 'title');
  const year = array(obj['year'], 'year').map((y) =>
    y === null ? null : isInt(y) ? y : fail('year must be an integer or null'),
  );
  const genre = array(obj['genre'], 'genre').map((g) => {
    const list = ints(g, 'genre entry');
    if (!list.every((i) => i >= 0 && i < genres.length)) fail('genre index out of range');
    return list;
  });
  const tags = array(obj['tags'], 'tags').map((t) => strings(t, 'tag list'));
  const ratings = ints(obj['ratings'], 'ratings');

  for (const [name, column] of Object.entries({ title, year, genre, tags, ratings })) {
    if (column.length !== n) fail(`${name} has ${column.length} entries, expected ${n}`);
  }
  const rowOf = new Map(id.map((movieId, row) => [movieId, row]));
  if (rowOf.size !== n) fail('movie ids are not unique');
  return { genres, id, title, year, genre, tags, ratings, rowOf };
}

function lists(value: Json, what: string, n: number, scale: number): number[][] {
  const rows = array(value, what);
  if (rows.length !== n) fail(`${what} has ${rows.length} lists, expected ${n}`);
  return rows.map((entry, row) => {
    const flat = ints(entry, `${what}[${row}]`);
    if (flat.length % 2 !== 0) fail(`${what}[${row}] has an odd length`);
    for (let i = 0; i < flat.length; i += 2) {
      const target = flat[i] ?? -1;
      const score = flat[i + 1] ?? -1;
      if (target < 0 || target >= n || target === row) fail(`${what}[${row}] has a bad neighbour`);
      if (score <= 0 || score > scale) fail(`${what}[${row}] has a bad score`);
    }
    return flat;
  });
}

export function parseNeighbours(raw: Json, catalogSize: number): Neighbours {
  const obj = record(raw, 'neighbors.json');
  const scale = finite(obj['scale'], 'scale');
  if (scale <= 0) fail('scale must be positive');
  const hybrid = record(obj['hybrid'], 'hybrid');
  const beta = finite(hybrid['beta'], 'beta');
  const gamma = finite(hybrid['gamma'], 'gamma');
  if (beta <= 0 || gamma < 0) fail('hybrid weights out of range');
  return {
    scale,
    hybrid: { beta, gamma },
    content: lists(obj['content'], 'content', catalogSize, scale),
    collab: lists(obj['collab'], 'collab', catalogSize, scale),
  };
}

export function parseMetrics(raw: Json): Metrics {
  const obj = record(raw, 'metrics.json');
  const profiles = record(obj['profiles'], 'profiles');
  for (const [profile, byModel] of Object.entries(profiles)) {
    for (const [model, m] of Object.entries(record(byModel, profile))) {
      const metrics = record(m, `${profile}.${model}`);
      for (const key of ['hit_rate', 'ndcg', 'coverage', 'avg_popularity', 'long_tail_share']) {
        finite(metrics[key], `${profile}.${model}.${key}`);
      }
      const ci = array(metrics['hit_rate_ci'], 'hit_rate_ci');
      if (ci.length !== 2) fail('hit_rate_ci must have two bounds');
      ci.forEach((v) => finite(v, 'hit_rate_ci'));
    }
  }
  finite(obj['users'], 'users');
  finite(obj['k'], 'k');
  record(obj['settings'], 'settings');
  record(obj['query_coverage'], 'query_coverage');
  return obj as unknown as Metrics;
}

async function fetchJson(path: string, signal: AbortSignal): Promise<Json> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${path}`, {
    signal,
    credentials: 'same-origin',
  });
  if (!response.ok) throw new DataError(`could not load ${path} (HTTP ${response.status})`);
  return (await response.json()) as Json;
}

export interface AppData {
  readonly catalog: Catalog;
  readonly neighbours: Neighbours;
  readonly metrics: Metrics;
}

export async function loadData(signal: AbortSignal): Promise<AppData> {
  const [movies, neighbours, metrics] = await Promise.all([
    fetchJson('movies.json', signal),
    fetchJson('neighbors.json', signal),
    fetchJson('metrics.json', signal),
  ]);
  const catalog = parseCatalog(movies);
  return {
    catalog,
    neighbours: parseNeighbours(neighbours, catalog.id.length),
    metrics: parseMetrics(metrics),
  };
}
