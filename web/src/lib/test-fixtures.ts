// Test helper: the real exported data, read from disk.
import { readFileSync } from 'node:fs';
import { parseCatalog, parseMetrics, parseNeighbours, type AppData } from './data';

function read(name: string): unknown {
  return JSON.parse(readFileSync(new URL(`../../public/data/${name}`, import.meta.url), 'utf8'));
}

let cached: AppData | undefined;

export function realData(): AppData {
  if (!cached) {
    const catalog = parseCatalog(read('movies.json'));
    cached = {
      catalog,
      neighbours: parseNeighbours(read('neighbors.json'), catalog.id.length),
      metrics: parseMetrics(read('metrics.json')),
    };
  }
  return cached;
}
