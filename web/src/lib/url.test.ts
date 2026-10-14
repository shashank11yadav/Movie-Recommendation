import { describe, expect, it } from 'vitest';
import { realData } from './test-fixtures';
import { MAX_LIKED, parseHash, toHash } from './url';

const { catalog } = realData();
const row = (id: number) => catalog.rowOf.get(id);

describe('URL state', () => {
  it('round-trips liked movies and model', () => {
    const state = {
      liked: [row(296), row(1)].filter((r) => r !== undefined),
      model: 'collab' as const,
    };
    const hash = toHash(state, catalog);
    expect(hash).toBe('#liked=296,1&model=collab');
    expect(parseHash(hash, catalog)).toEqual(state);
  });

  it('omits defaults', () => {
    expect(toHash({ liked: [], model: 'hybrid' }, catalog)).toBe('');
  });

  it('drops unknown ids, junk and duplicates', () => {
    const parsed = parseHash('#liked=296,abc,296,-1,99999999,<script>,1&model=evil', catalog);
    expect(parsed.liked).toEqual([row(296), row(1)]);
    expect(parsed.model).toBe('hybrid');
  });

  it('caps the number of liked movies', () => {
    const ids = catalog.id.slice(0, 50).join(',');
    expect(parseHash(`#liked=${ids}`, catalog).liked).toHaveLength(MAX_LIKED);
  });
});
