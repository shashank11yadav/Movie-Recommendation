import { describe, expect, it } from 'vitest';
import { realData } from './test-fixtures';
import { createSearch, fold, MAX_QUERY_LENGTH, movieLabel } from './search';

const { catalog } = realData();
const search = createSearch(catalog);
const labels = (query: string) => search(query).map((row) => movieLabel(catalog, row));

describe('fold', () => {
  it('drops accents and case', () => {
    expect(fold('Amélie')).toBe('amelie');
    expect(fold('ÉLITE')).toBe('elite');
  });
});

describe('createSearch', () => {
  it('finds titles regardless of case and moved articles', () => {
    expect(labels('the matrix')[0]).toBe('The Matrix (1999)');
    expect(labels('PULP FICTION')[0]).toBe('Pulp Fiction (1994)');
  });

  it('matches prefixes while typing', () => {
    expect(labels('shawsh')).toContain('The Shawshank Redemption (1994)');
  });

  it('forgives a typo in a longer word', () => {
    expect(labels('godfathr')).toContain('The Godfather (1972)');
  });

  it('matches accented titles without accents', () => {
    expect(labels('miserables').some((l) => l.startsWith('Les Misérables'))).toBe(true);
  });

  it('uses the year to narrow results', () => {
    expect(labels('nightmare elm street 1984')[0]).toBe('A Nightmare on Elm Street (1984)');
  });

  it('ranks the well-known film first among similar titles', () => {
    expect(labels('toy story')[0]).toBe('Toy Story (1995)');
  });

  it('returns nothing for blank input and caps very long input', () => {
    expect(search('   ')).toEqual([]);
    expect(() => search('a'.repeat(MAX_QUERY_LENGTH * 50))).not.toThrow();
  });

  it('respects the limit', () => {
    expect(search('the', 3)).toHaveLength(3);
  });
});
