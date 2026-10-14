// Accessible autocomplete following the WAI-ARIA combobox pattern: the input
// keeps focus, arrow keys move a virtual cursor exposed via aria-activedescendant.

import { useId, useMemo, useState, type KeyboardEvent } from 'react';
import type { Catalog } from '../lib/data';
import { MAX_QUERY_LENGTH, movieLabel, type SearchFn } from '../lib/search';

interface Props {
  catalog: Catalog;
  search: SearchFn;
  exclude: ReadonlySet<number>;
  disabled: boolean;
  onPick: (row: number) => void;
}

export function SearchBox({ catalog, search, exclude, disabled, onPick }: Props) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const inputId = useId();
  const listId = useId();
  const hintId = useId();

  const options = useMemo(
    () =>
      search(query, 12)
        .filter((row) => !exclude.has(row))
        .slice(0, 8),
    [search, query, exclude],
  );
  const expanded = open && query.trim().length > 0;
  const activeRow = options[Math.min(active, options.length - 1)];

  const pick = (row: number) => {
    onPick(row);
    setQuery('');
    setActive(0);
    setOpen(false);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setOpen(true);
        setActive((i) => (options.length ? (i + 1) % options.length : 0));
        break;
      case 'ArrowUp':
        event.preventDefault();
        setActive((i) => (options.length ? (i - 1 + options.length) % options.length : 0));
        break;
      case 'Enter':
        if (expanded && activeRow !== undefined) {
          event.preventDefault();
          pick(activeRow);
        }
        break;
      case 'Escape':
        if (expanded) {
          event.preventDefault();
          setOpen(false);
        } else {
          setQuery('');
        }
        break;
    }
  };

  return (
    <div className="search">
      <label htmlFor={inputId} className="search-label">
        Search for a film you like
      </label>
      <input
        id={inputId}
        type="search"
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={expanded}
        aria-controls={listId}
        aria-describedby={hintId}
        aria-activedescendant={
          expanded && activeRow !== undefined ? `${listId}-${activeRow}` : undefined
        }
        autoComplete="off"
        spellCheck={false}
        maxLength={MAX_QUERY_LENGTH}
        placeholder={disabled ? 'Remove a film to add another' : 'e.g. Pulp Fiction, Spirited Away'}
        disabled={disabled}
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(0);
          setOpen(true);
        }}
        onKeyDown={onKeyDown}
        onBlur={() => {
          setOpen(false);
        }}
        onFocus={() => {
          setOpen(true);
        }}
      />
      <p id={hintId} className="visually-hidden">
        Use the up and down arrow keys to choose a film and Enter to add it.
      </p>
      <ul
        id={listId}
        role="listbox"
        aria-label="Matching films"
        hidden={!expanded}
        className="options"
      >
        {options.map((row, i) => (
          <li
            key={row}
            id={`${listId}-${row}`}
            role="option"
            aria-selected={row === activeRow}
            className="option"
            // mousedown fires before the input's blur closes the list.
            onMouseDown={(e) => {
              e.preventDefault();
            }}
            onClick={() => {
              pick(row);
            }}
            onMouseEnter={() => {
              setActive(i);
            }}
          >
            <span className="option-title">{movieLabel(catalog, row)}</span>
            <span className="option-meta">{catalog.ratings[row] ?? 0} ratings</span>
          </li>
        ))}
        {options.length === 0 && (
          <li className="option option-empty" role="presentation">
            No matching films
          </li>
        )}
      </ul>
    </div>
  );
}
