import type { Catalog } from '../lib/data';
import { movieLabel } from '../lib/search';

interface Props {
  catalog: Catalog;
  liked: readonly number[];
  onRemove: (row: number) => void;
}

export function LikedList({ catalog, liked, onRemove }: Props) {
  if (liked.length === 0) return null;
  return (
    <ul className="liked" aria-label="Films you like">
      {liked.map((row) => {
        const label = movieLabel(catalog, row);
        return (
          <li key={row} className="liked-item">
            <span>{label}</span>
            <button
              type="button"
              className="remove"
              aria-label={`Remove ${label}`}
              onClick={() => {
                onRemove(row);
              }}
            >
              <span aria-hidden="true">×</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
