import type { Catalog, ModelName } from '../lib/data';
import type { Recommendation } from '../lib/recommend';
import { movieLabel } from '../lib/search';

const MAX_CHIPS = 5;

interface Props {
  catalog: Catalog;
  model: ModelName;
  recommendations: readonly Recommendation[];
  likedCount: number;
}

function percent(share: number): number {
  return Math.round(share * 100);
}

function SignalBar({ collabShare }: { collabShare: number }) {
  const collab = percent(collabShare);
  const content = 100 - collab;
  return (
    <div className="signal">
      <div className="signal-bar" aria-hidden="true">
        <span className="signal-collab" style={{ width: `${collab}%` }} />
        <span className="signal-content" style={{ width: `${content}%` }} />
      </div>
      <span className="signal-text">
        {collab}% co-ratings, {content}% content
      </span>
    </div>
  );
}

export function Results({ catalog, model, recommendations, likedCount }: Props) {
  if (likedCount === 0) return null;
  return (
    <section className="results" aria-labelledby="results-heading">
      <h2 id="results-heading">Recommendations</h2>
      <p className="visually-hidden" aria-live="polite" aria-atomic="true">
        {recommendations.length} recommendations from the {model} model.
      </p>
      {recommendations.length === 0 ? (
        <p className="empty">
          No recommendations from this model for your selection. Try the hybrid model.
        </p>
      ) : (
        <ol className="result-list">
          {recommendations.map((rec, i) => {
            const chips = [...rec.sharedGenres, ...rec.sharedTags].slice(0, MAX_CHIPS);
            const genres = (catalog.genre[rec.row] ?? []).map((g) => catalog.genres[g]).join(', ');
            return (
              <li key={rec.row} className="card">
                <span className="rank" aria-hidden="true">
                  {i + 1}
                </span>
                <div className="card-body">
                  <h3 className="card-title">{movieLabel(catalog, rec.row)}</h3>
                  {genres && <p className="card-genres">{genres}</p>}
                  <p className="because">
                    Because you liked <strong>{movieLabel(catalog, rec.because)}</strong>
                  </p>
                  {chips.length > 0 ? (
                    <ul className="chips" aria-label="Shared genres and tags">
                      {chips.map((chip) => (
                        <li key={chip} className="chip">
                          {chip}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="card-note">
                      {rec.collabShare >= 0.5
                        ? 'No shared genres or tags: linked by people who rated both.'
                        : 'No shared genres or tags: similar title or era.'}
                    </p>
                  )}
                  <SignalBar collabShare={rec.collabShare} />
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
