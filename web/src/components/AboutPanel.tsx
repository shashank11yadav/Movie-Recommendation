import type { Metrics } from '../lib/data';

const MODEL_LABELS: Record<string, string> = {
  popularity: 'Popularity baseline',
  content: 'Content',
  collab: 'Collaborative',
  hybrid: 'Hybrid',
};

const PROFILE_LABELS: Record<string, string> = {
  'recent-5': 'Query: each user’s 5 most recent ratings (closest to how this demo is used)',
  full: 'Query: all of a user’s training ratings',
};

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export function AboutPanel({ metrics }: { metrics: Metrics }) {
  const s = metrics.settings;
  return (
    <details className="about">
      <summary>About the model</summary>
      <div className="about-body">
        <p>
          Everything here was computed offline from <strong>MovieLens ml-latest-small</strong> (
          {metrics.catalog_size.toLocaleString('en')} films, {metrics.users} users). The browser
          only adds up precomputed neighbour lists: each film keeps its{' '}
          {metrics.neighbors_per_movie} closest films under two signals.
        </p>
        <ul>
          <li>
            <strong>Content</strong>: cosine similarity of{' '}
            <code>{String(s['embedding_model'])}</code> embeddings of title, year, genres and user
            tags.
          </li>
          <li>
            <strong>Collaborative</strong>: item-kNN, cosine over who rated what, with shrinkage{' '}
            {String(s['cf_shrink'])} to damp pairs that share only a few raters.
          </li>
          <li>
            <strong>Hybrid</strong>: each liked film with <em>n</em> ratings contributes its
            collaborative list with weight <em>n</em> / (<em>n</em> + {String(s['hybrid_beta'])})
            and its content list with the rest, scaled by {String(s['hybrid_gamma'])}. A film with
            fewer than {String(s['cf_min_ratings'])} ratings has no collaborative list and counts as{' '}
            <em>n</em> = 0, so it falls back to content entirely.
          </li>
        </ul>
        <p>
          Evaluation holds out each user’s most recent rating (temporal leave-last-out) and asks
          whether it shows up in the top {metrics.k}. Settings were tuned on the second-to-last
          rating, so the test rating was never used for tuning. Intervals are 95% bootstrap over
          users; with {metrics.users} users, small gaps between models are within noise.
        </p>
        {Object.entries(metrics.profiles).map(([profile, byModel]) => (
          <div className="table-wrap" key={profile}>
            <table>
              <caption>{PROFILE_LABELS[profile] ?? profile}</caption>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">HitRate@{metrics.k}</th>
                  <th scope="col">NDCG@{metrics.k}</th>
                  <th scope="col">Coverage</th>
                  <th scope="col">Long-tail share</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(byModel).map(([model, m]) => (
                  <tr key={model}>
                    <th scope="row">{MODEL_LABELS[model] ?? model}</th>
                    <td>
                      {m.hit_rate.toFixed(3)}{' '}
                      <span className="ci">
                        ({m.hit_rate_ci[0].toFixed(3)}–{m.hit_rate_ci[1].toFixed(3)})
                      </span>
                    </td>
                    <td>{m.ndcg.toFixed(3)}</td>
                    <td>{pct(m.coverage)}</td>
                    <td>{pct(m.long_tail_share)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
        <p>
          Picked on its own, a film returns a full list {pct(metrics.query_coverage['collab'] ?? 0)}{' '}
          of the time with the collaborative model and {pct(metrics.query_coverage['hybrid'] ?? 0)}{' '}
          with the hybrid.
        </p>
      </div>
    </details>
  );
}
