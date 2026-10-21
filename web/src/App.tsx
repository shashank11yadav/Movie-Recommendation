import { useCallback, useEffect, useMemo, useState } from 'react';
import { AboutPanel } from './components/AboutPanel';
import { LikedList } from './components/LikedList';
import { ModelToggle } from './components/ModelToggle';
import { Results } from './components/Results';
import { SearchBox } from './components/SearchBox';
import { loadData, type AppData, type ModelName } from './lib/data';
import { coldLiked, recommend } from './lib/recommend';
import { createSearch, movieLabel } from './lib/search';
import { MAX_LIKED, parseHash, toHash } from './lib/url';

const REPO_URL = 'https://github.com/shashank11yadav/Movie-Recommendation';
const CASE_STUDY_URL = 'https://www.shashankyadav.dev/work/movie-recommendation';

const EXAMPLES: readonly { label: string; ids: readonly number[] }[] = [
  { label: 'Pulp Fiction', ids: [296] },
  { label: 'Toy Story + Finding Nemo', ids: [1, 6377] },
  { label: 'Spirited Away', ids: [5618] },
  { label: 'Alien + Blade Runner', ids: [1214, 541] },
];

type LoadState =
  { status: 'loading' } | { status: 'error'; message: string } | { status: 'ready'; data: AppData };

function Demo({ data }: { data: AppData }) {
  const { catalog, neighbours, metrics } = data;
  const initial = useMemo(() => parseHash(window.location.hash, catalog), [catalog]);
  const [liked, setLiked] = useState<number[]>(initial.liked);
  const [model, setModel] = useState<ModelName>(initial.model);
  const search = useMemo(() => createSearch(catalog), [catalog]);

  useEffect(() => {
    const hash = toHash({ liked, model }, catalog);
    if (hash !== window.location.hash) {
      window.history.replaceState(null, '', hash || window.location.pathname);
    }
  }, [liked, model, catalog]);

  useEffect(() => {
    const onHashChange = () => {
      const next = parseHash(window.location.hash, catalog);
      setLiked(next.liked);
      setModel(next.model);
    };
    window.addEventListener('hashchange', onHashChange);
    return () => {
      window.removeEventListener('hashchange', onHashChange);
    };
  }, [catalog]);

  const recommendations = useMemo(
    () => recommend(catalog, neighbours, liked, model),
    [catalog, neighbours, liked, model],
  );
  const cold = useMemo(() => coldLiked(neighbours, liked), [neighbours, liked]);
  const likedSet = useMemo(() => new Set(liked), [liked]);

  const add = useCallback((row: number) => {
    setLiked((current) =>
      current.includes(row) || current.length >= MAX_LIKED ? current : [...current, row],
    );
  }, []);
  const remove = useCallback((row: number) => {
    setLiked((current) => current.filter((r) => r !== row));
  }, []);
  const loadExample = (ids: readonly number[]) => {
    setLiked(ids.map((id) => catalog.rowOf.get(id)).filter((r) => r !== undefined));
  };

  return (
    <>
      <section className="panel" aria-labelledby="pick-heading">
        <h2 id="pick-heading">Films you like</h2>
        <SearchBox
          catalog={catalog}
          search={search}
          exclude={likedSet}
          disabled={liked.length >= MAX_LIKED}
          onPick={add}
        />
        <LikedList catalog={catalog} liked={liked} onRemove={remove} />
        {liked.length === 0 && (
          <div className="examples">
            <p>Or start from an example:</p>
            <ul>
              {EXAMPLES.map((example) => (
                <li key={example.label}>
                  <button
                    type="button"
                    className="example"
                    onClick={() => {
                      loadExample(example.ids);
                    }}
                  >
                    {example.label}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
        <ModelToggle value={model} onChange={setModel} />
        {model === 'collab' && cold.length > 0 && (
          <p className="notice" role="note">
            {cold.map((row) => movieLabel(catalog, row)).join(', ')}{' '}
            {cold.length === 1 ? 'has' : 'have'} too few ratings for collaborative neighbours.{' '}
            <button
              type="button"
              className="link-button"
              onClick={() => {
                setModel('hybrid');
              }}
            >
              Switch to hybrid
            </button>{' '}
            to fall back on content.
          </p>
        )}
      </section>
      <Results
        catalog={catalog}
        model={model}
        recommendations={recommendations}
        likedCount={liked.length}
      />
      <AboutPanel metrics={metrics} />
    </>
  );
}

export function App() {
  const [state, setState] = useState<LoadState>({ status: 'loading' });

  useEffect(() => {
    const controller = new AbortController();
    loadData(controller.signal)
      .then((data) => {
        setState({ status: 'ready', data });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const message = error instanceof Error ? error.message : 'unknown error';
        setState({ status: 'error', message });
      });
    return () => {
      controller.abort();
    };
  }, []);

  return (
    <div className="page">
      <header className="masthead">
        <h1>Movie Recommender</h1>
        <p className="lede">
          Pick a few films you like. Recommendations come from 100,836 MovieLens ratings and
          sentence embeddings, precomputed offline. Your picks never leave the browser.
        </p>
      </header>
      <main>
        {state.status === 'loading' && (
          <p className="status" role="status">
            Loading the catalog…
          </p>
        )}
        {state.status === 'error' && (
          <p className="status error" role="alert">
            Could not load the recommendation data ({state.message}). Try reloading the page.
          </p>
        )}
        {state.status === 'ready' && <Demo data={state.data} />}
      </main>
      <footer className="footer">
        <p>
          Data:{' '}
          <a href="https://grouplens.org/datasets/movielens/" rel="noopener noreferrer">
            MovieLens ml-latest-small
          </a>{' '}
          by GroupLens Research (Harper and Konstan, 2015), used under its non-commercial research
          license. Not endorsed by the University of Minnesota or GroupLens.
        </p>
        <p>
          <a href={REPO_URL} rel="noopener noreferrer">
            Source code and evaluation
          </a>{' '}
          <span aria-hidden="true">·</span>{' '}
          <a href={CASE_STUDY_URL} rel="noopener noreferrer">
            By Shashank Yadav · shashankyadav.dev
          </a>
        </p>
      </footer>
    </div>
  );
}
