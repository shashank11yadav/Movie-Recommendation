import type { ModelName } from '../lib/data';
import { MODEL_NAMES } from '../lib/data';

const MODEL_INFO: Record<ModelName, { label: string; blurb: string }> = {
  hybrid: {
    label: 'Hybrid',
    blurb:
      'Co-ratings for well-rated films, content similarity for films with few ratings. Weights were tuned on a validation split.',
  },
  collab: {
    label: 'Collaborative',
    blurb: 'Item-kNN: films rated by the same people. Needs at least a few ratings per film.',
  },
  content: {
    label: 'Content',
    blurb: 'Sentence embeddings of title, year, genres and user tags. Works for every film.',
  },
};

interface Props {
  value: ModelName;
  onChange: (model: ModelName) => void;
}

export function ModelToggle({ value, onChange }: Props) {
  return (
    <fieldset className="toggle">
      <legend>Model</legend>
      <div className="toggle-options">
        {MODEL_NAMES.map((name) => (
          <label key={name} className="toggle-option">
            <input
              type="radio"
              name="model"
              value={name}
              checked={value === name}
              onChange={() => {
                onChange(name);
              }}
            />
            <span>{MODEL_INFO[name].label}</span>
          </label>
        ))}
      </div>
      <p className="toggle-blurb">{MODEL_INFO[value].blurb}</p>
    </fieldset>
  );
}
