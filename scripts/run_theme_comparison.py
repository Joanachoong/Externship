"""Run the Glassdoor comparison and save local tables without rerunning old cells."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for name, suffix in [('HF_HOME', 'huggingface'), ('NUMBA_CACHE_DIR', 'numba'),
                     ('MPLCONFIGDIR', 'matplotlib')]:
    os.environ.setdefault(name, str(ROOT / '.cache' / suffix))
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
from text_engine.model_comparison import compare_theme_models

if __name__ == '__main__':
    data = pd.read_csv(ROOT / 'data/raw/glass_reviews.csv')
    data.rename(columns={data.columns[0]: 'review_id'}, inplace=True)
    result = compare_theme_models(data)
    destination = ROOT / '.cache/theme_comparison'
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            # JSON preserves list/dict labels and full model scores for remapping.
            value.to_json(destination / f'{name}.json', orient='records', indent=2)
            value.to_csv(destination / f'{name}.csv', index=False)
    (destination / 'metadata.json').write_text(json.dumps(result['metadata'], indent=2))
    print(result['metadata'])
    print(result['coverage'].to_string(index=False))
    print('Mean Jaccard on nonempty label unions:', result['agreement'].jaccard.mean())
    print('Saved outputs:', destination)
