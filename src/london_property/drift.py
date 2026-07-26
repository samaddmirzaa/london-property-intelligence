import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

project_root = Path(__file__).resolve().parents[2]
reference_path = project_root / 'data' / 'reference' / 'training_reference.parquet'
new_data_path = project_root / 'data' / 'drift' / 'london_2025.parquet'
postcode_path = project_root / 'data' / 'supplementary' / 'postcodes_london.parquet'

NUMERIC = ['price', 'lat', 'lon', 'zone', 'imd_index', 'avg_income', 'dist_station']
CATEGORICAL = ['propertytype', 'duration']

TARGET = 'price'
PSI_THRESHOLD = 0.2
DRIFT_SHARE_THRESHOLD = 0.3
SEVERE_PSI = 0.5


def psi_numeric(reference, current, bins=10):
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(current) == 0 or len(reference) == 0:
        return np.nan

    edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.clip(np.histogram(reference, bins=edges)[0] / len(reference), 1e-6, None)
    cur_pct = np.clip(np.histogram(current, bins=edges)[0] / len(current), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_categorical(reference, current):
    categories = sorted(set(reference.dropna()) | set(current.dropna()))
    ref_pct = np.clip([(reference == c).mean() for c in categories], 1e-6, None)
    cur_pct = np.clip([(current == c).mean() for c in categories], 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift(reference, current):
    scores = {}
    for column in NUMERIC:
        scores[column] = psi_numeric(reference[column].values.astype(float),
                                     current[column].values.astype(float))
    for column in CATEGORICAL:
        scores[column] = psi_categorical(reference[column], current[column])
    return scores


def decide_retrain(scores):
    drifted = [k for k, v in scores.items() if v > PSI_THRESHOLD]
    share = len(drifted) / len(scores)
    worst = max(scores.values())

    if scores[TARGET] > PSI_THRESHOLD:
        return True, f'target ({TARGET}) drifted: PSI {scores[TARGET]:.3f}'
    if share > DRIFT_SHARE_THRESHOLD:
        return True, f'{len(drifted)}/{len(scores)} features drifted'
    if worst > SEVERE_PSI:
        return True, f'severe drift in a single feature: PSI {worst:.3f}'
    return False, 'drift within acceptable thresholds'


def main():
    reference = pd.read_parquet(reference_path)
    current = pd.read_parquet(new_data_path)
    postcodes = pd.read_parquet(postcode_path)
    current = current.merge(postcodes, on='postcode', how='left')

    scores = compute_drift(reference, current)
    retrain, reason = decide_retrain(scores)

    print(f'Reference: {len(reference):,} rows | Current: {len(current):,} rows\n')
    for column, value in sorted(scores.items(), key=lambda x: -x[1]):
        status = 'DRIFTED' if value > PSI_THRESHOLD else 'stable'
        print(f'  {column:14s} PSI={value:.3f}  {status}')

    print(f'\nDecision: {"RETRAIN" if retrain else "NO RETRAIN"} - {reason}')

    report = {'scores': scores, 'retrain': retrain, 'reason': reason}
    Path(project_root / 'drift_report.json').write_text(json.dumps(report, indent=2))

    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f'retrain={str(retrain).lower()}\n')
            f.write(f'reason={reason}\n')

    return retrain


if __name__ == '__main__':
    main()

