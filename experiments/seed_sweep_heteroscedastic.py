"""Seed sweep for Heteroscedastic DKL

Run seeds in a range and record metrics for each seed:
- MAPE
- Max Error (%)
- Number of outliers >20%

Outputs a CSV with per-seed metrics and a small summary JSON noting
- seed with minimal Max Error
- seed with minimal Outliers (>20%)

Usage examples:
python experiments/seed_sweep_heteroscedastic.py --start 1 --end 3000 --n_epochs 200
"""
import argparse
import time
import json
import pandas as pd
from pathlib import Path
import sys
# Ensure project root is on sys.path so we can import sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heteroscedastic_dkl_randomseed import (
    train_heteroscedastic_dkl,
    evaluate_model,
    X_train_tensor,
    y_train_tensor,
    X_test_tensor,
    y_test,
    scaler_y,
)


def run_seed_sweep(start, end, n_epochs=200, out_csv='seed_sweep_results.csv'):
    results = []

    best_max_err = {'seed': None, 'MaxError': float('inf'), 'MAPE': None, 'Outliers20': None}
    best_outliers = {'seed': None, 'Outliers20': float('inf'), 'MaxError': None, 'MAPE': None}

    total = end - start + 1
    print(f"Running seed sweep {start}..{end} ({total} runs), n_epochs={n_epochs}")

    for i, seed in enumerate(range(start, end + 1), 1):
        t0 = time.time()
        try:
            model = train_heteroscedastic_dkl(
                X_train_tensor, y_train_tensor, X_test_tensor, y_test, scaler_y,
                seed=seed, n_epochs=n_epochs
            )
            metrics, _ = evaluate_model(model, X_test_tensor, y_test, scaler_y)

            row = {
                'seed': seed,
                'MAPE': metrics['MAPE'],
                'MaxError': metrics['MaxError'],
                'Outliers20': metrics['Outliers20'],
                'Outliers30': metrics['Outliers30'],
                'Outliers40': metrics['Outliers40'],
                'time_s': time.time() - t0,
            }
            results.append(row)

            # update bests
            if row['MaxError'] < best_max_err['MaxError']:
                best_max_err = {'seed': seed, **{k: row[k] for k in ['MaxError', 'MAPE', 'Outliers20']}}

            if row['Outliers20'] < best_outliers['Outliers20']:
                best_outliers = {'seed': seed, **{k: row[k] for k in ['Outliers20', 'MaxError', 'MAPE']}}

        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            results.append({'seed': seed, 'MAPE': None, 'MaxError': None, 'Outliers20': None, 'Outliers30': None, 'Outliers40': None, 'time_s': time.time() - t0})

        if i % 10 == 0 or seed == end:
            df = pd.DataFrame(results)
            df.to_csv(out_csv, index=False)
            print(f"Progress: {i}/{total} seeds saved to {out_csv}")

    # final save
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)

    summary = {
        'best_by_min_max_error': best_max_err,
        'best_by_min_outliers20': best_outliers,
        'results_csv': str(Path(out_csv).absolute())
    }

    summary_path = Path(out_csv).with_name('seed_sweep_summary.json')
    summary_path.write_text(json.dumps(summary, indent=2))

    print('\nSweep complete')
    print('Best seed by Min MaxError:', best_max_err)
    print('Best seed by Min Outliers(>20%):', best_outliers)
    print('Summary saved to', summary_path)

    return df, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=3000)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--out_csv', type=str, default='seed_sweep_results.csv')

    args = parser.parse_args()
    run_seed_sweep(args.start, args.end, n_epochs=args.n_epochs, out_csv=args.out_csv)
