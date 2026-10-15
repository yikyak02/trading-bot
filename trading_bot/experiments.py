"""Comparable, multi-seed historical experiments; no broker order execution."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3

import numpy as np
import pandas as pd

from . import __version__
from .backtest import run_backtest
from .data import load_dataset
from .experimental_models import ExperimentOptions, experimental_signals, sequence_rows
from .features import FEATURES, build_features
from .news import add_sentiment, load_news


def run_experiments(dataset, output, config, options=None, news=None, kinds=None):
    options = options or ExperimentOptions()
    kinds = list(kinds or ['ml', 'lstm', 'transformer', 'q_learning'])
    if news and 'ml_sentiment' not in kinds:
        kinds.append('ml_sentiment')
    if 'ml' not in kinds:
        kinds.insert(0, 'ml')  # Always retain a matched supervised baseline.
    if len(set(kinds)) != len(kinds) or set(kinds) - {'ml', 'ml_sentiment', 'lstm', 'transformer', 'q_learning'}:
        raise ValueError('Unknown or repeated experimental model.')
    if 'ml_sentiment' in kinds and news is None:
        raise ValueError('ml_sentiment requires a historical --news snapshot.')
    for package in (['torch'] if set(kinds) & {'lstm', 'transformer'} else []) + (['vaderSentiment'] if news else []):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError('Install research extras: .venv/bin/python -m pip install -e ".[research]"') from None
    output = Path(output)
    if output.exists():
        raise ValueError('Output already exists. Choose a new experiment directory.')
    bars, metadata = load_dataset(dataset, config.symbols)
    features = build_features(bars)
    # Reject broken feature rows instead of silently shortening the experiment.
    valid_dates = pd.DatetimeIndex(sorted(features.date.unique()))[49:]
    if not np.isfinite(features.loc[features.date.isin(valid_dates), FEATURES].to_numpy()).all():
        raise ValueError('Require complete finite features after indicator warmup.')
    news_metadata = None
    if news:
        articles, news_metadata = load_news(news, config.symbols, bars.date.min(), bars.date.max())
        features = add_sentiment(features, articles)
        if not features.news_count.sum():
            raise ValueError('No news is available before any signal close in this interval.')
    frame, windows = sequence_rows(features, options.lookback)
    output.mkdir(parents=True)
    (output / 'RUNNING').write_text('Incomplete until this marker is removed. Do not interpret partial results.\n')
    features.to_parquet(output / 'features.parquet', index=False)
    summaries, stress_rows, forecast_rows, evaluations = [], [], [], {}
    for seed in options.seeds:
        seeded = replace(config, seed=seed)
        seed_dir = output / f'seed-{seed}'
        seed_dir.mkdir()
        signals, audit = experimental_signals(frame, windows, seeded, options, kinds, seed_dir / 'models')
        evaluations[str(seed)] = audit
        pd.concat([s.assign(strategy=k) for k, s in signals.items()], ignore_index=True).to_parquet(
            seed_dir / 'signals.parquet', index=False)
        seed_summary, seed_stress = [], []
        with sqlite3.connect(seed_dir / 'audit.sqlite') as db:
            for period in ('validation', 'test'):
                for strategy, source in signals.items():
                    subset = source.loc[source.period == period]
                    result = run_backtest(bars, subset, strategy, seeded)
                    folder = seed_dir / period / strategy
                    folder.mkdir(parents=True)
                    for name in ('equity', 'trades', 'decisions', 'holdings'):
                        table = getattr(result, name)
                        table.to_csv(folder / f'{name}.csv', index=False)
                        table.assign(period=period, strategy=strategy).to_sql(name, db, if_exists='append', index=False)
                    (folder / 'metrics.json').write_text(json.dumps(result.metrics, indent=2, allow_nan=False))
                    seed_summary.append({'seed': seed, 'period': period, 'strategy': strategy, **result.metrics})
                    if period == 'test':
                        expensive = replace(seeded, fee_bps=seeded.fee_bps*2,
                            slippage_bps=seeded.slippage_bps*2, spread_bps=seeded.spread_bps*2)
                        stress = run_backtest(bars, subset, strategy, expensive)
                        seed_stress.append({'seed': seed, 'strategy': strategy, 'cost_multiplier': 2, **stress.metrics})
                    if strategy in {'ml', 'ml_sentiment', 'lstm', 'transformer'}:
                        observed = subset.dropna(subset=['target', 'prediction'])
                        # Validation diagnostics cannot consume targets ending in final test.
                        if period == 'validation':
                            observed = observed.loc[observed.label_end < pd.Timestamp(audit['test_signal_start'])]
                        errors = observed.prediction - observed.target
                        forecast_rows.append({'seed': seed, 'period': period, 'strategy': strategy,
                            'rows': len(observed), 'mae': float(errors.abs().mean()),
                            'rmse': float(np.sqrt((errors**2).mean())),
                            'direction_accuracy': float(((observed.prediction > 0) == (observed.target > 0)).mean())})
        pd.DataFrame(seed_summary).to_csv(seed_dir / 'summary.csv', index=False)
        pd.DataFrame(seed_stress).to_csv(seed_dir / 'cost_stress.csv', index=False)
        summaries.extend(seed_summary)
        stress_rows.extend(seed_stress)
        print(f'Seed {seed}: backtests and doubled-cost stress tests complete.', flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(stress_rows).to_csv(output / 'cost_stress.csv', index=False)
    pd.DataFrame(forecast_rows).to_csv(output / 'prediction_metrics.csv', index=False)
    metric_columns = ['total_return', 'sharpe_ratio', 'max_drawdown', 'transaction_costs', 'average_exposure']
    aggregate = summary.groupby(['period', 'strategy'])[metric_columns].agg(['mean', 'std'])
    aggregate.columns = ['_'.join(c) for c in aggregate.columns]
    aggregate.reset_index().to_csv(output / 'aggregate.csv', index=False)
    comparisons = []
    pairs = [('lstm', 'technical'), ('transformer', 'technical'), ('lstm', 'ml'), ('transformer', 'ml'),
             ('ml_sentiment', 'ml'), ('q_learning', 'ml')]
    for period in ('validation', 'test'):
        part = summary.loc[summary.period == period]
        for candidate, baseline in pairs:
            if candidate not in part.strategy.values:
                continue
            left = part.loc[part.strategy == candidate].set_index('seed')
            right = part.loc[part.strategy == baseline].set_index('seed')
            delta = left.total_return - right.total_return
            comparisons.append({'period': period, 'candidate': candidate, 'baseline': baseline,
                'seeds': len(delta), 'return_difference_mean': float(delta.mean()),
                'return_difference_std': float(delta.std()) if len(delta) > 1 else None,
                'seeds_with_higher_return': int((delta > 0).sum()),
                'sharpe_difference_mean': float((left.sharpe_ratio - right.sharpe_ratio).mean())})
    pd.DataFrame(comparisons).to_csv(output / 'comparisons.csv', index=False)
    coverage = None
    if news:
        coverage = features.groupby('symbol').agg(articles=('news_count', 'sum'),
            sessions_with_news=('news_count', lambda s: int((s > 0).sum())), sessions=('date', 'size')).to_dict('index')
    packages = ['pandas', 'numpy', 'scikit-learn', 'exchange-calendars']
    if set(kinds) & {'lstm', 'transformer'}:
        packages.append('torch')
    if news:
        packages.append('vaderSentiment')
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'version': __version__,
        'experiment': 'advanced_comparison', 'status': 'complete', 'dataset': metadata,
        'config': config.as_dict(), 'options': asdict(options), 'strategies': list(signals),
        'news': news_metadata, 'news_coverage': coverage, 'evaluation': evaluations,
        'environment': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in packages}},
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
        'assumptions': [
            'Research only; no new models are enabled for paper order submission.',
            'All candidates use identical sequence-ready sessions, splits, capital and execution costs.',
            'Features/scalers/bins fit on purged training data; final-test model frozen per seed.',
            'Neural hyperparameters fixed in advance; no early stopping or tuning on final test.',
            'Active strategies share stops, allocation and drawdown rules; passive benchmark differs.',
            'Q-learning training uses a single-asset open-to-open reward proxy; shared risk overlay only at evaluation.',
            'VADER is a general-language headline baseline, not a finance-specific language model.',
            'News available at max(created_at, updated_at) + 15 minutes; archived revisions are not original text vintages.',
            'A zero news count means no retrieved available article, not proof that no news existed.',
            'Seeds measure optimization variation, not independent market samples or statistical significance.',
            '2020–2025 was already inspected; treat these results as exploratory, not a fresh confirmatory test.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'RUNNING').unlink()
    return summary
