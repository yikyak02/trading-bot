"""Small sequence regressors and a tabular Q-learning research baseline."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .features import FEATURES
from .news import NEWS_FEATURES


@dataclass(frozen=True)
class ExperimentOptions:
    lookback: int = 20
    epochs: int = 8
    hidden: int = 16
    batch_size: int = 128
    learning_rate: float = 0.001
    rl_episodes: int = 30
    seeds: tuple = (42, 43, 44)

    def __post_init__(self):
        for name in ('lookback', 'epochs', 'hidden', 'batch_size', 'rl_episodes'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.hidden % 2:
            raise ValueError('hidden must be even for two-head attention.')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite.')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(
                not isinstance(s, int) or not 0 <= s < 2**32 for s in self.seeds):
            raise ValueError('Provide distinct integer seeds in [0, 2**32).')


def sequence_rows(features, lookback):
    """Windows never cross symbols; all inputs end at the signal close."""
    samples, rows = [], []
    for _, group in features.groupby('symbol', sort=True):
        group = group.sort_values('date')
        values = group[FEATURES].to_numpy(dtype=np.float32)
        for end in range(lookback - 1, len(group)):
            window = values[end - lookback + 1:end + 1]
            if np.isfinite(window).all():
                samples.append(window)
                rows.append(group.iloc[end].to_dict())
    if not rows:
        raise ValueError('Insufficient complete history for sequence windows.')
    frame = pd.DataFrame(rows)
    frame['_window'] = np.arange(len(frame))
    return frame.sort_values(['date', 'symbol']).reset_index(drop=True), np.stack(samples)


def split_dates(frame, config):
    dates = pd.DatetimeIndex(sorted(frame.date.unique()))
    counts = frame.groupby('date').symbol.nunique().reindex(dates)
    if (counts != len(config.symbols)).any():
        raise ValueError('Require complete feature windows for every symbol/session.')
    train_end = int(len(dates) * config.train_fraction)
    test_start = int(len(dates) * (config.train_fraction + config.validation_fraction))
    if train_end < config.min_train_sessions + 2 or test_start >= len(dates) - 2:
        raise ValueError('Dataset too short for experimental splits.')
    return dates, train_end, test_start


def torch_module(kind, features, options):
    import torch
    from torch import nn

    class SequenceRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == 'lstm':
                self.encoder = nn.LSTM(features, options.hidden, batch_first=True)
            else:
                self.project = nn.Linear(features, options.hidden)
                self.position = nn.Parameter(torch.zeros(1, options.lookback, options.hidden))
                layer = nn.TransformerEncoderLayer(options.hidden, nhead=2,
                    dim_feedforward=options.hidden * 2, dropout=0.0, batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            self.head = nn.Linear(options.hidden, 1)

        def forward(self, x):
            if kind == 'lstm':
                encoded, _ = self.encoder(x)
            else:
                # All tokens are historical observations, not future prediction targets.
                encoded = self.encoder(self.project(x) + self.position)
            return self.head(encoded[:, -1]).squeeze(-1)
    return SequenceRegressor()


class Candidate:
    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed

    def predict(self, rows, windows):
        if self.kind == 'q_learning':
            states = self.states(rows)
            return np.stack([self.q[states, h].argmax(axis=1) for h in (0, 1)], axis=1)
        if self.kind in ('ml', 'ml_sentiment'):
            return self.model.predict(rows[self.columns])
        import torch
        self.model.eval()
        x = (windows[rows._window.to_numpy(dtype=int)] - self.mean) / self.scale
        with torch.no_grad():
            chunks = [self.model(torch.from_numpy(x[i:i + 512])).numpy() for i in range(0, len(x), 512)]
        return np.concatenate(chunks) * self.y_scale + self.y_mean

    def states(self, rows):
        x = rows[self.columns].to_numpy()
        bins = (x > self.edges[0]).astype(int) + (x > self.edges[1]).astype(int)
        return (bins * (3 ** np.arange(len(self.columns)))).sum(axis=1)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f'{self.version}.json').write_text(json.dumps(self.audit, indent=2))
        if self.kind == 'q_learning':
            np.savez(folder / f'{self.version}.npz', q=self.q, edges=self.edges, columns=self.columns)
        elif self.kind in ('lstm', 'transformer'):
            import torch
            torch.save(self.model.state_dict(), folder / f'{self.version}.pt')
            np.savez(folder / f'{self.version}.npz', mean=self.mean, scale=self.scale,
                     y_mean=self.y_mean, y_scale=self.y_scale)
        else:
            import pickle
            # Local training artifact only; never load untrusted pickle files.
            with (folder / f'{self.version}.pkl').open('wb') as out:
                pickle.dump(self.model, out)


def fit_candidate(kind, frame, windows, cutoff, config, options):
    cutoff = pd.Timestamp(cutoff)
    train = frame.loc[(frame.date < cutoff) & (frame.label_end < cutoff)].dropna(subset=['target'])
    if train.date.nunique() < config.min_train_sessions:
        raise ValueError('Insufficient purged training data.')
    result = Candidate(kind, config.seed)
    result.columns = FEATURES + (NEWS_FEATURES if kind == 'ml_sentiment' else [])
    x = windows[train._window.to_numpy(dtype=int)]
    digest = hashlib.sha256(pd.util.hash_pandas_object(
        train[['date', 'symbol', *result.columns, 'target']], index=False).values.tobytes()
        + x.tobytes() + json.dumps({'options': asdict(options), 'config': config.as_dict(), 'kind': kind},
                                  sort_keys=True).encode()).hexdigest()[:12]
    result.version = f'{kind}-{config.seed}-{digest}'
    result.audit = {'version': result.version, 'kind': kind, 'seed': config.seed,
                    'cutoff': str(cutoff.date()), 'last_label_end': str(train.label_end.max().date()),
                    'training_rows': len(train), 'features': result.columns, 'options': asdict(options)}
    if kind in ('ml', 'ml_sentiment'):
        result.model = GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=.05,
            min_samples_leaf=20, random_state=config.seed, loss='huber').fit(train[result.columns], train.target)
    elif kind == 'q_learning':
        result.columns = ['return_5', 'sma_ratio', 'distance_sma50', 'rsi']
        result.edges = np.quantile(train[result.columns], [1/3, 2/3], axis=0)
        result.q = np.zeros((3 ** len(result.columns), 2, 2))
        rng = np.random.default_rng(config.seed)
        cost = (config.fee_bps + config.slippage_bps + config.spread_bps / 2) / 10000
        trajectories = [(result.states(g), g.target.to_numpy())
                        for _, g in train.groupby('symbol', sort=True)]
        for episode in range(options.rl_episodes):
            epsilon = max(.05, .4 * (1 - episode / options.rl_episodes))
            for states, rewards in trajectories:
                held = 0
                for i, state in enumerate(states):
                    action = int(rng.integers(2)) if rng.random() < epsilon else int(result.q[state, held].argmax())
                    reward = action * rewards[i] - cost * abs(action - held)
                    continuation = .95 * result.q[states[i+1], action].max() if i + 1 < len(states) else 0.0
                    result.q[state, held, action] += .1 * (reward + continuation - result.q[state, held, action])
                    held = action
        result.audit.update(features=result.columns, reward='next open-to-open return minus estimated switching cost',
            discount=.95, alpha=.1, state='training-quantile bins plus cash/long position',
            limitation='Training is a per-stock proxy without stops/allocation; evaluation uses the shared portfolio risk overlay.')
    elif kind in ('lstm', 'transformer'):
        try:
            import torch
        except ImportError:
            raise ValueError('Install research extras: .venv/bin/python -m pip install -e ".[research]"') from None
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        result.mean = x.mean(axis=(0, 1), keepdims=True)
        result.scale = np.maximum(x.std(axis=(0, 1), keepdims=True), 1e-6)
        result.y_mean = float(train.target.mean())
        result.y_scale = max(float(train.target.std()), 1e-6)
        inputs = torch.from_numpy((x - result.mean) / result.scale)
        targets = torch.tensor((train.target.to_numpy() - result.y_mean) / result.y_scale, dtype=torch.float32)
        result.model = torch_module(kind, len(FEATURES), options)
        optimizer = torch.optim.Adam(result.model.parameters(), lr=options.learning_rate)
        generator = torch.Generator().manual_seed(config.seed)
        losses = []
        for _ in range(options.epochs):
            result.model.train()
            order = torch.randperm(len(train), generator=generator)
            total = 0.0
            for ids in order.split(options.batch_size):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(result.model(inputs[ids]), targets[ids])
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite sequence-model training loss.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(result.model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(ids)
            losses.append(total / len(train))
        result.audit['training_loss'] = losses
    else:
        raise ValueError(f'Unknown experimental model: {kind}')
    return result


def experimental_signals(frame, windows, config, options, kinds, model_dir=None):
    dates, train_end, test_start = split_dates(frame, config)
    fits = sorted(set(range(train_end, test_start, config.retrain_sessions)) | {test_start})
    outputs, audits = {}, []
    for kind in kinds:
        pieces = []
        for j, i in enumerate(fits):
            stop = fits[j+1] if j + 1 < len(fits) else len(dates)
            model = fit_candidate(kind, frame, windows, dates[i], config, options)
            if model_dir is not None:
                model.save(model_dir)
            audits.append(model.audit)
            subset = frame.loc[frame.date.isin(dates[i:stop])].copy()
            prediction = model.predict(subset, windows)
            if not np.isfinite(prediction).all():
                raise ValueError('Non-finite experimental model predictions.')
            if kind == 'q_learning':
                subset['rl_cash_action'], subset['rl_long_action'] = prediction[:, 0], prediction[:, 1]
                subset['prediction'] = np.nan  # Policy actions are not return forecasts.
            else:
                subset['prediction'] = prediction
            subset['model_version'] = model.version
            subset['model_cutoff'] = str(dates[i].date())
            subset['sensitivity'] = '[]'
            subset['period'] = 'validation' if i < test_start else 'test'
            pieces.append(subset)
        outputs[kind] = pd.concat(pieces, ignore_index=True)
        print(f'Seed {config.seed}: trained {kind} ({len(fits)} chronological fits).', flush=True)
    reference = next(iter(outputs.values())).copy()
    reference['prediction'], reference['model_version'], reference['sensitivity'] = np.nan, '', '[]'
    outputs['buy_hold'] = reference.copy()
    outputs['technical'] = reference.copy()
    return outputs, {'models': audits, 'validation_signal_start': str(dates[train_end].date()),
                     'test_signal_start': str(dates[test_start].date()),
                     'policy': 'Shared sequence-ready sessions; expanding validation; purged labels; frozen final-test fits.'}
