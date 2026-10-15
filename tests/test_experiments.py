from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.data import save_dataset
from trading_bot.experimental_models import (ExperimentOptions, experimental_signals,
                                              fit_candidate, sequence_rows)
from trading_bot.experiments import run_experiments
from trading_bot.features import FEATURES, build_features
from trading_bot.news import add_sentiment, fetch_news, load_news
from trading_bot.strategy import signal


def test_sequence_windows_are_causal_and_do_not_cross_symbols(bars):
    features = build_features(bars)
    rows, windows = sequence_rows(features, 5)
    for _, row in rows.iloc[::40].iterrows():
        history = features.loc[(features.symbol == row.symbol) & (features.date <= row.date)].tail(5)
        np.testing.assert_allclose(windows[int(row._window)], history[FEATURES], rtol=1e-6)


@pytest.mark.parametrize('kind', ['ml', 'ml_sentiment', 'lstm', 'transformer', 'q_learning'])
def test_experimental_fits_cannot_see_future(kind, bars, config):
    if kind in {'lstm', 'transformer'}:
        pytest.importorskip('torch')
    features = build_features(bars)
    features['news_sentiment'], features['news_count'] = .1, 1
    frame, windows = sequence_rows(features, 5)
    cutoff = pd.Timestamp(bars.date.unique()[200])
    options = ExperimentOptions(lookback=5, epochs=1, rl_episodes=2, seeds=(42,))
    model = fit_candidate(kind, frame, windows, cutoff, config, options)
    changed = frame.copy()
    changed.loc[changed.label_end >= cutoff, 'target'] = 99
    changed.loc[changed.date >= cutoff, FEATURES + ['news_sentiment']] = 999
    later_windows = windows.copy()
    later_windows[changed.loc[changed.date >= cutoff, '_window'].to_numpy(dtype=int)] = 999
    later = fit_candidate(kind, changed, later_windows, cutoff, config, options)
    assert model.version == later.version
    assert pd.Timestamp(model.audit['last_label_end']) < cutoff
    sample = frame.loc[frame.date < cutoff].tail(6)
    # CPU float32 kernels may differ at rounding scale even with seeded deterministic algorithms.
    np.testing.assert_allclose(model.predict(sample, windows), later.predict(sample, windows), rtol=1e-4, atol=1e-7)


def test_models_share_dates_and_test_models_are_frozen(bars, config):
    frame, windows = sequence_rows(build_features(bars), 5)
    options = ExperimentOptions(lookback=5, epochs=1, rl_episodes=2, seeds=(42,))
    result, audit = experimental_signals(frame, windows, config, options, ['ml', 'q_learning'])
    keys = result['ml'][['date', 'symbol', 'period']]
    for kind, rows in result.items():
        pd.testing.assert_frame_equal(keys, rows[['date', 'symbol', 'period']])
        if kind in {'ml', 'q_learning'}:
            assert rows.loc[rows.period == 'test', 'model_version'].nunique() == 1
    assert all(pd.Timestamp(m['last_label_end']) < pd.Timestamp(m['cutoff']) for m in audit['models'])


def article(id, created, updated=None, headline='positive'):
    return {'id': id, 'created_at': created, 'updated_at': updated or created,
            'headline': headline, 'symbols': ['AAPL']}


def test_news_respects_updates_latency_early_closes_and_weekends():
    # July 3 closes at 17:00 UTC; July 4 is closed.
    dates = pd.to_datetime(['2024-07-03', '2024-07-05', '2024-07-08'])
    frame = pd.DataFrame({'date': dates, 'symbol': 'AAPL'})
    stories = [article(1, '2024-07-03T16:40:00Z'),
               article(2, '2024-07-03T16:50:00Z'),
               article(3, '2024-07-03T12:00:00Z', '2024-07-06T12:00:00Z'),
               article(4, '2024-07-08T20:00:00Z')]
    scored = add_sentiment(frame, stories, scorer=lambda text: {'compound': .5})
    assert list(scored.news_count) == [1, 1, 1]
    assert list(scored.news_sentiment) == [.5, .5, .5]
    earlier = add_sentiment(frame, stories[:2], scorer=lambda text: {'compound': .5})
    pd.testing.assert_frame_equal(scored.iloc[:2], earlier.iloc[:2])


def test_news_no_articles_and_neutral_article_distinct():
    frame = pd.DataFrame({'date': pd.to_datetime(['2024-01-02', '2024-01-03']), 'symbol': 'AAPL'})
    result = add_sentiment(frame, [article(1, '2024-01-03T12:00:00Z')], lambda text: {'compound': 0})
    assert list(result.news_count) == [0, 1]
    assert list(result.news_sentiment) == [0, 0]


def test_holiday_news_is_available_for_first_session():
    frame = pd.DataFrame({'date': pd.to_datetime(['2024-01-02']), 'symbol': 'AAPL'})
    result = add_sentiment(frame, [article(1, '2024-01-01T12:00:00Z')], lambda text: {'compound': .2})
    assert result.news_count.iloc[0] == 1


class FakeNews:
    def data_get(self, path, params):
        assert path == '/v1beta1/news'
        if 'page_token' in params:
            return {'news': [article(2, '2024-01-03T12:00:00Z')], 'next_page_token': None}
        return {'news': [article(1, '2024-01-02T12:00:00Z')], 'next_page_token': 'next'}


def test_news_snapshot_pagination_coverage_and_integrity(tmp_path):
    path = tmp_path / 'news'
    metadata = fetch_news(FakeNews(), path, ['AAPL'], '2024-01-01', '2024-01-03')
    assert metadata['pages'] == metadata['articles'] == 2
    articles, _ = load_news(path, ['AAPL'], pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03'))
    assert len(articles) == 2
    with pytest.raises(ValueError, match='entire price interval'):
        load_news(path, ['AAPL'], pd.Timestamp('2023-12-29'), pd.Timestamp('2024-01-03'))
    with pytest.raises(ValueError, match='already exists'):
        fetch_news(FakeNews(), path, ['AAPL'], '2024-01-01', '2024-01-03')
    (path / 'articles.json').write_text('[]')
    with pytest.raises(ValueError, match='integrity'):
        load_news(path, ['AAPL'], pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03'))


def test_news_versioned_route_remains_read_only():
    from trading_bot.broker import AlpacaPaper, BrokerError
    client = AlpacaPaper('fake', 'fake')
    with pytest.raises(BrokerError):
        client._request('POST', '/v1beta1/news', data=True)
    with pytest.raises(BrokerError):
        client._request('GET', '/v1beta1/news', data=False)


def test_rl_uses_actual_position(config):
    row = {'rl_cash_action': 0, 'rl_long_action': 1}
    assert not signal('q_learning', row, config, held=False).eligible
    assert signal('q_learning', row, config, held=True).eligible


def test_experiment_artifacts_and_dashboard(tmp_path, bars, config, monkeypatch):
    from streamlit.testing.v1 import AppTest
    dataset, output = tmp_path / 'data', tmp_path / 'runs' / 'experiment'
    save_dataset(dataset, bars, bars, {'source': 'synthetic_demo'})
    summary = run_experiments(dataset, output, config,
        ExperimentOptions(lookback=5, epochs=1, rl_episodes=2, seeds=(42, 43)), kinds=['ml', 'q_learning'])
    assert len(summary) == 2 * 2 * 4
    assert (output / 'comparisons.csv').exists()
    assert not (output / 'RUNNING').exists()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert summary.groupby(['seed', 'period']).sessions.nunique().eq(1).all()
    dashboard = Path(__file__).resolve().parents[1] / 'dashboard.py'
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    assert not app.exception
    app.sidebar.selectbox[1].set_value(43).run()
    assert not app.exception
    app.sidebar.selectbox[2].set_value('q_learning').run()
    assert not app.exception


def test_full_suite_with_news_and_neural_models(tmp_path, bars, config):
    pytest.importorskip('torch')
    pytest.importorskip('vaderSentiment')
    dataset, news, output = tmp_path / 'prices', tmp_path / 'news', tmp_path / 'experiment'
    save_dataset(dataset, bars, bars, {'source': 'synthetic_demo'})
    fetch_news(FakeNews(), news, config.symbols, str(bars.date.min().date()), str(bars.date.max().date()))
    summary = run_experiments(dataset, output, replace(config, retrain_sessions=252),
        ExperimentOptions(lookback=5, epochs=1, rl_episodes=2, seeds=(42,)), news=news)
    assert len(summary) == 14
    assert set(summary.strategy) == {'ml', 'ml_sentiment', 'lstm', 'transformer', 'q_learning', 'technical', 'buy_hold'}
    predictions = pd.read_csv(output / 'prediction_metrics.csv')
    assert len(predictions) == 8
    assert np.isfinite(predictions[['mae', 'rmse', 'direction_accuracy']]).all().all()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['news_coverage']['AAPL']['articles'] == 2
    assert len(list((output / 'seed-42' / 'models').glob('*.pt'))) == 4
