from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest import run_backtest
from trading_bot.data import BAR_COLUMNS, load_dataset, save_dataset, sessions, validate_bars
from trading_bot.features import FEATURES, build_features
from trading_bot.models import fit_model, walk_forward
from trading_bot.strategy import position_budget


def small_market(config, opening=(100, 100, 110), low=None, eligibility=(1, -1)):
    dates = sessions("2024-01-02", "2024-01-04")
    rows = []
    for i, d in enumerate(dates):
        for s in config.symbols:
            price = opening[i]
            rows.append([d, s, price, price + 1, low[i] if low else price - 1, price, 1000])
    bars = pd.DataFrame(rows, columns=BAR_COLUMNS)
    signals = pd.DataFrame([{"date": dates[i], "symbol": "AAPL", **{f: 1.0 for f in FEATURES},
                             "sma_ratio": eligibility[i]} for i in range(2)])
    return bars, signals


def test_features_cannot_see_future(bars):
    cutoff = bars.date.unique()[150]
    original = build_features(bars)
    changed = bars.copy()
    changed.loc[changed.date > cutoff, ["open", "high", "low", "close", "volume"]] *= 4
    recomputed = build_features(changed)
    pd.testing.assert_frame_equal(original.loc[original.date <= cutoff, FEATURES],
                                  recomputed.loc[recomputed.date <= cutoff, FEATURES])


def test_model_purges_unknown_labels_and_is_stable(bars, config):
    cutoff = bars.date.unique()[170]
    original = fit_model(build_features(bars), cutoff, config)
    changed = bars.copy()
    changed.loc[changed.date >= cutoff, ["open", "high", "low", "close"]] *= 3
    later = fit_model(build_features(changed), cutoff, config)
    assert original.version == later.version
    assert pd.Timestamp(original.last_label_end) < cutoff


def test_final_test_model_frozen_and_chronological(bars, config):
    predictions, audit = walk_forward(build_features(bars), config)
    test = predictions[predictions.period == "test"]
    assert test.model_version.nunique() == 1
    assert all(pd.Timestamp(x["last_label_end"]) < pd.Timestamp(x["cutoff"]) for x in audit["models"])
    assert predictions[predictions.period == "validation"].date.max() < test.date.min()


def test_next_open_fill_and_accounting(config):
    bars, signals = small_market(config)
    result = run_backtest(bars, signals, "technical", config)
    assert list(result.trades.side) == ["buy", "sell"]
    assert result.trades.iloc[0].date > result.trades.iloc[0].signal_date
    assert result.trades.iloc[0].quantity == pytest.approx(20)
    assert result.equity.iloc[-1].equity == pytest.approx(10200)
    assert result.trades.iloc[-1].realized_pnl == pytest.approx(200)
    assert result.metrics["win_rate"] == 1


def test_gap_stop_uses_worse_open_and_no_reentry(config):
    bars, signals = small_market(config, opening=(100, 100, 80), eligibility=(1, 1))
    result = run_backtest(bars, signals, "technical", config)
    assert len(result.trades) == 2
    assert result.trades.iloc[-1].fill_price == 80
    assert result.equity.iloc[-1].equity == pytest.approx(9600)
    assert result.metrics["max_drawdown"] == pytest.approx(-.04)


def test_same_day_entry_can_stop(config):
    bars, signals = small_market(config, low=(99, 90, 109))
    result = run_backtest(bars, signals, "technical", config)
    assert result.trades.iloc[-1].fill_price == 95
    assert result.trades.iloc[0].date == result.trades.iloc[-1].date
    assert result.equity.iloc[-1].equity == pytest.approx(9900)


def test_costs_are_charged_both_sides(config):
    config = replace(config, fee_bps=10, slippage_bps=10, spread_bps=4)
    bars, signals = small_market(config)
    result = run_backtest(bars, signals, "technical", config)
    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    assert buy.fill_price == pytest.approx(100 * 1.0012)
    assert sell.fill_price == pytest.approx(110 * .9988)
    assert result.metrics["transaction_costs"] == pytest.approx(result.trades.fee.sum() + result.trades.impact_cost.sum())
    assert result.equity.iloc[-1].equity == pytest.approx(config.initial_cash + sell.realized_pnl)
    assert (result.equity.cash >= 0).all()


def test_drawdown_latches(config):
    config = replace(config, max_drawdown=.03)
    bars, signals = small_market(config, opening=(100, 100, 80), eligibility=(1, 1))
    result = run_backtest(bars, signals, "technical", config)
    assert result.equity.iloc[-1].halted


def test_risk_budget_and_position_cap(config):
    budget, rule, _ = position_budget(10000, 10000, 3, .05, config)
    assert budget == pytest.approx(2000)
    assert rule == "stop_risk_budget"
    budget, rule, _ = position_budget(10000, 10000, 3, .01, config)
    assert budget == pytest.approx(3000)
    assert rule == "position_cap"


def test_no_trades_metrics_are_defined(config):
    bars, signals = small_market(config, eligibility=(-1, -1))
    result = run_backtest(bars, signals, "technical", config)
    assert result.metrics["total_return"] == 0
    assert result.metrics["sharpe_ratio"] is None
    assert result.metrics["win_rate"] is None


def test_initial_buy_cost_included_in_drawdown(config):
    bars, signals = small_market(config, opening=(100, 100, 100), eligibility=(1, 1))
    result = run_backtest(bars, signals, "technical", replace(config, fee_bps=100))
    assert result.metrics["max_drawdown"] < 0


def test_missing_and_duplicate_bars_rejected(bars):
    with pytest.raises(ValueError, match="missing sessions"):
        validate_bars(bars.drop(index=60))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_bars(pd.concat([bars, bars.iloc[:1]]))


def test_invalid_prices_rejected(bars):
    broken = bars.copy()
    broken.loc[0, "close"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_bars(broken)


def test_market_holidays_and_dst():
    assert len(sessions("2024-07-04", "2024-07-05")) == 1
    from trading_bot.data import calendar
    cal = calendar("2024-03-08", "2024-03-11")
    assert cal.session_open("2024-03-08").hour == 14
    assert cal.session_open("2024-03-11").hour == 13


def test_dataset_immutable_and_hash_checked(tmp_path, bars):
    path = tmp_path / "dataset"
    save_dataset(path, bars, bars, {"source": "synthetic_demo"})
    loaded, _ = load_dataset(path)
    assert len(loaded) == len(bars)
    with pytest.raises(FileExistsError):
        save_dataset(path, bars, bars, {})
    with open(path / "bars.parquet", "ab") as file:
        file.write(b"tampering")
    with pytest.raises(ValueError, match="integrity"):
        load_dataset(path)


@pytest.mark.parametrize("kwargs", [{"max_drawdown": 0}, {"initial_cash": -1}, {"symbols": ["AAPL"]},
                                    {"fee_bps": float("nan")}, {"train_fraction": .9, "validation_fraction": .2}])
def test_invalid_config_rejected(kwargs):
    from trading_bot.config import Config
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_incomplete_features_are_not_silently_dropped(bars, config):
    features = build_features(bars)
    features.loc[features.index[200], "volume_ratio"] = np.nan
    with pytest.raises(ValueError, match="complete features"):
        walk_forward(features, config)
