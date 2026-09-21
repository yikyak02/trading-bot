import numpy as np
import pandas as pd
import pytest

from trading_bot.config import Config
from trading_bot.data import BAR_COLUMNS, save_dataset, sessions


@pytest.fixture
def config():
    return Config(initial_cash=10000, fee_bps=0, slippage_bps=0, spread_bps=0, min_train_sessions=30)


def price_frame(start="2023-01-03", end="2024-03-08", seed=1):
    dates = sessions(start, end)
    rng = np.random.default_rng(seed)
    rows = []
    for j, symbol in enumerate(Config().symbols):
        for i, date in enumerate(dates):
            price = 100 + j * 25 + i * 0.2 + np.sin(i / 10)
            rows.append([date, symbol, price, price * 1.01, price * .99,
                         price * 1.001, 1_000_000 + int(rng.integers(10000))])
    return pd.DataFrame(rows, columns=BAR_COLUMNS).sort_values(["date", "symbol"]).reset_index(drop=True)


@pytest.fixture
def bars():
    return price_frame()


@pytest.fixture
def paper_dataset(tmp_path, bars):
    directory = tmp_path / "prices"
    save_dataset(directory, bars, bars, {"source": "alpaca", "feed": "iex"})
    return directory
