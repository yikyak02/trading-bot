import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from trading_bot.data import fetch_dataset, load_dataset, save_dataset
from trading_bot.research import run_research


def test_research_artifacts_and_dashboard(tmp_path, bars, config, monkeypatch):
    dataset = tmp_path / "data"
    save_dataset(dataset, bars, bars, {"source": "synthetic_demo"})
    output = tmp_path / "runs" / "smoke"
    summary = run_research(dataset, output, config)
    assert len(summary) == 6
    assert len(pd.read_csv(output / "cost_stress.csv")) == 3
    assert (output / "audit.sqlite").exists()
    for period in ("validation", "test"):
        starts, ends = [], []
        for strategy in ("buy_hold", "technical", "ml"):
            curve = pd.read_csv(output / period / strategy / "equity.csv")
            starts.append(curve.date.min())
            ends.append(curve.date.max())
        assert len(set(starts)) == len(set(ends)) == 1
    with pytest.raises(ValueError, match="already exists"):
        run_research(dataset, output, config)
    dashboard = Path(__file__).resolve().parents[1] / "dashboard.py"
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 4
    assert "Synthetic demo" in app.warning[0].value
    app.sidebar.selectbox[1].set_value("technical").run()
    app.sidebar.radio[0].set_value("validation").run()
    assert not app.exception
    app.multiselect[0].set_value([]).run()
    assert not app.exception


def test_dashboard_empty_state(tmp_path, monkeypatch):
    dashboard = Path(__file__).resolve().parents[1] / "dashboard.py"
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(dashboard)).run()
    assert not app.exception
    assert "first experiment" in app.info[0].value


class PaginatedData:
    def __init__(self):
        self.calls = []

    def data_get(self, path, params):
        self.calls.append((path, dict(params)))
        days = [3, 4] if "page_token" in params else [2]
        return {"bars": [{"t": f"2024-01-0{day}T05:00:00Z", "o": 100, "h": 102,
                          "l": 99, "c": 101, "v": 1000} for day in days],
                "next_page_token": None if "page_token" in params else "page-2"}


def test_fetch_paginates_and_discards_unfinished_session(tmp_path, config):
    client = PaginatedData()
    path = tmp_path / "data"
    fetch_dataset(client, path, config.symbols, "2024-01-02", "2024-01-04",
                  now=pd.Timestamp("2024-01-04T20:30:00Z"))
    bars, metadata = load_dataset(path)
    assert bars.date.max() == pd.Timestamp("2024-01-03")
    assert len(client.calls) == 12  # 3 symbols x 2 price bases x 2 pages
    assert metadata["source"] == "alpaca"
    assert (path / "provider_responses.json").exists()


def test_fetch_requires_complete_requested_range(tmp_path, config):
    with pytest.raises(ValueError, match="entire requested"):
        fetch_dataset(PaginatedData(), tmp_path / "data", config.symbols, "2023-12-29", "2024-01-04",
                      now=pd.Timestamp("2024-01-05T12:00:00Z"))
