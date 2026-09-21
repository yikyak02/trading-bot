from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3

import pandas as pd

from . import __version__
from .backtest import run_backtest
from .data import load_dataset
from .features import build_features
from .models import walk_forward


def run_research(dataset, output, config):
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists. Choose a new run directory to preserve the experiment.")
    bars, metadata = load_dataset(dataset, config.symbols)
    features = build_features(bars)
    signals, audit = walk_forward(features, config)
    output.mkdir(parents=True, exist_ok=False)
    signals.to_parquet(output / "signals.parquet", index=False)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "version": __version__,
                "dataset": metadata, "config": config.as_dict(), "evaluation": audit,
                "environment": {"python": platform.python_version(), **{
                    pkg: importlib.metadata.version(pkg) for pkg in ("pandas", "numpy", "scikit-learn", "exchange-calendars")}},
                "assumptions": ["Adjusted OHLC total-return proxy; no separate dividend cash credits.",
                    "Signals at close; next-session open execution; constant per-fill costs.",
                    "Active strategies share risk rules; passive benchmark is fully invested without stops.",
                    "Entry allocation caps may drift during holding; no daily rebalancing.",
                    "Each evaluation period starts with fresh cash; open positions marked at last close."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary, stress = [], []
    with sqlite3.connect(output / "audit.sqlite") as db:
        for period in ("validation", "test"):
            subset = signals.loc[signals.period == period]
            for strategy in ("buy_hold", "technical", "ml"):
                result = run_backtest(bars, subset, strategy, config)
                folder = output / period / strategy
                folder.mkdir(parents=True)
                for name in ("equity", "trades", "decisions", "holdings"):
                    table = getattr(result, name)
                    table.to_csv(folder / f"{name}.csv", index=False)
                    table.assign(period=period, strategy=strategy).to_sql(name, db, if_exists="append", index=False)
                (folder / "metrics.json").write_text(json.dumps(result.metrics, indent=2, allow_nan=False))
                summary.append({"period": period, "strategy": strategy, **result.metrics})
                if period == "test":
                    higher_costs = replace(config, fee_bps=config.fee_bps * 2,
                                           slippage_bps=config.slippage_bps * 2, spread_bps=config.spread_bps * 2)
                    stressed = run_backtest(bars, subset, strategy, higher_costs)
                    stress.append({"strategy": strategy, "cost_multiplier": 2, **stressed.metrics})
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(stress).to_csv(output / "cost_stress.csv", index=False)
    return pd.DataFrame(summary)
