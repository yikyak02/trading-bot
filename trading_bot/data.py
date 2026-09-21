"""Validated, versioned daily bars; adjusted research prices, raw execution prices."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close"]
BAR_COLUMNS = ["date", "symbol", *PRICE_COLUMNS, "volume"]


def calendar(start, end):
    # Pad calendar construction so weekend/holiday request boundaries stay in bounds.
    return xcals.get_calendar("XNYS", start=str((pd.Timestamp(start) - pd.Timedelta(days=10)).date()),
                              end=str((pd.Timestamp(end) + pd.Timedelta(days=10)).date()))


def sessions(start, end):
    result = calendar(start, end).sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return result.tz_localize(None).normalize()


def validate_bars(frame: pd.DataFrame, symbols=None) -> pd.DataFrame:
    missing = set(BAR_COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Missing or empty OHLCV data: {sorted(missing)}")
    df = frame[BAR_COLUMNS].copy()
    df["date"] = pd.to_datetime(df.date, errors="raise").dt.tz_localize(None).dt.normalize()
    if df.date.isna().any() or df.symbol.isna().any():
        raise ValueError("Missing date or symbol.")
    if df.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate symbol/session rows.")
    numeric = PRICE_COLUMNS + ["volume"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise ValueError("OHLCV contains non-finite values.")
    if (df[PRICE_COLUMNS] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative.")
    if ((df.high < df[["open", "close", "low"]].max(axis=1)) |
            (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC price ordering.")
    expected_symbols = set(symbols or df.symbol.unique())
    if set(df.symbol.unique()) != expected_symbols:
        raise ValueError("Dataset symbols do not match the configured universe.")
    expected = sessions(df.date.min(), df.date.max())
    for symbol in sorted(expected_symbols):
        actual = pd.DatetimeIndex(df.loc[df.symbol == symbol, "date"]).sort_values()
        if not actual.equals(expected):
            gaps = expected.difference(actual).strftime("%Y-%m-%d").tolist()[:5]
            raise ValueError(f"{symbol}: missing sessions {gaps} or non-session bars. No forward filling.")
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def save_dataset(directory, adjusted, raw, metadata, responses=None):
    directory = Path(directory)
    adjusted, raw = validate_bars(adjusted), validate_bars(raw)
    if not adjusted[["date", "symbol"]].equals(raw[["date", "symbol"]]):
        raise ValueError("Adjusted and raw bars must cover identical sessions and symbols.")
    directory.mkdir(parents=True, exist_ok=False)
    adjusted.to_parquet(directory / "bars.parquet", index=False)
    raw.to_parquet(directory / "raw.parquet", index=False)
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in ("bars.parquet", "raw.parquet")}
    metadata = {
        **metadata, "created_at": datetime.now(timezone.utc).isoformat(),
        "adjustment": "all", "price_basis": "adjusted_total_return_proxy",
        "start": str(adjusted.date.min().date()), "end": str(adjusted.date.max().date()),
        "symbols": sorted(adjusted.symbol.unique()), "rows": len(adjusted), "sha256": hashes,
    }
    if responses is not None:
        (directory / "provider_responses.json").write_text(json.dumps(responses))
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def load_dataset(directory, symbols=None):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("adjustment") != "all":
        raise ValueError("Research requires consistently split/dividend-adjusted OHLC data.")
    for name in ("bars.parquet", "raw.parquet"):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != metadata["sha256"][name]:
            raise ValueError(f"Dataset integrity check failed: {name}")
    return validate_bars(pd.read_parquet(directory / "bars.parquet"), symbols), metadata


def demo_dataset(directory, symbols, seed=42):
    """Artificial correlated prices. Never represented as historical stock performance."""
    rng = np.random.default_rng(seed)
    dates = sessions("2018-01-01", "2025-12-31")
    market = rng.normal(0.0002, 0.010, len(dates))
    market[500:650] -= 0.002
    rows = []
    for j, symbol in enumerate(symbols):
        previous = 70.0 + 40 * j
        for i, date in enumerate(dates):
            opening = previous * np.exp(rng.normal(0, 0.004))
            close = opening * np.exp(market[i] + rng.normal(0, 0.007))
            high = max(opening, close) * (1 + abs(rng.normal(0, 0.006)))
            low = min(opening, close) * (1 - min(abs(rng.normal(0, 0.006)), 0.1))
            rows.append([date, symbol, opening, high, low, close, int(rng.lognormal(15, 0.4))])
            previous = close
    frame = pd.DataFrame(rows, columns=BAR_COLUMNS)
    return save_dataset(directory, frame, frame, {"source": "synthetic_demo", "seed": seed,
                         "warning": "Artificial prices, not actual market history."})


def fetch_dataset(client, directory, symbols, start, end, feed="iex", now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    requested = sessions(start, end)
    cal = calendar(start, end)
    completed = [d for d in requested if cal.session_close(d) + pd.Timedelta(minutes=15) < now]
    if not completed:
        raise ValueError("No completed sessions in the requested interval.")
    first, last = completed[0], completed[-1]
    responses, frames = [], {}
    for adjustment in ("all", "raw"):
        rows = []
        for symbol in symbols:
            token = None
            seen = set()
            while True:
                params = {"timeframe": "1Day", "start": str(first.date()),
                          "end": (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                          "adjustment": adjustment, "feed": feed, "limit": 10000, "sort": "asc"}
                if token:
                    params["page_token"] = token
                response = client.data_get(f"/v2/stocks/{symbol}/bars", params)
                responses.append({"symbol": symbol, "adjustment": adjustment, "response": response})
                for bar in response.get("bars") or []:
                    date = pd.Timestamp(bar["t"]).tz_convert("America/New_York").tz_localize(None).normalize()
                    if first <= date <= last:
                        rows.append([date, symbol, bar["o"], bar["h"], bar["l"], bar["c"], bar["v"]])
                token = response.get("next_page_token")
                if not token:
                    break
                if token in seen:
                    raise ValueError("Provider repeated a pagination token.")
                seen.add(token)
        frames[adjustment] = validate_bars(pd.DataFrame(rows, columns=BAR_COLUMNS), symbols)
        if frames[adjustment].date.min() != first or frames[adjustment].date.max() != last:
            raise ValueError("Provider did not return the entire requested completed-session interval.")
    return save_dataset(directory, frames["all"], frames["raw"],
                        {"source": "alpaca", "feed": feed}, responses)
