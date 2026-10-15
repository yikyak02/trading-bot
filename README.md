# Explainable AI Stock Trading Bot

Download daily US stock prices, test trading strategies on historical data, and inspect the results in a dashboard. The example uses **AAPL, MSFT, and JPM**, with long-only positions and no leverage.

The normal research workflow is:

```text
Alpaca historical data → fetch → data/<dataset>/
                                     ↓
                                  backtest → runs/<experiment>/ → dashboard
```

**Downloading data and backtesting do not place orders.** Optional paper trading is a separate command that uses simulated money in Alpaca. Live trading is not supported.

## Where to start

| What you want to do | Where to go |
|---|---|
| Install the project and save your API keys once | [One-time setup](#one-time-setup) |
| Download real prices and test the strategies | [Run a historical experiment](#run-a-historical-experiment) |
| Understand `.env`, `.venv`, `data/`, and `runs/` | [Understand the files](#understand-the-files) |
| Try the project without an Alpaca account | [Optional offline demo](#optional-offline-demo) |
| Change stocks, strategy settings, or risk limits | [Strategies and configuration](#strategies-and-configuration) |
| Compare LSTM, Transformer, news sentiment, and reinforcement learning | [Research experiments](#research-experiments) |
| Understand performance reports | [Results and metrics](#results-and-metrics) |
| Send simulated orders to Alpaca | [Optional paper trading](#optional-paper-trading) |

**Already set up on this machine:** the SIP dataset is in `data/stocks-2020-2025/`, and the completed backtest is in `runs/stocks-01/`. You can go straight to [opening the dashboard](#3-open-the-dashboard). The download and backtest examples below describe how those folders were created; do not rerun them with the same destination paths.

## One-time setup

### 1. Install Python dependencies

Requires Python **3.11+** and macOS/Linux/WSL. Paper-cycle locking uses POSIX `fcntl`.

From the project folder, run the following only if the environment is not already installed:

```bash
cd "/Users/lucas/TRADING BOT"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
```

`.venv` holds this project's Python interpreter and packages. Dependencies are declared in `pyproject.toml`; `requirements-lock.txt` records the verified environment. On another machine, replace the project path with your own.

### 2. Save your Alpaca paper credentials

Log in to the [Alpaca dashboard](https://app.alpaca.markets/account/login), select your **Paper Trading** account, and generate a key pair in the API Keys panel.

If `.env` does not exist yet, create it once:

```bash
cp -n .env.example .env
chmod 600 .env
```

Open `.env` in your editor and replace the placeholders with your own values:

```bash
APCA_API_KEY_ID='your-paper-key'
APCA_API_SECRET_KEY='your-paper-secret'
```

Keep this file private. It is excluded from Git; never commit credentials. If your keys are already saved, leave them in place.

### 3. Use the launcher for future commands

Run commands from the project folder using `./run-bot`. It loads `.env` and uses `.venv` automatically, including after a computer restart. You do not need to activate the virtual environment or repeat `export` commands.

The launcher reads `.env` as trusted shell syntax. Its values override existing shell variables for the bot process. Direct `python -m trading_bot` or `stockbot` commands do **not** automatically load `.env`; they require the correct Python environment and exported credentials.

## Run a historical experiment

### 1. Download the data

```bash
./run-bot fetch \
  --start 2020-01-01 --end 2025-12-31 \
  --dataset data/stocks-2020-2025 --config config/default.json
```

This asks Alpaca for daily prices and volume for the stocks in the configuration, validates them, and saves the dataset locally. It downloads both adjusted and raw prices using the configured SIP feed.

| Argument | Meaning |
|---|---|
| `--start` / `--end` | Inclusive date range; only completed exchange sessions are included. |
| `--dataset` | New folder in which to save the downloaded data. |
| `--config` | JSON file defining the stocks, data feed, and strategy/risk settings. |

Download once and reuse that dataset for multiple experiments. An existing snapshot is never overwritten; use a new dataset folder for a fresh download.

### 2. Backtest the strategies

```bash
./run-bot backtest \
  --dataset data/stocks-2020-2025 \
  --output runs/stocks-01 --config config/default.json
```

This reads the saved prices and simulates **buy-and-hold**, a **technical strategy**, and a **machine-learning strategy**, including configured trading costs. It saves reports in the new folder supplied with `--output`.

The history is divided chronologically into training, validation, and final-test periods. Training teaches the model; validation measures it while it is periodically retrained; the final test evaluates a frozen model on later data. See [evaluation timing](#evaluation-timing-and-simulation-details) for the exact rules.

For another experiment on the same data, choose a new output folder:

```bash
./run-bot backtest \
  --dataset data/stocks-2020-2025 \
  --output runs/stocks-02 --config config/default.json
```

Use an unused name each time. Keep old results so you can compare experiments.

### 3. Open the dashboard

```bash
.venv/bin/python -m streamlit run dashboard.py
```

Open the local URL printed in your terminal, usually `http://localhost:8501`. In the sidebar, select **stocks-01**, an evaluation period, and a strategy.

| Dashboard tab | What it shows |
|---|---|
| Performance | Portfolio value, drawdown, and strategy comparisons. |
| Decision journal | Buy/sell/hold reasons and simulated fills. |
| Risk & costs | Cost stress results, holdings, and risk settings. |
| Paper account | Last recorded paper-account snapshot, if available. |
| Experiment details | Configuration, dataset provenance, and run assumptions. |

The dashboard is read-only and cannot submit orders. Stop it with **Ctrl+C**. For a quick table without the dashboard, open `runs/stocks-01/summary.csv`.

## Understand the files

| File or folder | What it contains |
|---|---|
| `.venv/` | Python and installed packages, not API credentials. |
| `.env` | Your private, persistent Alpaca credentials. |
| `.env.example` | Placeholder names for creating `.env`; no real keys. |
| `run-bot` | Launcher that loads credentials and starts the bot. |
| `config/default.json` | Stocks, data feed, starting capital, and strategy/risk settings. |
| `data/<dataset>/` | Downloaded market data used as input. |
| `runs/<experiment>/` | Backtest reports produced as output. |
| `state/paper.sqlite` | Persistent order/recovery state for optional paper trading. |

### Adjusted prices versus raw prices

Each downloaded dataset contains:

| File | Contents and use |
|---|---|
| `bars.parquet` | Prices adjusted with Alpaca's `adjustment=all`; used for research and backtesting. |
| `raw.parquet` | Unadjusted daily prices, preserving the historical quoted price basis. |
| `metadata.json` | Source, feed, date range, symbols, and file hashes used for integrity checks. |
| `provider_responses.json` | Original provider responses from the successful download. |

Both price files cover the same stocks and sessions. A daily **bar** contains open, high, low, close (OHLC), and volume. **Parquet** is an efficient table-storage format, similar in purpose to CSV.

After a 4-for-1 split, a $400 stock becomes roughly $100 without a 75% economic loss. Adjusted history rescales earlier prices so the strategy does not mistake the split for a crash. Dividend adjustments are also reflected in the research prices; the simulator does not add dividend payments again.

## Common questions and errors

| Situation | What to do |
|---|---|
| “Missing sessions … No forward filling” | The dataset is incomplete. Inspect provider responses or query those dates, and verify the feed and each stock's listed history. Do not fill gaps with invented prices. |
| Dataset or experiment folder already exists | Reuse the dataset for a backtest, or choose a new destination for a download/run. Existing results are protected from overwriting. |
| Alpaca returns HTTP 401 or 403 | Check the paper keys in `.env` and your account's data permissions. |
| Unsure whether to repeat setup | Saved `.env` credentials persist. Use `./run-bot` from the project folder. |
| Changed stocks in the configuration | Use a dataset matching those stocks and a common listed date range. Download to a new folder if needed. |
| Want to inspect results without trading | Use `backtest` and the dashboard. Only the separate `paper --submit` command sends paper orders. |

For this account's 2020–2025 download, IEX was missing three JPM sessions. Historical SIP access succeeded and the full SIP dataset passed validation. `config/default.json` therefore uses `sip`; commands without `--config` still use the built-in `iex` default.

## Optional offline demo

No Alpaca account is required. This uses **artificial prices**, not real stock history:

```bash
./run-bot demo
.venv/bin/python -m streamlit run dashboard.py
```

The demo writes `data/demo/` and `runs/demo/`. If `runs/demo/` already exists, use an unused output folder:

```bash
./run-bot demo --output runs/demo-02
```

The existing demo dataset is reused and integrity-checked. For a different universe or seed, choose a new `--dataset` directory too. Artificial demo results do not measure actual stock performance.

## Strategies and configuration

Default example stocks are `AAPL`, `MSFT`, and `JPM`. They are an illustrative universe, not investment recommendations. Select your universe before inspecting final-test performance.

| Component | Behavior |
|---|---|
| Buy-and-hold | Buy equal weights once at the first evaluation open, then hold. Include entry costs. No stop or drawdown halt. |
| Technical | Enter/hold when SMA20 > SMA50 and close > SMA50; otherwise exit or stay in cash. |
| ML | Gradient-boosted regressor forecasts the next executable open-to-open return; enter/hold when forecast exceeds 0.2%. |
| Entry allocation | Minimum of equal-weight budget, 30% of equity, stop-risk budget, and available cash. |
| Risk budget | Estimated loss to the initial stop capped at 1% of equity before costs/gaps. A 5% stop normally limits allocation to 20%. |
| Stop | Fixed 5% default; optional ATR multiple, clipped to a 1–50% distance. Set at entry, not trailing. |
| Drawdown | A 15% observed drawdown latches a halt. Liquidate at the next executable open, or at that open if first detected there. |
| Costs | Per fill: 5 bps fee + 5 bps slippage + half of a 2 bps spread. |

Both active strategies share risk rules. Buy-and-hold remains a conventional fully invested benchmark, so its exposure differs. Reports include average exposure. Position caps apply **at entry**; weights may drift during holding. There is no daily resizing, leverage, or shorting.

Edit a copy of `config/default.json` and pass `--config` to any command. Stops cannot guarantee a maximum loss during gaps. Changing parameters after seeing final-test results makes that period exploratory; reserve new unseen data for a fresh test.

## Research experiments

The `experiment` command compares these research questions on the same trading sessions and portfolio simulator:

| Question | Comparison |
|---|---|
| Do sequence models outperform technical indicators? | Small PyTorch LSTM and Transformer versus the technical strategy and gradient-boosted model. |
| Does news sentiment help? | The same gradient-boosted model with price features alone versus price features plus daily headline sentiment and article count. |
| How does reinforcement learning compare? | A tabular Q-learning cash/long policy versus supervised return prediction. |

These are offline experiments. The additional models are not enabled for paper-order submission. Implementing a model does not establish that it performs better.

### Install the optional research packages once

```bash
.venv/bin/python -m pip install -e '.[research]'
```

This adds PyTorch and VADER sentiment analysis. `requirements-research-lock.txt` records the verified research environment, including the base dependencies.

### Download historical headlines once

Use a news range covering the entire stock dataset:

```bash
./run-bot fetch-news \
  --start 2020-01-01 --end 2025-12-31 \
  --output data/news-2020-2025 --config config/default.json
```

The snapshot includes original paginated responses, deduplicated articles, and integrity hashes. A failed download without `metadata.json` is incomplete and cannot be used. Choose a new destination if a snapshot already exists.

Headlines are scored with [VADER](https://github.com/cjhutto/vaderSentiment), a general-language lexicon baseline. An article becomes available **15 minutes after the later of its creation and update timestamps**, then contributes to the first eligible exchange-session close. After-close and holiday/weekend news therefore enters a later signal. Each article is used once. No retrieved news gives sentiment zero and count zero; a neutral headline has a positive count.

Alpaca archives can contain revised text, not the original article version. Using the update timestamp avoids backdating the retrieved revision but cannot reconstruct perfect historical availability. Coverage counts are saved in the manifest; absent news is not proof that no news existed. The source is [Alpaca's historical news API](https://docs.alpaca.markets/us/reference/news-3).

### Run all three comparisons

```bash
./run-bot experiment \
  --dataset data/stocks-2020-2025 \
  --news data/news-2020-2025 \
  --output runs/research-01 --config config/default.json
```

By default this trains models with seeds 42, 43, and 44, a 20-session lookback, and 8 neural training epochs. It runs seven strategies: `buy_hold`, `technical`, `ml`, `lstm`, `transformer`, `q_learning`, and `ml_sentiment`. Expect more time than an ordinary backtest. Progress is printed after each model and seed.

Omit `--news` to run the price-only and reinforcement-learning comparisons. To run a smaller development check, use a new output directory with `--seeds 42 --epochs 1`. This checks the workflow, not model quality. `--models` selects a subset; the supervised `ml` baseline is always included, along with technical and buy-and-hold benchmarks.

### Read the comparison

Open the usual dashboard and select the research experiment. Choose a training seed in the sidebar, or expand **Comparison across seeds**.

| Output | What to look for |
|---|---|
| `comparisons.csv` | Paired return differences versus each question's baseline and the number of seeds with higher returns. |
| `aggregate.csv` | Mean and standard deviation of performance across seeds. |
| `prediction_metrics.csv` | MAE, RMSE, and direction accuracy for supervised forecasts; policy actions are not treated as return predictions. |
| `summary.csv` | All strategy, seed, and evaluation-period results. |
| `seed-<number>/` | Signals, fitted model artifacts, individual backtests, doubled-cost stress results, and audit database. |
| `manifest.json` | Model settings, split dates, training cutoffs, source hashes, dependency versions, and news coverage. |

An experiment with a `RUNNING` marker is incomplete and hidden from the dashboard. Completed outputs are never overwritten. Serialized model files are local audit artifacts; do not load pickle files from untrusted sources.

### How the comparisons stay consistent

- LSTM and Transformer use sequences of the existing ten price/volume indicators. All strategies start on the same sequence-ready sessions, which begin later than the ordinary backtest's warmup. Compare strategies within this experiment rather than directly against `stocks-01`.
- All models use purged chronological training labels. Scalers and Q-learning state bins are fit only on training data. Validation refits every configured 21 sessions; the final-test fit stays frozen. Neural hyperparameters are fixed, with no test-set tuning or early stopping.
- VADER sentiment and article count are added only to `ml_sentiment`, producing a matched price-versus-news comparison with `ml`.
- Q-learning uses four indicators binned by training quantiles plus cash/long position state. It trains for 30 passes with discounted open-to-open returns minus estimated switching costs. This is a basic tabular baseline, not deep reinforcement learning or a complete trading environment.
- Q-learning's training proxy omits portfolio allocation and stops. At evaluation, its decisions use actual simulated holdings and the same allocation, stop, drawdown, and cost rules as the other active strategies. This training/evaluation difference is recorded in the manifest.
- Each period starts with fresh cash. Buy-and-hold remains a fully invested benchmark without active risk controls. Cost stress reuses trained policies/signals; it does not retrain them.

Compare returns together with drawdown, exposure, trading costs, prediction errors, and variation across seeds. Seeds are repeated optimization runs on the same market history, **not independent evidence of statistical significance**. Because the 2020–2025 results have already been inspected, these comparisons are exploratory. Reserve unseen dates before making a claim that a model consistently outperforms.

## Results and metrics

```text
runs/<run>/
  manifest.json           # Config, source hashes, environment, fitting cutoffs, assumptions
  signals.parquet         # Features, predictions, model versions, sensitivity
  summary.csv             # All strategies, validation and final test
  cost_stress.csv         # Final-test results with all execution costs doubled
  audit.sqlite            # Equity, trades, decisions, holdings
  validation/<strategy>/  # equity/trades/decisions/holdings CSVs and metrics.json
  test/<strategy>/        # Same artifacts for the frozen-model final test
```

The dashboard compares equity and drawdown, filters decision explanations, shows costs and holdings, displays recorded paper-account state, and exposes experiment provenance. It cannot submit orders.

- **Total return:** ending equity / starting equity − 1, including costs.
- **Annualized return:** total growth compounded to 252 trading sessions.
- **Sharpe:** mean daily excess return / daily sample standard deviation × √252. Annual risk-free rate defaults to zero and is converted to daily. Undefined with constant returns.
- **Maximum drawdown:** most negative equity / running peak − 1, including starting capital before the first fill.
- **Volatility:** daily sample standard deviation × √252.
- **Win rate:** fraction of fully closed trades with positive P&L after entry/exit costs; undefined when none close.
- **Transaction costs:** fees plus modeled slippage and half-spread. Already embedded in equity, not subtracted again.
- **Turnover:** total traded notional / average equity over the period; not annualized.
- **Average exposure:** average invested fraction at session closes.

Journals include buys, sells, holds, rejected entries, feature values, reasons, and model versions. Actual simulated fill quantities/prices are in the trade ledger. The cost stress test reuses the same signals, not retrained models.

## Data validation and price assumptions

`fetch` downloads raw and `adjustment=all` daily bars, follows pagination, and saves original provider responses. Dates are inclusive; a session is excluded until at least 15 minutes after its official close. The exchange calendar handles holidays, early closes, and daylight-saving changes. Daily labels represent exchange sessions, not UTC-midnight trading days.

Validation rejects missing sessions, duplicates, invalid OHLC ranges, non-finite values, incomplete symbol coverage, and changed dataset files. There is no forward filling. Choose a common listed history for all selected stocks. Downloads create new immutable snapshots rather than incrementally overwriting files.

`config/default.json` uses `sip`, verified with this account for the 2020–2025 download. Without a config file, the built-in default remains `iex`. IEX covers one exchange, so its prices and volume differ from consolidated data; its JPM history was missing sessions on 2021-04-19, 2021-10-25, and 2022-03-08. Verify data entitlements and historical coverage with your account. References: [Alpaca data sources](https://docs.alpaca.markets/us/docs/historical-stock-data-1), [historical bar parameters](https://docs.alpaca.markets/us/reference/stockbarsingle-1).

### Corporate actions

Research uses **adjusted OHLC as a total-return proxy** for every strategy. Splits and dividends are represented by the provider's adjusted history; the simulator does **not** credit dividends again. Quantities are fractional adjusted units, not reconstructed historical exchange share counts.

This is a consistent comparison basis, not an exact corporate-action cash ledger. Dividend payment timing, taxes, actual share rounding, and actual stop adjustment on ex-dates are not modeled. Historical stops/fills in adjusted prices are approximations. Raw bars are preserved separately; paper orders use current unadjusted broker quotes and positions. Do not compare adjusted simulated fill prices directly with historical exchange orders.

## Evaluation timing and simulation details

```text
Market data → Validation → Features → Model / technical signal
                                                ↓
                                    Portfolio and risk rules
                                                ↓
                                    Backtest / paper execution
                                                ↓
                                     Audit logs and dashboard
```

1. Features use completed candles through close of session `t`; simulated orders execute at open of `t+1`.
2. The prediction target is `open(t+2) / open(t+1) - 1`.
3. Training excludes labels whose ending session is on or after the fitting cutoff. Assets are split together by session, never randomly by row.
4. After indicator warmup, usable sessions are divided into 60% initial training, 20% validation, and 20% final test.
5. Validation uses expanding-window refits every 21 sessions. The final-test model is fitted once before that period and frozen. Model parameters and threshold are fixed; there is no test-set optimization.
6. Each evaluation period starts with fresh cash and empty holdings. Strategies share execution dates and capital. A final signal without a following bar is not executed.

Features: lagged returns, SMA ratios, simple 14-session RSI, normalized MACD, ATR, volatility, and relative volume. Local explanations show prediction changes after replacing one feature with its training median. These are **model sensitivity measurements, not causal explanations or SHAP values**.

Simulation order: opening gap stops → scheduled exits → entries → intraday stops → closing valuation. Entry-day stops are possible. Gaps below a stop fill at the worse opening price, minus impact. No same-day reentry follows a stop. Daily candles cannot reproduce intraday paths, liquidity, partial fills, or queue position.

Open positions at the end are marked to the final close, not force-liquidated. Unrealized P&L contributes to returns but not closed-trade win rate; no hypothetical exit fee is charged. Cash earns zero interest.

## Optional paper trading

The trading host is hardcoded to `https://paper-api.alpaca.markets`; there is no live endpoint setting. Authentication is not saved in audit logs.

Use an **empty dedicated paper account** on the first submitted cycle and one strategy/config per state database. State binds account ID, strategy, and config. Preserve it across restarts. Do not mix manual orders or another bot into the account.

Before each session, fetch a fresh dataset covering several years and ending on the latest completed US session. These dates are examples; update the date and directory each trading day:

```bash
./run-bot fetch \
  --start 2020-01-01 --end 2026-09-17 \
  --dataset data/paper-through-2026-09-17 --config config/default.json

# Run in the first 30 minutes after the regular US market open.
# Preview only: no orders placed or canceled.
./run-bot paper \
  --dataset data/paper-through-2026-09-17 \
  --config config/default.json --strategy technical --state state/paper.sqlite

# Explicitly submit PAPER orders.
./run-bot paper \
  --dataset data/paper-through-2026-09-17 \
  --config config/default.json --strategy technical --state state/paper.sqlite --submit
```

For ML use `--strategy ml` with a separate empty paper account/state. Paper ML refits on available history each cycle; this deployment policy differs from the frozen final-test model. The latest bar must exactly match the prior exchange session. Synthetic data, stale quotes, unsupported assets, short/out-of-universe positions, and unrecognized orders are rejected.

### Execution and recovery

- The command runs **one cycle**, not a daemon. Schedule externally in `America/New_York` during the first 30 minutes of the regular session. DST matters when scheduling from another timezone. Closed sessions are rejected; no scheduler is installed automatically.
- Entries are whole-share market OTO orders with a broker-held stop leg. Stops use the fresh raw bid and shared stop-distance policy; quantity uses the ask with a cost allowance. See [Alpaca OTO orders](https://docs.alpaca.markets/us/docs/orders-at-alpaca).
- Market fills can move beyond the quote and intended allocation. Exact opening fills, fractional adjusted simulation units, and simulated fees will differ from the broker.
- Before a signal/drawdown exit, cancel protective orders, confirm cancellation, then reread positions. If a stop fills during cancellation, stop the cycle and reconcile before another order.
- Order IDs are deterministic per account/session/strategy/symbol/side. SQLite saves the explanation and intent **before** the POST, including when the response is lost. POSTs are never retried automatically. Restarts reconcile existing orders instead of duplicating them.
- An uncertain network outcome blocks execution until the existing order is found. If absent at the broker, inspect the account and ledger before manual repair. **Do not delete the state file or change IDs to bypass this protection.**
- Pending non-stop orders block new entries. A follow-up cycle within the entry window reconciles fills. Snapshots represent account state before the current cycle's orders settle.
- Broker stops persist without the local process. Their activation follows parent-order fill rules; protection during partial fills is not guaranteed. Retained positions are checked for stop coverage on the next cycle.
- Drawdown is observed at cycle times, not continuously. High-water mark and halt persist; no automatic halt reset exists. Keep paper capital unchanged: deposits/withdrawals are not performance-adjusted.
- Preview makes no broker mutations but may create local tables and update reconciled statuses. The dashboard displays recorded state, not a streaming broker view.

API connectivity must be verified with your own credentials. Automated tests mock the broker; the demo and tests need no real orders or account.

## Development and tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q trading_bot dashboard.py
```

Tests cover future-data invariance, purged labels, frozen test models, next-open fills, hand-calculated accounting, costs, gap/intraday stops, drawdown, no-trade metrics, holiday/DST behavior, data integrity, paper-only transport, stale inputs, restart idempotency, and uncertain network outcomes. Dashboard smoke tests use Streamlit's application test runner.

## Code layout

```text
trading_bot/
  config.py      # Validated settings
  data.py        # Calendar, data checks, snapshots, demo, historical API
  features.py    # Indicators and executable-horizon labels
  models.py      # Gradient boosting, purged fitting, walk-forward, sensitivity
  experimental_models.py # LSTM, Transformer, Q-learning, shared research splits
  experiments.py # Multi-seed comparisons and research artifacts
  news.py        # Historical headlines and timestamp-aligned sentiment
  strategy.py    # Shared signals, stop distance, position sizing
  backtest.py    # Event simulation, accounting, metrics
  research.py    # Experiments, cost stress, reproducibility artifacts
  broker.py      # Paper-only API and confirmed cancellation
  paper.py       # Durable order ledger, reconciliation, execution cycle
  cli.py         # Command-line interface
dashboard.py     # Read-only Streamlit app
config/default.json
tests/
```

## Limits and next research steps

This is a research MVP, not evidence of profitability. Artificial demo results say nothing about real stocks. Today's chosen symbols introduce selection/survivorship bias; delisted-universe data is not included. Adjusted daily prices, constant costs, and full fills limit realism. Statistical significance, confidence intervals, market-regime analysis, exact corporate-action accounting, and continuous operational monitoring remain future work.

Freeze the universe and hypothesis, obtain a verified dataset, and preserve unseen data. The research suite now provides LSTM, Transformer, headline-sentiment, and tabular Q-learning baselines. Next steps include finance-specific sentiment, richer RL environments, and parameter selection using training/validation data only. Evaluate risk-adjusted performance across costs and market conditions, not profit alone.
