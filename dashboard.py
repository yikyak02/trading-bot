"""Read-only dashboard. Start with: streamlit run dashboard.py"""
import json
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stock Research Lab", page_icon="📈", layout="wide")
st.title("Stock Research Lab")
st.caption("Daily US equities · explainable decisions · paper execution only")

runs_root = Path(st.sidebar.text_input("Runs directory", "runs"))
available = sorted([p.parent for p in runs_root.glob("*/summary.csv")
                    if not (p.parent / "RUNNING").exists()], key=lambda p: p.name, reverse=True)
if not available:
    st.info("Create your first experiment with the offline demo.")
    st.code("python -m trading_bot demo\nstreamlit run dashboard.py", language="bash")
    st.stop()
run = st.sidebar.selectbox("Experiment", available, format_func=lambda p: p.name)
manifest = json.loads((run / "manifest.json").read_text())
advanced = manifest.get("experiment") == "advanced_comparison"
if advanced:
    st.info("Exploratory model comparison. Seeds measure training variation, not statistical significance.")
    seed = st.sidebar.selectbox("Training seed", manifest["options"]["seeds"])
    with st.expander("Comparison across seeds"):
        st.dataframe(pd.read_csv(run / "comparisons.csv"), hide_index=True)
        st.dataframe(pd.read_csv(run / "prediction_metrics.csv"), hide_index=True)
    run = run / f"seed-{seed}"
if manifest["dataset"]["source"] == "synthetic_demo":
    st.warning("Synthetic demo — artificial prices. These results do not measure actual stock performance.")
else:
    st.info("Research uses adjusted OHLC as a total-return proxy. Fills and costs are simulated.")
period = st.sidebar.radio("Evaluation period", ["test", "validation"],
                          format_func=lambda x: "Final test · frozen model" if x == "test" else "Walk-forward validation")
strategies = manifest.get("strategies", ["ml", "technical", "buy_hold"])
strategy = st.sidebar.selectbox("Inspect strategy", strategies)
summary = pd.read_csv(run / "summary.csv")
selected = summary[(summary.period == period) & (summary.strategy == strategy)].iloc[0]
cols = st.columns(4)
cols[0].metric("Total return", f"{selected.total_return:.2%}")
cols[1].metric("Sharpe ratio", "N/A" if pd.isna(selected.sharpe_ratio) else f"{selected.sharpe_ratio:.2f}")
cols[2].metric("Maximum drawdown", f"{selected.max_drawdown:.2%}")
cols[3].metric("Transaction costs", f"${selected.transaction_costs:,.2f}")
performance, decisions_tab, risk_tab, paper_tab, provenance = st.tabs([
    "Performance", "Decision journal", "Risk & costs", "Paper account", "Experiment details"])
with performance:
    curves, drawdowns = {}, {}
    for name in strategies:
        frame = pd.read_csv(run / period / name / "equity.csv", parse_dates=["date"]).set_index("date")
        curves[name] = frame.equity
        drawdowns[name] = frame.drawdown * 100
    st.subheader("Equity comparison")
    st.line_chart(pd.DataFrame(curves), y_label="Portfolio value ($)")
    st.subheader("Drawdown")
    st.line_chart(pd.DataFrame(drawdowns), y_label="Drawdown (%)")
    st.dataframe(summary[summary.period == period].drop(columns="period"), hide_index=True, use_container_width=True)
    st.caption("Active strategies share entry risk limits. Buy-and-hold is fully invested without stops. Each period starts with fresh cash.")
with decisions_tab:
    decisions = pd.read_csv(run / period / strategy / "decisions.csv")
    symbols = st.multiselect("Symbols", sorted(decisions.symbol.unique()), default=sorted(decisions.symbol.unique()))
    actions = st.multiselect("Actions", sorted(decisions.action.unique()), default=sorted(decisions.action.unique()))
    filtered = decisions[decisions.symbol.isin(symbols) & decisions.action.isin(actions)]
    st.dataframe(filtered[["date", "symbol", "action", "reason", "risk_rule"]].tail(200), hide_index=True, use_container_width=True)
    if not filtered.empty:
        indices = list(filtered.tail(200).index)[::-1]
        chosen = st.selectbox("Inspect a decision", indices, format_func=lambda i:
                              f"{decisions.loc[i, 'date']} · {decisions.loc[i, 'symbol']} · {decisions.loc[i, 'action']}")
        decision = decisions.loc[chosen]
        st.write(decision.reason)
        st.json(json.loads(decision.to_json()))
        st.caption("Experimental models record forecasts or policy actions; no feature attribution is claimed." if advanced else
                   "ML sensitivity replaces one feature with its training median. It is model sensitivity, not causal attribution or SHAP.")
    st.download_button("Download decision journal", decisions.to_csv(index=False), "decisions.csv", "text/csv")
    st.subheader("Executed fills")
    st.dataframe(pd.read_csv(run / period / strategy / "trades.csv").tail(100), hide_index=True, use_container_width=True)
with risk_tab:
    st.subheader("Double-cost stress test · final test period")
    st.dataframe(pd.read_csv(run / "cost_stress.csv"), hide_index=True, use_container_width=True)
    st.subheader("Last-session holdings")
    holdings = pd.read_csv(run / period / strategy / "holdings.csv")
    last_date = str(pd.DataFrame(curves).index.max().date())
    latest = holdings[holdings.date == last_date]
    if latest.empty:
        st.info("This strategy ended the period in cash.")
    else:
        st.dataframe(latest, hide_index=True, use_container_width=True)
    st.json({key: manifest["config"][key] for key in ["max_position_weight", "risk_per_position", "stop_method",
             "stop_loss_pct", "atr_multiple", "max_drawdown", "fee_bps", "slippage_bps", "spread_bps"]})
with paper_tab:
    state_path = Path(st.text_input("Paper state database", "state/paper.sqlite"))
    if state_path.is_file():
        try:
            with sqlite3.connect(state_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
                row = db.execute("SELECT value FROM state WHERE key='last_snapshot'").fetchone()
                journal = pd.read_sql_query("SELECT timestamp, event FROM events ORDER BY rowid DESC LIMIT 100", db)
            if row:
                snapshot = json.loads(row[0])
                st.caption(f"Last cycle: {snapshot['timestamp']} — snapshot taken before this cycle's orders settle.")
                st.json(snapshot)
            else:
                st.info("No submitted paper cycle recorded yet.")
            st.dataframe(journal, hide_index=True, use_container_width=True)
        except sqlite3.Error as exc:
            st.error(f"Cannot read paper state: {exc}")
    else:
        st.info("No paper account connected. Configure API credentials and run a paper cycle from the CLI.")
    st.caption("This dashboard is read-only. Refresh to load the latest recorded snapshot.")
with provenance:
    st.json(manifest)
    st.download_button("Download experiment manifest", json.dumps(manifest, indent=2), "manifest.json", "application/json")
