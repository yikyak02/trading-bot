"""Daily event simulator: previous close signals, next open fills, gap-aware stops."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURES
from .strategy import order_quantity, position_budget, signal, stop_distance


@dataclass
class Result:
    equity: pd.DataFrame
    trades: pd.DataFrame
    decisions: pd.DataFrame
    holdings: pd.DataFrame
    metrics: dict


def metrics(equity, trades, config):
    values = equity.equity.to_numpy(dtype=float)
    returns = pd.Series(values).pct_change().dropna()
    total_return = values[-1] / values[0] - 1
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    daily_rf = (1 + config.risk_free_rate) ** (1 / 252) - 1
    sells = trades.loc[trades.side == "sell"] if not trades.empty else trades
    peaks = np.maximum.accumulate(values)
    result = {
        "total_return": float(total_return),
        "annualized_return": float((values[-1] / values[0]) ** (252 / max(1, len(returns))) - 1),
        "sharpe_ratio": float((returns.mean() - daily_rf) / std * np.sqrt(252)) if std > 1e-12 else None,
        "max_drawdown": float(np.min(values / peaks - 1)),
        "volatility": float(std * np.sqrt(252)),
        "win_rate": float((sells.realized_pnl > 0).mean()) if len(sells) else None,
        "closed_trades": len(sells), "fills": len(trades),
        "fees": float(trades.fee.sum()) if len(trades) else 0.0,
        "slippage_and_spread": float(trades.impact_cost.sum()) if len(trades) else 0.0,
        "transaction_costs": float((trades.fee + trades.impact_cost).sum()) if len(trades) else 0.0,
        "turnover": float(trades.notional.sum() / equity.equity.mean()) if len(trades) else 0.0,
        "average_exposure": float(equity.exposure.iloc[1:].mean()),
        "ending_equity": float(values[-1]), "sessions": len(returns),
    }
    return result


def run_backtest(bars, signals, strategy, config):
    if signals.empty:
        raise ValueError("No signals to simulate.")
    dates = pd.DatetimeIndex(sorted(bars.date.unique()))
    next_session = dict(zip(dates[:-1], dates[1:]))
    indexed = bars.set_index(["date", "symbol"])
    cash = config.initial_cash
    positions, trades, decisions, holdings = {}, [], [], []
    first_signal = signals.date.min()
    curve = [{"date": first_signal, "equity": cash, "cash": cash, "exposure": 0.0,
              "drawdown": 0.0, "halted": False}]
    peak, halted = cash, False
    passive = strategy == "buy_hold"
    entered_passive = set()
    fee_rate = config.fee_bps / 10000
    impact_rate = (config.slippage_bps + config.spread_bps / 2) / 10000

    def fill(date, symbol, side, qty, reference, reason, signal_date, distance=0.0):
        nonlocal cash
        price = reference * (1 + impact_rate if side == "buy" else 1 - impact_rate)
        notional, fee = qty * price, qty * price * fee_rate
        pnl = None
        if side == "buy":
            cash -= notional + fee
            positions[symbol] = {"qty": qty, "entry": price, "cost": notional + fee,
                                 "stop": reference * (1 - distance)}
        else:
            cash += notional - fee
            pnl = notional - fee - positions[symbol]["cost"]
            del positions[symbol]
        trades.append({"date": date, "signal_date": signal_date, "symbol": symbol,
                       "side": side, "quantity": qty, "reference_price": reference, "fill_price": price,
                       "notional": notional, "fee": fee, "impact_cost": qty * abs(price - reference),
                       "realized_pnl": pnl, "reason": reason})

    for signal_date, day in signals.groupby("date", sort=True):
        if signal_date not in next_session:
            continue
        date = next_session[signal_date]
        held_at_signal = set(positions)
        market = {symbol: indexed.loc[(date, symbol)] for symbol in config.symbols}
        open_equity = cash + sum(p["qty"] * market[s].open for s, p in positions.items())
        if not passive and open_equity / peak - 1 <= -config.max_drawdown:
            halted = True
        stopped = set()
        # Opening gaps occur before the strategy's scheduled opening orders.
        for symbol, position in list(positions.items()):
            if not passive and market[symbol].open <= position["stop"]:
                fill(date, symbol, "sell", position["qty"], market[symbol].open,
                     "Stop-loss: opening gap, filled at open less impact.", signal_date)
                stopped.add(symbol)
        records = {}
        # Sell first. Entry allocations use the same opening equity snapshot.
        for _, row in day.iterrows():
            symbol = row.symbol
            intent = signal(strategy, row, config, held=symbol in held_at_signal)
            record = {"date": date, "signal_date": signal_date, "symbol": symbol,
                      "strategy": strategy, "action": "hold", "reason": intent.reason,
                      "prediction": intent.prediction, "requested_budget": 0.0, "approved_budget": 0.0,
                      "quantity": 0.0, "risk_rule": "none", "model_version": row.get("model_version", ""),
                      "sensitivity": row.get("sensitivity", "[]") if strategy == "ml" else "[]",
                      **{f"feature_{name}": row[name] for name in FEATURES}}
            records[symbol] = (row, intent, record)
            if symbol in stopped:
                record.update(action="sell", risk_rule="gap_stop", reason=record["reason"] + " Opening gap stop executed.")
            elif symbol in positions and not passive and (halted or not intent.eligible):
                why = "Portfolio drawdown halt: liquidate and suspend entries." if halted else intent.reason
                qty = positions[symbol]["qty"]
                fill(date, symbol, "sell", qty, market[symbol].open, why, signal_date)
                record.update(action="sell", quantity=qty, reason=why,
                              risk_rule="drawdown_halt" if halted else "signal_exit")
        for symbol in sorted(records):
            row, intent, record = records[symbol]
            if symbol in positions or symbol in stopped or (passive and symbol in entered_passive):
                continue
            if not intent.eligible:
                record["reason"] += " No position; entry conditions not met."
                continue
            if halted and not passive:
                record.update(action="rejected", risk_rule="drawdown_halt",
                              reason="Portfolio drawdown halt is latched; no new entries.")
                continue
            distance = stop_distance(row, config)
            if passive:
                requested = config.initial_cash / len(config.symbols)
                budget, limit = min(requested, cash), "passive_equal_weight"
            else:
                budget, limit, requested = position_budget(open_equity, cash, len(config.symbols), distance, config)
            qty = order_quantity(budget, market[symbol].open, config)
            record.update(requested_budget=requested, approved_budget=budget, risk_rule=limit, quantity=qty)
            if qty <= 1e-10:
                record.update(action="rejected", reason="Insufficient cash for entry.")
                continue
            record.update(action="buy", reason=record["reason"] + f" Entry budget limited by {limit}; stop distance {distance:.2%}.")
            fill(date, symbol, "buy", qty, market[symbol].open, record["reason"], signal_date, distance)
            if passive:
                entered_passive.add(symbol)
        decisions.extend(record for _, _, record in records.values())
        # A newly entered position may stop out on the entry candle. No same-day reentry.
        for symbol, position in list(positions.items()):
            if not passive and market[symbol].low <= position["stop"]:
                qty = position["qty"]
                stop = position["stop"]
                fill(date, symbol, "sell", qty, stop, "Intraday stop-loss at stop less impact.", signal_date)
                decisions.append({**records[symbol][2], "date": date, "signal_date": signal_date, "symbol": symbol,
                                  "strategy": strategy, "action": "sell", "quantity": qty,
                                  "reason": "Intraday low crossed the entry stop.", "risk_rule": "intraday_stop"})
        value = sum(p["qty"] * market[s].close for s, p in positions.items())
        equity = cash + value
        peak = max(peak, equity)
        if not passive and equity / peak - 1 <= -config.max_drawdown:
            halted = True
        if cash < -1e-6 or equity <= 0:
            raise AssertionError("Cash-only portfolio accounting violated.")
        curve.append({"date": date, "equity": equity, "cash": max(0, cash), "exposure": value / equity,
                      "drawdown": equity / peak - 1, "halted": halted})
        for symbol, p in positions.items():
            holdings.append({"date": date, "symbol": symbol, "quantity": p["qty"],
                             "market_value": p["qty"] * market[symbol].close,
                             "weight": p["qty"] * market[symbol].close / equity, "stop": None if passive else p["stop"]})
    if len(curve) < 2:
        raise ValueError("No subsequent session exists for these signals.")
    equity_frame = pd.DataFrame(curve)
    trade_frame = pd.DataFrame(trades, columns=["date", "signal_date", "symbol", "side", "quantity",
                 "reference_price", "fill_price", "notional", "fee", "impact_cost", "realized_pnl", "reason"])
    return Result(equity_frame, trade_frame, pd.DataFrame(decisions),
                  pd.DataFrame(holdings, columns=["date", "symbol", "quantity", "market_value", "weight", "stop"]),
                  metrics(equity_frame, trade_frame, config))
