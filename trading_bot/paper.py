"""One scheduled paper cycle with persistent, fail-closed order reconciliation."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import pandas as pd

from .broker import BrokerError
from .data import calendar, load_dataset, sessions
from .features import FEATURES, build_features
from .models import fit_model
from .strategy import order_quantity, position_budget, signal, stop_distance

TERMINAL = {"filled", "canceled", "expired", "rejected", "replaced"}


class Ledger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS orders (client_id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL, response TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS events (timestamp TEXT NOT NULL, event TEXT NOT NULL)")
        self.db.commit()

    def close(self):
        self.db.close()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO state VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()

    def event(self, value):
        self.db.execute("INSERT INTO events VALUES (?, ?)", (datetime.now(timezone.utc).isoformat(), json.dumps(value)))
        self.db.commit()

    def reconcile(self, client):
        for client_id, status in self.db.execute("SELECT client_id, status FROM orders").fetchall():
            # Refresh filled parents too: protective legs may appear after acceptance.
            if status not in {"canceled", "expired", "rejected", "replaced"}:
                order = client.find_order(client_id)
                if order is None:
                    raise BrokerError(f"Order {client_id} has uncertain broker state. Inspect it before continuing; automatic resubmission is disabled.")
                self.db.execute("UPDATE orders SET status=?, response=? WHERE client_id=?",
                                (order["status"], json.dumps(order), client_id))
        self.db.commit()

    def submit_once(self, client, payload):
        client_id = payload["client_order_id"]
        row = self.db.execute("SELECT status, response FROM orders WHERE client_id=?", (client_id,)).fetchone()
        if row:
            existing = client.find_order(client_id)
            if existing:
                self.db.execute("UPDATE orders SET status=?, response=? WHERE client_id=?",
                                (existing["status"], json.dumps(existing), client_id))
                self.db.commit()
                return existing
            raise BrokerError("A local order intent already exists but is absent at the broker; refusing to resubmit.")
        existing = client.find_order(client_id)
        if existing:
            self.db.execute("INSERT INTO orders VALUES (?, ?, ?, ?)",
                            (client_id, json.dumps(payload), existing["status"], json.dumps(existing)))
            self.db.commit()
            return existing
        self.db.execute("INSERT INTO orders VALUES (?, ?, 'uncertain', NULL)", (client_id, json.dumps(payload)))
        self.db.commit()  # Durable intent before the network request.
        response = client.submit(payload)
        self.db.execute("UPDATE orders SET status=?, response=? WHERE client_id=?",
                        (response["status"], json.dumps(response), client_id))
        self.db.commit()
        return response


@contextmanager
def cycle_lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BrokerError("Another paper cycle is running.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def client_order_id(account, session, strategy, symbol, side):
    # Independent of config changes or process restarts: at most one same-side order per day.
    digest = hashlib.sha256(f"{account}|{session}|{strategy}|{symbol}|{side}".encode()).hexdigest()[:28]
    return f"xstock-{digest}"


def flatten_orders(orders):
    result = {}
    for order in orders:
        if order.get("status") not in TERMINAL:
            result[order["id"]] = order
        for leg in order.get("legs") or []:
            if leg.get("status") not in TERMINAL:
                result[leg["id"]] = leg
    return list(result.values())


def paper_cycle(client, dataset, strategy, config, state_path="state/paper.sqlite", submit=False):
    if strategy not in {"technical", "ml"}:
        raise ValueError("Paper execution supports technical or ml strategies.")
    with cycle_lock(state_path):
        ledger = Ledger(state_path)
        try:
            return _cycle(client, dataset, strategy, config, ledger, submit)
        finally:
            ledger.close()


def _cycle(client, dataset, strategy, config, ledger, submit):
    bars, metadata = load_dataset(dataset, config.symbols)
    if metadata.get("source") != "alpaca" or metadata.get("feed") != config.feed:
        raise BrokerError("Paper execution requires an Alpaca dataset with the configured feed; demo data is prohibited.")
    clock = client.get("/v2/clock")
    now = pd.Timestamp(clock["timestamp"])
    today = now.tz_convert("America/New_York").tz_localize(None).normalize()
    cal = calendar(today - pd.Timedelta(days=14), today + pd.Timedelta(days=7))
    if not clock["is_open"] or not cal.is_session(today):
        raise BrokerError("Run the daily cycle during the first 30 minutes of the regular US trading session.")
    if not cal.session_open(today) <= now <= cal.session_open(today) + pd.Timedelta(minutes=30):
        raise BrokerError("Entry window closed: run during the first 30 minutes after the regular market open.")
    previous = sessions(today - pd.Timedelta(days=14), today - pd.Timedelta(days=1))[-1]
    if bars.date.max() != previous:
        raise BrokerError(f"Stale or unfinished data: expected last completed session {previous.date()}.")
    account = client.get("/v2/account")
    if account.get("trading_blocked") or account.get("account_blocked") or account.get("status") != "ACTIVE":
        raise BrokerError("Paper account is not active or is blocked.")
    binding = {"account": account["id"], "strategy": strategy, "config": config.as_dict()}
    # Roundtrip normalizes tuples to lists before comparison.
    binding = json.loads(json.dumps(binding))
    prior_binding = ledger.get("binding")
    if prior_binding and prior_binding != binding:
        raise BrokerError("State is bound to another account, strategy, or config. Use a dedicated paper account and state file.")
    ledger.reconcile(client)
    positions = {p["symbol"]: p for p in client.get("/v2/positions")}
    if any(s not in config.symbols or p.get("side") != "long" for s, p in positions.items()):
        raise BrokerError("Use a dedicated paper account containing only the configured long-only universe.")
    orders_response = client.get("/v2/orders", {"status": "open", "nested": "true", "limit": 500})
    if len(orders_response) >= 500:
        raise BrokerError("Too many open orders to reconcile safely.")
    orders = flatten_orders(orders_response)
    if submit and not prior_binding and (positions or orders):
        raise BrokerError("First submitted cycle requires an empty dedicated paper account.")
    if any(o["symbol"] not in config.symbols for o in orders):
        raise BrokerError("Unexpected open orders outside the configured universe.")
    # Only this bot's parent orders / protective legs may be managed.
    known_ids = set()
    for (response,) in ledger.db.execute("SELECT response FROM orders WHERE response IS NOT NULL"):
        order = json.loads(response)
        known_ids.add(order["id"])
        known_ids.update(leg["id"] for leg in order.get("legs") or [])
    for order in orders_response:
        if order.get("client_order_id", "").startswith("xstock-"):
            known_ids.add(order["id"])
            known_ids.update(leg["id"] for leg in order.get("legs") or [])
    if any(o["id"] not in known_ids for o in orders):
        raise BrokerError("Unrecognized open order. Refusing to change manually managed orders.")
    if submit and not prior_binding:
        ledger.set("binding", binding)
    equity, cash = float(account["equity"]), max(0.0, float(account["cash"]))
    if not math.isfinite(equity) or not math.isfinite(cash) or equity <= 0:
        raise BrokerError("Account equity must be positive.")
    peak = max(float(ledger.get("peak_equity", equity)), equity)
    halted = bool(ledger.get("halted", False)) or equity / peak - 1 <= -config.max_drawdown
    if submit:
        ledger.set("peak_equity", peak)
        ledger.set("halted", halted)
    features = build_features(bars)
    day = features.loc[features.date == previous].dropna(subset=FEATURES).copy()
    if set(day.symbol) != set(config.symbols):
        raise BrokerError("Insufficient valid feature history for all assets.")
    model = None
    if strategy == "ml":
        model = fit_model(features, previous, config)
        day["prediction"] = model.predict(day)
    # No new buys while any parent/market exit is pending. Stops do not reserve cash.
    pending = [o for o in orders if o.get("type") not in {"stop", "stop_limit"}]
    plans = []
    for _, row in day.sort_values("symbol").iterrows():
        symbol = row.symbol
        intent = signal(strategy, row, config)
        record = {"symbol": symbol, "signal_date": str(previous.date()), "session": str(today.date()),
                  "action": "hold", "reason": intent.reason, "prediction": intent.prediction,
                  "features": {f: float(row[f]) for f in FEATURES},
                  "model": model.audit() if model else None,
                  "sensitivity": model.explain(row) if model else [], "mode": "submit" if submit else "preview"}
        symbol_orders = [o for o in orders if o["symbol"] == symbol]
        if symbol in positions:
            if pending:
                record["reason"] += " Pending order; reconcile on next cycle."
            elif halted or not intent.eligible:
                record.update(action="sell", reason="Drawdown halt exit." if halted else intent.reason)
                payload = {"symbol": symbol, "side": "sell", "type": "market", "time_in_force": "day",
                           "qty": str(positions[symbol]["qty"]),
                           "client_order_id": client_order_id(account["id"], today.date(), strategy, symbol, "sell")}
                record["cancel_before_exit"] = [o["id"] for o in symbol_orders]
                if submit:
                    execution_clock = client.get("/v2/clock")
                    if not execution_clock["is_open"] or pd.Timestamp(execution_clock["timestamp"]) > cal.session_open(today) + pd.Timedelta(minutes=30):
                        raise BrokerError("Execution window closed during this cycle; no exit submitted.")
                    for order in symbol_orders:
                        client.cancel_and_confirm(order["id"])
                    refreshed = {p["symbol"]: p for p in client.get("/v2/positions")}
                    if symbol not in refreshed:
                        record.update(action="hold", reason="Position already closed during reconciliation.")
                        plans.append(record)
                        ledger.event(record)
                        continue
                    payload["qty"] = str(refreshed[symbol]["qty"])
                    ledger.event({**record, "phase": "order_intent", "order": payload})
                    record["broker_response"] = ledger.submit_once(client, payload)
                record["order"] = payload
            else:
                protected = sum(float(o.get("qty", 0)) - float(o.get("filled_qty", 0))
                                for o in symbol_orders if o.get("side") == "sell" and o.get("type") in {"stop", "stop_limit"})
                if protected + 1e-6 < float(positions[symbol]["qty"]):
                    raise BrokerError(f"{symbol} has insufficient protective-stop coverage; inspect the paper account before continuing.")
                record["reason"] += " Existing position retained with broker stop."
        elif halted or pending or symbol_orders:
            record.update(action="rejected" if intent.eligible else "hold",
                          reason="Entry blocked by drawdown halt or pending broker orders.")
        elif intent.eligible:
            asset = client.get(f"/v2/assets/{symbol}")
            if not asset.get("tradable") or asset.get("status") != "active":
                raise BrokerError(f"{symbol} is not currently tradable.")
            quote = client.data_get(f"/v2/stocks/{symbol}/quotes/latest", {"feed": config.feed})["quote"]
            quote_clock = client.get("/v2/clock")
            quote_now = pd.Timestamp(quote_clock["timestamp"])
            if not quote_clock["is_open"] or quote_now > cal.session_open(today) + pd.Timedelta(minutes=30):
                raise BrokerError("Entry window closed while preparing orders.")
            age = (quote_now - pd.Timestamp(quote["t"])).total_seconds()
            bid, ask = float(quote["bp"]), float(quote["ap"])
            if not -5 <= age <= 60 or not (math.isfinite(bid) and math.isfinite(ask)) or not 0 < bid <= ask or (ask - bid) / ask > 0.02:
                raise BrokerError(f"{symbol}: quote is stale, invalid, or excessively wide.")
            if bid < 1:
                raise BrokerError("This stock MVP excludes sub-dollar shares.")
            distance = stop_distance(row, config)
            budget, rule, requested = position_budget(equity, cash, len(config.symbols), distance, config)
            qty = order_quantity(budget, ask, config, whole=True)
            record.update(requested_budget=requested, approved_budget=budget, risk_rule=rule)
            if qty < 1:
                record.update(action="rejected", reason="Budget cannot fund one whole share.")
            else:
                payload = {"symbol": symbol, "side": "buy", "type": "market", "time_in_force": "gtc",
                           "qty": str(int(qty)), "order_class": "oto",
                           "stop_loss": {"stop_price": f"{bid * (1 - distance):.2f}"},
                           "client_order_id": client_order_id(account["id"], today.date(), strategy, symbol, "buy")}
                record.update(action="buy", order=payload)
                if submit:
                    ledger.event({**record, "phase": "order_intent"})
                    record["broker_response"] = ledger.submit_once(client, payload)
                # Reserve this cycle's planned cash; do not spend expected sale proceeds.
                cash -= budget
        plans.append(record)
        if submit:
            ledger.event({**record, "phase": "cycle_result"})
    snapshot = {"timestamp": str(now), "account_equity": equity, "peak_equity": peak,
                "drawdown": equity / peak - 1, "halted": halted, "positions": positions,
                "open_orders": orders, "decisions": plans, "submitted": submit}
    if submit:
        ledger.set("last_snapshot", snapshot)
    return snapshot
