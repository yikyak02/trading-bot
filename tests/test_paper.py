from copy import deepcopy
import json

import pytest
import requests

from trading_bot.broker import AlpacaPaper, BrokerError
from trading_bot.paper import Ledger, paper_cycle


class FakeBroker:
    def __init__(self):
        self.orders = {}
        self.submissions = []
        self.open_orders = []
        self.positions = []
        self.timestamp = "2024-03-11T13:35:00Z"
        self.quote_timestamp = self.timestamp
        self.is_open = True

    def get(self, path, params=None):
        if path == "/v2/clock":
            return {"timestamp": self.timestamp, "is_open": self.is_open}
        if path == "/v2/account":
            return {"id": "paper123", "status": "ACTIVE", "equity": "100000", "cash": "100000"}
        if path == "/v2/positions":
            return deepcopy(self.positions)
        if path == "/v2/orders":
            return deepcopy(self.open_orders)
        if path.startswith("/v2/assets/"):
            return {"tradable": True, "status": "active"}
        raise AssertionError(path)

    def data_get(self, path, params=None):
        return {"quote": {"t": self.quote_timestamp, "bp": 150, "ap": 150.1}}

    def find_order(self, client_id):
        return self.orders.get(client_id)

    def submit(self, payload):
        self.submissions.append(payload)
        response = {**payload, "status": "accepted", "id": f"order-{len(self.submissions)}"}
        self.orders[payload["client_order_id"]] = response
        return response


def test_preview_never_posts(paper_dataset, config, tmp_path):
    broker = FakeBroker()
    result = paper_cycle(broker, paper_dataset, "technical", config, tmp_path / "paper.db")
    assert not result["submitted"]
    assert all(x["action"] == "buy" for x in result["decisions"])
    assert not broker.submissions


def test_restarts_do_not_duplicate_orders(paper_dataset, config, tmp_path):
    broker = FakeBroker()
    state = tmp_path / "paper.db"
    first = paper_cycle(broker, paper_dataset, "technical", config, state, submit=True)
    paper_cycle(broker, paper_dataset, "technical", config, state, submit=True)
    assert len(broker.submissions) == 3
    assert all(x["order_class"] == "oto" for x in broker.submissions)
    assert all(float(x["qty"]).is_integer() for x in broker.submissions)
    ledger = Ledger(state)
    assert ledger.get("binding")["account"] == "paper123"
    ledger.close()


def test_timeout_reconciles_accepted_order_without_reposting(tmp_path):
    ledger = Ledger(tmp_path / "state.db")
    broker = FakeBroker()
    payload = {"client_order_id": "xstock-demo", "symbol": "AAPL"}
    submit = broker.submit

    def timeout_after_accept(order):
        submit(order)
        raise requests.Timeout("Simulated response loss")

    broker.submit = timeout_after_accept
    with pytest.raises(requests.Timeout):
        ledger.submit_once(broker, payload)
    ledger.reconcile(broker)
    assert ledger.submit_once(broker, payload)["status"] == "accepted"
    assert len(broker.submissions) == 1
    ledger.close()


def test_uncertain_order_blocks_resubmission(tmp_path):
    ledger = Ledger(tmp_path / "state.db")
    broker = FakeBroker()

    def timeout(order):
        raise requests.Timeout("Unknown outcome")

    broker.submit = timeout
    payload = {"client_order_id": "xstock-demo"}
    with pytest.raises(requests.Timeout):
        ledger.submit_once(broker, payload)
    with pytest.raises(BrokerError, match="uncertain"):
        ledger.reconcile(broker)
    with pytest.raises(BrokerError, match="refusing to resubmit"):
        ledger.submit_once(broker, payload)
    ledger.close()


def test_stale_quote_rejected(paper_dataset, config, tmp_path):
    broker = FakeBroker()
    broker.quote_timestamp = "2024-03-08T20:00:00Z"
    with pytest.raises(BrokerError, match="quote is stale"):
        paper_cycle(broker, paper_dataset, "technical", config, tmp_path / "paper.db", submit=True)
    assert not broker.submissions


def test_closed_market_rejected(paper_dataset, config, tmp_path):
    broker = FakeBroker()
    broker.is_open = False
    with pytest.raises(BrokerError, match="first 30 minutes"):
        paper_cycle(broker, paper_dataset, "technical", config, tmp_path / "paper.db")


def test_stale_dataset_rejected(paper_dataset, config, tmp_path):
    broker = FakeBroker()
    broker.timestamp = "2024-03-12T13:35:00Z"
    with pytest.raises(BrokerError, match="Stale or unfinished"):
        paper_cycle(broker, paper_dataset, "technical", config, tmp_path / "paper.db")


def test_demo_cannot_submit(paper_dataset, config, tmp_path):
    meta_path = paper_dataset / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    metadata["source"] = "synthetic_demo"
    meta_path.write_text(json.dumps(metadata))
    broker = FakeBroker()
    with pytest.raises(BrokerError, match="demo data is prohibited"):
        paper_cycle(broker, paper_dataset, "technical", config, tmp_path / "paper.db", submit=True)
    assert not broker.submissions


def test_transport_only_uses_paper_host_and_does_not_retry_posts(monkeypatch):
    client = AlpacaPaper("fake-key", "fake-secret")
    captured = []

    class Response:
        status_code = 200
        content = b"{}"
        def json(self):
            return {}

    def fake_request(method, url, **kwargs):
        captured.append((method, url, kwargs))
        return Response()

    monkeypatch.setattr(client.session, "request", fake_request)
    client.submit({"symbol": "AAPL"})
    assert captured[0][1] == "https://paper-api.alpaca.markets/v2/orders"
    assert captured[0][2]["allow_redirects"] is False
    assert client.session.get_adapter("https://").max_retries.total == 0


def test_first_submission_requires_empty_account(paper_dataset, config, tmp_path):
    broker = FakeBroker()
    broker.positions = [{"symbol": "AAPL", "side": "long", "qty": "10"}]
    with pytest.raises(BrokerError, match="empty dedicated"):
        paper_cycle(broker, paper_dataset, "technical", config, tmp_path / "paper.db", submit=True)
    ledger = Ledger(tmp_path / "paper.db")
    assert ledger.get("binding") is None
    ledger.close()


def test_drawdown_halt_blocks_entries(paper_dataset, config, tmp_path):
    state = tmp_path / "paper.db"
    ledger = Ledger(state)
    ledger.set("peak_equity", 150000)
    ledger.close()
    broker = FakeBroker()
    result = paper_cycle(broker, paper_dataset, "technical", config, state, submit=True)
    assert result["halted"]
    assert all(d["action"] == "rejected" for d in result["decisions"])
    assert not broker.submissions


def test_existing_halt_never_auto_resets(paper_dataset, config, tmp_path):
    state = tmp_path / "paper.db"
    ledger = Ledger(state)
    ledger.set("halted", True)
    ledger.close()
    broker = FakeBroker()
    assert paper_cycle(broker, paper_dataset, "technical", config, state, submit=True)["halted"]
    assert not broker.submissions


def test_stop_filling_during_cancellation_aborts(monkeypatch):
    client = AlpacaPaper("fake-key", "fake-secret")
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: {})
    monkeypatch.setattr(client, "get", lambda *args, **kwargs: {"status": "filled"})
    with pytest.raises(BrokerError, match="filled during cancellation"):
        client.cancel_and_confirm("stop-id")


def test_oto_lookup_fetches_protective_legs(monkeypatch):
    client = AlpacaPaper("fake-key", "fake-secret")
    calls = []
    def request(method, path, **kwargs):
        calls.append((path, kwargs))
        if path.endswith("by_client_order_id"):
            return {"id": "parent", "order_class": "oto"}
        return {"id": "parent", "legs": [{"id": "protective-stop"}]}
    monkeypatch.setattr(client, "_request", request)
    order = client.find_order("xstock-test")
    assert order["legs"][0]["id"] == "protective-stop"
    assert calls[-1][1]["params"] == {"nested": "true"}


def test_current_quote_after_initial_clock_is_accepted(paper_dataset, config, tmp_path):
    broker = FakeBroker()
    broker.quote_timestamp = "2024-03-11T13:35:02Z"
    result = paper_cycle(broker, paper_dataset, "technical", config, tmp_path / "state.db")
    assert all(d["action"] == "buy" for d in result["decisions"])


def test_explanation_is_durable_before_network_submission(paper_dataset, config, tmp_path):
    broker = FakeBroker()
    def timeout(payload):
        raise requests.Timeout("Unknown outcome")
    broker.submit = timeout
    state = tmp_path / "state.db"
    with pytest.raises(requests.Timeout):
        paper_cycle(broker, paper_dataset, "technical", config, state, submit=True)
    ledger = Ledger(state)
    events = [json.loads(row[0]) for row in ledger.db.execute("SELECT event FROM events")]
    assert events[0]["phase"] == "order_intent"
    assert events[0]["features"] and events[0]["reason"]
    ledger.close()
