"""Alpaca REST transport. Trading host is hardcoded to the paper environment."""
import os
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class BrokerError(RuntimeError):
    pass


class AlpacaPaper:
    def __init__(self, key=None, secret=None):
        key = key or os.getenv("APCA_API_KEY_ID")
        secret = secret or os.getenv("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise BrokerError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY to your Alpaca paper credentials.")
        self.session = requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.read_session = requests.Session()
        self.read_session.headers.update(self.session.headers)
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=frozenset({"GET"}), raise_on_status=False)
        self.read_session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=0))

    def _request(self, method, path, *, data=False, params=None, payload=None, missing_ok=False):
        if not path.startswith("/v2/") and not (data and method == "GET" and path == "/v1beta1/news"):
            raise BrokerError("Only versioned Alpaca API paths are permitted.")
        if data and method != "GET":
            raise BrokerError("Market data adapter is read-only.")
        transport = self.read_session if method == "GET" else self.session
        response = transport.request(method, (DATA_URL if data else PAPER_URL) + path,
                                        params=params, json=payload, timeout=(10, 30), allow_redirects=False)
        if response.status_code == 404 and missing_ok:
            return None
        if not 200 <= response.status_code < 300:
            # Never echo auth headers, arbitrary response content, or secrets.
            raise BrokerError(f"Alpaca {method} {path} returned HTTP {response.status_code}.")
        return response.json() if response.content else {}

    def data_get(self, path, params=None):
        return self._request("GET", path, data=True, params=params)

    def get(self, path, params=None):
        return self._request("GET", path, params=params)

    def find_order(self, client_id):
        order = self._request("GET", "/v2/orders:by_client_order_id",
                              params={"client_order_id": client_id}, missing_ok=True)
        if order and order.get("order_class") in {"oto", "oco", "bracket"}:
            # Client-ID lookup need not include protective legs; explicitly request them.
            return self.get(f"/v2/orders/{order['id']}", {"nested": "true"})
        return order

    def submit(self, payload):
        # No transport-level POST retries. Reconcile client_order_id after uncertainty.
        return self._request("POST", "/v2/orders", payload=payload)

    def cancel_and_confirm(self, order_id):
        self._request("DELETE", f"/v2/orders/{order_id}")
        for _ in range(10):
            order = self.get(f"/v2/orders/{order_id}")
            if order["status"] in {"canceled", "expired", "rejected"}:
                return
            if order["status"] == "filled":
                raise BrokerError("Stop filled during cancellation. Rerun to reconcile before any new order.")
            time.sleep(0.3)
        raise BrokerError("Cancellation not confirmed; no replacement or exit order was submitted.")
