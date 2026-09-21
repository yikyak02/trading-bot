from dataclasses import asdict, dataclass, fields
import json
import math
import re
from pathlib import Path


@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "JPM")
    initial_cash: float = 100_000.0
    max_position_weight: float = 0.30
    risk_per_position: float = 0.01
    stop_loss_pct: float = 0.05
    stop_method: str = "fixed"
    atr_multiple: float = 2.0
    max_drawdown: float = 0.15
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    spread_bps: float = 2.0
    risk_free_rate: float = 0.0
    ml_entry_threshold: float = 0.002
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    retrain_sessions: int = 21
    min_train_sessions: int = 126
    seed: int = 42
    feed: str = "iex"

    def __post_init__(self):
        object.__setattr__(self, "symbols", tuple(self.symbols))
        if not 3 <= len(self.symbols) <= 5 or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Choose 3–5 unique US stock symbols.")
        if any(not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", s) for s in self.symbols):
            raise ValueError("Invalid stock symbol.")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite.")
        for name in ("max_position_weight", "risk_per_position", "stop_loss_pct", "max_drawdown"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.initial_cash <= 0 or self.atr_multiple <= 0:
            raise ValueError("Cash and ATR multiple must be positive.")
        if any(not 0 <= getattr(self, n) < 1000 for n in ("fee_bps", "slippage_bps", "spread_bps")):
            raise ValueError("Costs must be between 0 and 1000 basis points.")
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("Train and validation fractions must be positive.")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("Reserve a nonempty final test period.")
        if self.min_train_sessions < 30 or self.retrain_sessions < 1:
            raise ValueError("Need >=30 training sessions and a positive retraining interval.")
        if self.stop_method not in {"fixed", "atr"} or self.feed not in {"iex", "sip"}:
            raise ValueError("Unsupported stop method or data feed.")
        if self.risk_free_rate <= -1 or self.ml_entry_threshold < 0:
            raise ValueError("Invalid risk-free rate or prediction threshold.")

    def as_dict(self):
        return asdict(self)


def load_config(path: str | Path | None = None) -> Config:
    return Config(**json.loads(Path(path).read_text())) if path else Config()
