"""Shared daily entry/exit conditions and stop-distance portfolio sizing."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Intent:
    eligible: bool
    reason: str
    prediction: float | None = None


def signal(strategy, row, config):
    if strategy == "buy_hold":
        return Intent(True, "Passive equal-weight allocation; buy once and hold.")
    if strategy == "technical":
        eligible = row["sma_ratio"] > 0 and row["distance_sma50"] > 0
        return Intent(bool(eligible), f"SMA20/SMA50 - 1 = {row['sma_ratio']:.4f}; "
                      f"close/SMA50 - 1 = {row['distance_sma50']:.4f}. Both must be positive.")
    if strategy == "ml":
        pred = float(row["prediction"])
        return Intent(pred > config.ml_entry_threshold,
                      f"Forecast next open-to-open return {pred:.4%}; entry threshold "
                      f"{config.ml_entry_threshold:.4%}.", pred)
    raise ValueError(f"Unknown strategy: {strategy}")


def stop_distance(row, config):
    if config.stop_method == "fixed":
        return config.stop_loss_pct
    return min(0.5, max(0.01, float(row["atr_pct"]) * config.atr_multiple))


def position_budget(equity, cash, asset_count, distance, config):
    """Caps apply at entry. Positions are held without daily resizing."""
    requested = equity / asset_count
    caps = {"equal_weight": requested, "position_cap": equity * config.max_position_weight,
            "stop_risk_budget": equity * config.risk_per_position / distance,
            "available_cash": max(0.0, cash)}
    limiting = min(caps, key=caps.get)
    return max(0.0, caps[limiting]), limiting, requested


def order_quantity(budget, price, config, whole=False):
    impact = (config.slippage_bps + config.spread_bps / 2) / 10000
    quantity = budget / (price * (1 + impact) * (1 + config.fee_bps / 10000))
    return float(math.floor(quantity)) if whole else quantity
