import argparse
import json
from pathlib import Path
import sys

import requests

from .broker import AlpacaPaper, BrokerError
from .config import load_config
from .data import demo_dataset, fetch_dataset
from .paper import paper_cycle
from .research import run_research


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explainable daily stock research and paper-only execution.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate artificial prices and run the complete offline experiment")
    demo.add_argument("--dataset", default="data/demo")
    demo.add_argument("--output", default="runs/demo")
    fetch = commands.add_parser("fetch", help="Download completed Alpaca sessions, raw and adjusted")
    fetch.add_argument("--start", required=True, help="YYYY-MM-DD, inclusive")
    fetch.add_argument("--end", required=True, help="YYYY-MM-DD, inclusive; unfinished sessions excluded")
    fetch.add_argument("--dataset", required=True, help="New immutable dataset directory")
    backtest = commands.add_parser("backtest", help="Run chronological validation and frozen final-test comparisons")
    backtest.add_argument("--dataset", required=True)
    backtest.add_argument("--output", required=True)
    news = commands.add_parser("fetch-news", help="Download an immutable historical headline snapshot")
    news.add_argument("--start", required=True)
    news.add_argument("--end", required=True)
    news.add_argument("--output", required=True)
    experiment = commands.add_parser("experiment", help="Compare sequence models, sentiment, and Q-learning offline")
    experiment.add_argument("--dataset", required=True)
    experiment.add_argument("--output", required=True)
    experiment.add_argument("--news", help="Historical news snapshot; adds price-plus-sentiment comparison")
    experiment.add_argument("--models", nargs="+", choices=["ml", "lstm", "transformer", "q_learning", "ml_sentiment"],
                            default=["ml", "lstm", "transformer", "q_learning"])
    experiment.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    experiment.add_argument("--epochs", type=int, default=8)
    experiment.add_argument("--lookback", type=int, default=20)
    paper = commands.add_parser("paper", help="Preview or submit one daily Alpaca paper cycle")
    paper.add_argument("--dataset", required=True)
    paper.add_argument("--strategy", choices=["technical", "ml"], default="technical")
    paper.add_argument("--state", default="state/paper.sqlite")
    paper.add_argument("--submit", action="store_true", help="Submit paper orders; otherwise read-only broker preview")
    for command in (demo, fetch, backtest, paper, news, experiment):
        command.add_argument("--config", default=None, help="JSON config; omitted uses built-in defaults (IEX feed)")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "demo":
            if not Path(args.dataset).exists():
                demo_dataset(args.dataset, config.symbols, config.seed)
            print("SYNTHETIC DEMO: these prices are artificial, not actual stock history.")
            result = run_research(args.dataset, args.output, config)
            print(result[["period", "strategy", "total_return", "sharpe_ratio", "max_drawdown", "transaction_costs"]].to_string(index=False))
            print(f"Reports: {args.output}\nDashboard: streamlit run dashboard.py")
        elif args.command == "fetch":
            result = fetch_dataset(AlpacaPaper(), args.dataset, config.symbols, args.start, args.end, config.feed)
            print(json.dumps(result, indent=2))
        elif args.command == "backtest":
            print(run_research(args.dataset, args.output, config).to_string(index=False))
        elif args.command == "fetch-news":
            from .news import fetch_news
            result = fetch_news(AlpacaPaper(), args.output, config.symbols, args.start, args.end)
            print(json.dumps({k: v for k, v in result.items() if k != "sha256"}, indent=2))
        elif args.command == "experiment":
            from .experiments import run_experiments
            from .experimental_models import ExperimentOptions
            options = ExperimentOptions(seeds=tuple(args.seeds), epochs=args.epochs, lookback=args.lookback)
            result = run_experiments(args.dataset, args.output, config, options, args.news, args.models)
            print(result[["seed", "period", "strategy", "total_return", "sharpe_ratio", "max_drawdown"]].to_string(index=False))
        elif args.command == "paper":
            print(json.dumps(paper_cycle(AlpacaPaper(), args.dataset, args.strategy, config,
                                        args.state, args.submit), indent=2))
        return 0
    except (ValueError, BrokerError, OSError, requests.RequestException) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
