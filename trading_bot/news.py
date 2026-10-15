"""Immutable historical headlines and conservative close-time sentiment features."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import calendar

NEWS_FEATURES = ['news_sentiment', 'news_count']


def fetch_news(client, directory, symbols, start, end):
    directory = Path(directory)
    if directory.exists():
        raise ValueError('News output already exists. Choose a new snapshot directory.')
    first = pd.Timestamp(start, tz='America/New_York').tz_convert('UTC')
    last = (pd.Timestamp(end, tz='America/New_York') + pd.DateOffset(days=1)).tz_convert('UTC')
    if last <= first:
        raise ValueError('News end must be on or after start.')
    # A partial download is deliberately not a usable snapshot: metadata is written last.
    directory.mkdir(parents=True)
    token, seen, articles = None, set(), {}
    pages = 0
    while True:
        params = {'start': first.isoformat(), 'end': last.isoformat(), 'symbols': ','.join(symbols),
                  'limit': 50, 'sort': 'asc', 'include_content': 'false'}
        if token:
            params['page_token'] = token
        response = client.data_get('/v1beta1/news', params)
        (directory / f'page-{pages:05d}.json').write_text(json.dumps(response))
        pages += 1
        for article in response.get('news') or []:
            key = str(article['id'])
            if key not in articles or pd.Timestamp(article['updated_at']) > pd.Timestamp(articles[key]['updated_at']):
                articles[key] = article
        token = response.get('next_page_token')
        if not token:
            break
        if token in seen:
            raise ValueError('Provider repeated a news pagination token; snapshot is incomplete.')
        seen.add(token)
        if pages % 20 == 0:
            print(f'News: downloaded {pages} pages, {len(articles)} unique articles.', flush=True)
    if not articles:
        raise ValueError('No news returned; snapshot is incomplete. Check access and date coverage.')
    (directory / 'articles.json').write_text(json.dumps(list(articles.values())))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob('*.json'))}
    metadata = {'source': 'alpaca_news', 'start': first.isoformat(), 'end_exclusive': last.isoformat(),
                'symbols': list(symbols), 'articles': len(articles), 'pages': pages, 'sha256': hashes,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'availability': 'max(created_at, updated_at) + 15 minutes; latest archived text, not original revisions'}
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def load_news(directory, symbols, start, end):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata.get('source') != 'alpaca_news' or not set(symbols).issubset(metadata['symbols']):
        raise ValueError('News snapshot source/symbol coverage does not match the experiment.')
    first = pd.Timestamp(start).tz_localize('America/New_York').tz_convert('UTC')
    last = (pd.Timestamp(end).tz_localize('America/New_York') + pd.DateOffset(days=1)).tz_convert('UTC')
    if pd.Timestamp(metadata['start']) > first or pd.Timestamp(metadata['end_exclusive']) < last:
        raise ValueError('News snapshot must cover the entire price interval.')
    if 'articles.json' not in metadata['sha256']:
        raise ValueError('Missing news integrity record.')
    for name, digest in metadata['sha256'].items():
        if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('News snapshot integrity check failed.')
    return json.loads((directory / 'articles.json').read_text()), metadata


def add_sentiment(features, articles, scorer=None):
    if scorer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            raise ValueError('Install research extras: .venv/bin/python -m pip install -e ".[research]"') from None
        scorer = SentimentIntensityAnalyzer().polarity_scores
    dates = pd.DatetimeIndex(sorted(features.date.unique()))
    cal = calendar(dates.min(), dates.max())
    closes = pd.DatetimeIndex([cal.session_close(d) for d in dates])
    first_window_start = cal.session_close(cal.previous_session(dates[0]))
    records, seen = [], set()
    for article in articles:
        key = str(article['id'])
        if key in seen:
            raise ValueError('Duplicate news article ID.')
        seen.add(key)
        created, updated = pd.Timestamp(article['created_at']), pd.Timestamp(article['updated_at'])
        if pd.isna(created) or pd.isna(updated) or created.tzinfo is None or updated.tzinfo is None:
            raise ValueError('News timestamps must be valid and timezone-aware.')
        available = max(created, updated) + pd.Timedelta(minutes=15)
        # Each item is used once, at the first close after it is available.
        pos = closes.searchsorted(available, side='left')
        if pos == len(dates):
            continue
        # Avoid folding arbitrary old news into the first dataset session.
        if available <= first_window_start:
            continue
        score = float(scorer(article.get('headline', ''))['compound'])
        if not np.isfinite(score) or not -1 <= score <= 1:
            raise ValueError('Invalid sentiment score.')
        for symbol in set(article['symbols']):
            records.append({'date': dates[pos], 'symbol': symbol, 'news_sentiment': score})
    out = features.copy()
    if records:
        daily = pd.DataFrame(records).groupby(['date', 'symbol']).news_sentiment.agg(['mean', 'size'])
        daily.columns = NEWS_FEATURES
        out = out.merge(daily, on=['date', 'symbol'], how='left', validate='one_to_one')
    else:
        out[NEWS_FEATURES] = 0.0
    out[NEWS_FEATURES] = out[NEWS_FEATURES].fillna(0.0)
    out['news_count'] = out.news_count.astype('int64')
    # Count distinguishes no article from a neutral article. No price data is filled.
    return out
