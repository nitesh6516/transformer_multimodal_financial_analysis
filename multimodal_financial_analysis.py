"""
Transformer-Based Multimodal Financial Analysis for Trading Entry and Exit
Prediction with Flask deployment.

Run locally:
    python multimodal_financial_analysis.py

On this Windows workspace, the Microsoft Store python alias is broken. Use:
    py multimodal_financial_analysis.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import pickle
import random
import sqlite3
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)


MISSING_DEPENDENCIES: Dict[str, str] = {}


def _missing(import_name: str, pip_name: Optional[str] = None) -> None:
    MISSING_DEPENDENCIES[import_name] = pip_name or import_name


try:
    import numpy as np
except Exception:
    np = None  # type: ignore
    _missing("numpy")

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore
    _missing("pandas")

try:
    import requests
except Exception:
    requests = None  # type: ignore
    _missing("requests")

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore
    _missing("yfinance")

try:
    import ta
except Exception:
    ta = None  # type: ignore
    _missing("ta")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    px = None  # type: ignore
    go = None  # type: ignore
    make_subplots = None  # type: ignore
    _missing("plotly")

try:
    from flask import Flask, jsonify, render_template_string, request
except Exception:
    Flask = None  # type: ignore
    jsonify = None  # type: ignore
    render_template_string = None  # type: ignore
    request = None  # type: ignore
    _missing("flask")

try:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        auc,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_curve,
    )
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.preprocessing import RobustScaler
    from sklearn.utils.class_weight import compute_class_weight
except Exception:
    DummyClassifier = None  # type: ignore
    RandomForestClassifier = None  # type: ignore
    accuracy_score = None  # type: ignore
    auc = None  # type: ignore
    classification_report = None  # type: ignore
    confusion_matrix = None  # type: ignore
    f1_score = None  # type: ignore
    roc_curve = None  # type: ignore
    GridSearchCV = None  # type: ignore
    TimeSeriesSplit = None  # type: ignore
    RobustScaler = None  # type: ignore
    compute_class_weight = None  # type: ignore
    _missing("sklearn", "scikit-learn")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore
    _missing("torch")

TorchModuleBase = nn.Module if nn is not None else object

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None  # type: ignore
    _missing("vaderSentiment")

try:
    from loguru import logger as loguru_logger
except Exception:
    loguru_logger = None  # type: ignore

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline
except Exception:
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    hf_pipeline = None  # type: ignore


OPTIONAL_DEPENDENCIES = {"transformers", "loguru"}
LABEL_ORDER = [-1, 0, 1]
LABEL_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_LABEL = {0: -1, 1: 0, 2: 1}
CLASS_NAMES = ["SELL", "HOLD", "BUY"]
FOREX_CURRENCIES = {
    "USD",
    "EUR",
    "JPY",
    "GBP",
    "AUD",
    "CAD",
    "CHF",
    "NZD",
    "CNY",
    "HKD",
    "SGD",
}


def ensure_dependencies() -> None:
    required_missing = {
        name: pip_name
        for name, pip_name in MISSING_DEPENDENCIES.items()
        if name not in OPTIONAL_DEPENDENCIES
    }
    if not required_missing:
        return
    packages = " ".join(dict.fromkeys(required_missing.values()))
    message = (
        "Missing required Python packages:\n"
        f"  {', '.join(sorted(required_missing.values()))}\n\n"
        "Install them with:\n"
        f"  py -m pip install {packages}\n\n"
        "Then rerun:\n"
        "  py multimodal_financial_analysis.py\n"
    )
    raise RuntimeError(message)


def setup_logging(log_file: str = "financial_analysis.log"):
    if loguru_logger is not None:
        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level="INFO", enqueue=False)
        loguru_logger.add(log_file, level="INFO", rotation="5 MB", retention=3)
        return loguru_logger

    logger = logging.getLogger("multimodal_financial_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if np is not None:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if pd is not None and isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


def is_forex_symbol(symbol: str) -> bool:
    normalized = symbol.upper().replace("/", "").replace("_", "").replace("=X", "")
    if len(normalized) != 6:
        return False
    return normalized[:3] in FOREX_CURRENCIES and normalized[3:] in FOREX_CURRENCIES


def normalize_symbol_for_provider(symbol: str, provider: str) -> str:
    raw = symbol.upper().strip()
    if not is_forex_symbol(raw):
        return raw

    pair = raw.replace("/", "").replace("_", "").replace("=X", "")
    base, quote = pair[:3], pair[3:]
    if provider == "yfinance":
        return f"{base}{quote}=X"
    if provider == "twelvedata":
        return f"{base}/{quote}"
    if provider == "finnhub":
        return f"OANDA:{base}_{quote}"
    if provider == "polygon":
        return f"C:{base}{quote}"
    return f"{base}{quote}"


def date_string(value: Any) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


@dataclass
class PipelineConfig:
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "TSLA"])
    start_date: str = "2020-01-01"
    end_date: str = "2026-05-01"
    model_type: str = "transformer"
    sentiment_mode: str = "random"
    initial_capital: float = 10000.0
    position_size_pct: float = 0.10
    stop_loss: float = 0.02
    take_profit: float = 0.04
    sequence_length: int = 20
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.0001
    plotly_studio_mode: bool = True

    label_mode: str = "dynamic"
    fixed_threshold: float = 0.01
    forward_days: int = 5
    rolling_vol_window: int = 20
    probability_threshold: float = 0.60
    cache_path: str = "financial_data_cache.db"
    artifacts_dir: str = "."
    random_seed: int = 42
    min_training_rows: int = 80
    rf_cv_splits: int = 5
    rf_n_jobs: int = -1
    lstm_hidden_size: int = 96
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    transformer_d_model: int = 96
    transformer_heads: int = 4
    transformer_layers: int = 2
    fusion_dim: int = 96
    dropout: float = 0.20
    early_stopping_patience: int = 7
    finbert_local_only: bool = True
    allow_offline_demo_data: bool = True

    alpha_vantage_api_key: Optional[str] = None
    twelve_data_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    tiingo_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PipelineConfig":
        payload = dict(raw or {})
        env_keys = {
            "alpha_vantage_api_key": "ALPHA_VANTAGE_API_KEY",
            "twelve_data_api_key": "TWELVE_DATA_API_KEY",
            "finnhub_api_key": "FINNHUB_API_KEY",
            "tiingo_api_key": "TIINGO_API_KEY",
            "polygon_api_key": "POLYGON_API_KEY",
        }
        for field_name, env_name in env_keys.items():
            payload[field_name] = payload.get(field_name) or os.getenv(env_name)
        config = cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})
        config.symbols = [str(symbol).upper().strip() for symbol in config.symbols if str(symbol).strip()]
        config.model_type = config.model_type.lower().strip()
        config.sentiment_mode = config.sentiment_mode.lower().strip()
        config.label_mode = config.label_mode.lower().strip()
        if config.model_type not in {"random_forest", "lstm", "transformer"}:
            raise ValueError("model_type must be one of: random_forest, lstm, transformer")
        if config.sentiment_mode not in {"random", "finbert", "vader"}:
            raise ValueError("sentiment_mode must be one of: random, finbert, vader")
        if config.label_mode not in {"dynamic", "fixed"}:
            raise ValueError("label_mode must be one of: dynamic, fixed")
        if not config.symbols:
            raise ValueError("At least one symbol is required")
        if pd.to_datetime(config.start_date) >= pd.to_datetime(config.end_date):
            raise ValueError("start_date must be earlier than end_date")
        return config


class DataAggregator:
    """Fetches OHLCV data from free sources with SQLite caching and fallbacks."""

    REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Adjusted Close", "Symbol", "Source"]
    PROVIDER_PRIORITY = {
        "yfinance": 0,
        "alpha_vantage": 1,
        "twelvedata": 2,
        "finnhub": 3,
        "tiingo": 4,
        "polygon": 5,
        "offline_demo": 99,
    }

    def __init__(self, config: PipelineConfig, logger: Any):
        self.config = config
        self.logger = logger
        self.cache_path = Path(config.cache_path)
        self._init_cache()

    def _init_cache(self) -> None:
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS price_cache (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    adjusted_close REAL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY(symbol, date, source)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price_cache_symbol_date ON price_cache(symbol, date)")

    def fetch(self, symbols: List[str], start_date: str, end_date: str) -> "pd.DataFrame":
        frames = []
        for symbol in symbols:
            symbol_df = self.fetch_symbol(symbol, start_date, end_date)
            if not symbol_df.empty:
                frames.append(symbol_df)
        if not frames:
            raise RuntimeError("No usable market data was fetched from cache or configured providers.")
        data = pd.concat(frames, ignore_index=True)
        data = data.sort_values(["Symbol", "Date"]).reset_index(drop=True)
        return data

    def fetch_symbol(self, symbol: str, start_date: str, end_date: str) -> "pd.DataFrame":
        cached = self._read_cache(symbol, start_date, end_date)
        expected_days = max(len(pd.bdate_range(start_date, end_date)), 1)
        cached_coverage = cached["Date"].nunique() / expected_days if not cached.empty else 0.0

        if cached_coverage >= 0.92:
            self.logger.info(f"{symbol}: cache hit with {cached_coverage:.1%} business-day coverage")
            return self._merge_sources(cached, symbol, start_date, end_date)

        self.logger.info(f"{symbol}: cache coverage {cached_coverage:.1%}; fetching provider fallbacks")
        fetched_frames = []
        provider_plan = [
            ("yfinance", self._fetch_yfinance, True),
            ("alpha_vantage", self._fetch_alpha_vantage, bool(self.config.alpha_vantage_api_key)),
            ("twelvedata", self._fetch_twelvedata, bool(self.config.twelve_data_api_key)),
            ("finnhub", self._fetch_finnhub, bool(self.config.finnhub_api_key)),
            ("tiingo", self._fetch_tiingo, bool(self.config.tiingo_api_key)),
            ("polygon", self._fetch_polygon, bool(self.config.polygon_api_key)),
        ]

        for provider, fetcher, enabled in provider_plan:
            if not enabled:
                self.logger.info(f"{symbol}: skipping {provider}; API key not configured")
                continue
            try:
                frame = fetcher(symbol, start_date, end_date)
                if frame is None or frame.empty:
                    self.logger.warning(f"{symbol}: {provider} returned no rows")
                    continue
                frame = self._normalize_frame(frame, symbol, provider, start_date, end_date)
                self._write_cache(frame)
                fetched_frames.append(frame)
                self.logger.info(f"{symbol}: {provider} returned {len(frame)} rows")
            except Exception as exc:
                self.logger.warning(f"{symbol}: {provider} failed: {exc}")
                continue
            time.sleep(0.15)

        refreshed_cache = self._read_cache(symbol, start_date, end_date)
        frames = [frame for frame in [cached, refreshed_cache, *fetched_frames] if frame is not None and not frame.empty]
        if not frames:
            if self.config.allow_offline_demo_data:
                self.logger.warning(
                    f"{symbol}: no live provider data available; generating deterministic offline demo bars"
                )
                demo = self._generate_offline_demo_data(symbol, start_date, end_date)
                self._write_cache(demo)
                return demo
            raise RuntimeError(f"{symbol}: no usable data after all provider attempts")
        merged = self._merge_sources(pd.concat(frames, ignore_index=True), symbol, start_date, end_date)
        if merged.empty:
            raise RuntimeError(f"{symbol}: provider data existed but could not be normalized")
        return merged

    def _read_cache(self, symbol: str, start_date: str, end_date: str) -> "pd.DataFrame":
        with sqlite3.connect(self.cache_path) as conn:
            query = """
                SELECT
                    date AS Date,
                    open AS Open,
                    high AS High,
                    low AS Low,
                    close AS Close,
                    volume AS Volume,
                    adjusted_close AS "Adjusted Close",
                    symbol AS Symbol,
                    source AS Source
                FROM price_cache
                WHERE symbol = ? AND date BETWEEN ? AND ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol.upper(), start_date, end_date))
        if df.empty:
            return df
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def _write_cache(self, frame: "pd.DataFrame") -> None:
        if frame.empty:
            return
        rows = []
        fetched_at = datetime.utcnow().isoformat(timespec="seconds")
        for _, row in frame.iterrows():
            rows.append(
                (
                    str(row["Symbol"]).upper(),
                    date_string(row["Date"]),
                    str(row["Source"]),
                    safe_float(row.get("Open"), None),
                    safe_float(row.get("High"), None),
                    safe_float(row.get("Low"), None),
                    safe_float(row.get("Close"), None),
                    safe_float(row.get("Volume"), 0.0),
                    safe_float(row.get("Adjusted Close"), safe_float(row.get("Close"), 0.0)),
                    fetched_at,
                )
            )
        with sqlite3.connect(self.cache_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO price_cache
                (symbol, date, source, open, high, low, close, volume, adjusted_close, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def _merge_sources(self, frame: "pd.DataFrame", symbol: str, start_date: str, end_date: str) -> "pd.DataFrame":
        if frame.empty:
            return frame
        df = self._normalize_frame(frame, symbol, "cache", start_date, end_date)
        if df.empty:
            return df
        df["_priority"] = df["Source"].map(self.PROVIDER_PRIORITY).fillna(99).astype(int)
        merged_rows = []
        numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Adjusted Close"]
        for date_value, group in df.sort_values(["Date", "_priority"]).groupby("Date", sort=True):
            best = group.iloc[0].copy()
            for col in numeric_cols:
                if pd.isna(best[col]):
                    replacements = group[col].dropna()
                    if not replacements.empty:
                        best[col] = replacements.iloc[0]
            best["Source"] = "+".join(sorted(group["Source"].dropna().unique(), key=lambda x: self.PROVIDER_PRIORITY.get(x, 99)))
            best["Date"] = date_value
            best["Symbol"] = symbol.upper()
            merged_rows.append(best[self.REQUIRED_COLUMNS])
        merged = pd.DataFrame(merged_rows)
        merged = merged.dropna(subset=["Date", "Open", "High", "Low", "Close", "Adjusted Close"])
        merged["Volume"] = merged["Volume"].fillna(0.0)
        return merged.sort_values("Date").reset_index(drop=True)

    def _normalize_frame(
        self, frame: "pd.DataFrame", symbol: str, source: str, start_date: str, end_date: str
    ) -> "pd.DataFrame":
        if frame is None or frame.empty:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        df = frame.copy()
        rename_map = {
            "date": "Date",
            "datetime": "Date",
            "timestamp": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adjClose": "Adjusted Close",
            "adj_close": "Adjusted Close",
            "adjusted_close": "Adjusted Close",
            "adjusted close": "Adjusted Close",
            "Adj Close": "Adjusted Close",
        }
        df = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})
        if "Date" not in df.columns and df.index.name is not None:
            df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
        elif "Date" not in df.columns:
            df = df.reset_index().rename(columns={"index": "Date"})

        if "Adjusted Close" not in df.columns and "Close" in df.columns:
            df["Adjusted Close"] = df["Close"]
        if "Volume" not in df.columns:
            df["Volume"] = 0.0

        for col in ["Open", "High", "Low", "Close", "Volume", "Adjusted Close"]:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
        df["Symbol"] = symbol.upper()
        if "Source" not in df.columns or source != "cache":
            df["Source"] = source
        df = df[self.REQUIRED_COLUMNS].dropna(subset=["Date"])
        start = pd.to_datetime(start_date).normalize()
        end = pd.to_datetime(end_date).normalize()
        df = df[(df["Date"] >= start) & (df["Date"] <= end)]
        df = df.drop_duplicates(subset=["Date", "Symbol", "Source"], keep="last")
        return df.sort_values("Date").reset_index(drop=True)

    def _fetch_yfinance(self, symbol: str, start_date: str, end_date: str) -> "pd.DataFrame":
        provider_symbol = normalize_symbol_for_provider(symbol, "yfinance")
        end_exclusive = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download(
            provider_symbol,
            start=start_date,
            end=end_exclusive,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        if "Adj Close" not in data.columns and "Close" in data.columns:
            data["Adj Close"] = data["Close"]
        return data.rename(columns={"Adj Close": "Adjusted Close"})

    def _fetch_alpha_vantage(self, symbol: str, start_date: str, end_date: str) -> "pd.DataFrame":
        key = self.config.alpha_vantage_api_key
        if not key:
            return pd.DataFrame()

        if is_forex_symbol(symbol):
            provider_symbol = normalize_symbol_for_provider(symbol, "alpha_vantage")
            params = {
                "function": "FX_DAILY",
                "from_symbol": provider_symbol[:3],
                "to_symbol": provider_symbol[3:],
                "outputsize": "full",
                "apikey": key,
            }
            payload = self._request_json("https://www.alphavantage.co/query", params=params)
            series = payload.get("Time Series FX (Daily)", {})
            rows = []
            for day, values in series.items():
                rows.append(
                    {
                        "Date": day,
                        "Open": values.get("1. open"),
                        "High": values.get("2. high"),
                        "Low": values.get("3. low"),
                        "Close": values.get("4. close"),
                        "Volume": 0.0,
                        "Adjusted Close": values.get("4. close"),
                    }
                )
            return pd.DataFrame(rows)

        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": normalize_symbol_for_provider(symbol, "alpha_vantage"),
            "outputsize": "full",
            "apikey": key,
        }
        payload = self._request_json("https://www.alphavantage.co/query", params=params)
        series = payload.get("Time Series (Daily)", {})
        rows = []
        for day, values in series.items():
            rows.append(
                {
                    "Date": day,
                    "Open": values.get("1. open"),
                    "High": values.get("2. high"),
                    "Low": values.get("3. low"),
                    "Close": values.get("4. close"),
                    "Volume": values.get("6. volume") or values.get("5. volume"),
                    "Adjusted Close": values.get("5. adjusted close") or values.get("4. close"),
                }
            )
        return pd.DataFrame(rows)

    def _fetch_twelvedata(self, symbol: str, start_date: str, end_date: str) -> "pd.DataFrame":
        key = self.config.twelve_data_api_key
        if not key:
            return pd.DataFrame()
        params = {
            "symbol": normalize_symbol_for_provider(symbol, "twelvedata"),
            "interval": "1day",
            "start_date": start_date,
            "end_date": end_date,
            "outputsize": 5000,
            "order": "ASC",
            "adjust": "all",
            "apikey": key,
        }
        payload = self._request_json("https://api.twelvedata.com/time_series", params=params)
        if payload.get("status") == "error":
            raise RuntimeError(payload.get("message", "Twelve Data error"))
        rows = payload.get("values", [])
        return pd.DataFrame(rows).rename(columns={"datetime": "Date"})

    def _fetch_finnhub(self, symbol: str, start_date: str, end_date: str) -> "pd.DataFrame":
        key = self.config.finnhub_api_key
        if not key:
            return pd.DataFrame()
        start_epoch = int(pd.to_datetime(start_date).timestamp())
        end_epoch = int((pd.to_datetime(end_date) + pd.Timedelta(days=1)).timestamp())
        endpoint = "forex/candle" if is_forex_symbol(symbol) else "stock/candle"
        params = {
            "symbol": normalize_symbol_for_provider(symbol, "finnhub"),
            "resolution": "D",
            "from": start_epoch,
            "to": end_epoch,
            "token": key,
        }
        payload = self._request_json(f"https://finnhub.io/api/v1/{endpoint}", params=params)
        if payload.get("s") != "ok":
            raise RuntimeError(payload.get("s", "Finnhub error"))
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(payload.get("t", []), unit="s"),
                "Open": payload.get("o", []),
                "High": payload.get("h", []),
                "Low": payload.get("l", []),
                "Close": payload.get("c", []),
                "Volume": payload.get("v", [0.0] * len(payload.get("c", []))),
                "Adjusted Close": payload.get("c", []),
            }
        )

    def _fetch_tiingo(self, symbol: str, start_date: str, end_date: str) -> "pd.DataFrame":
        key = self.config.tiingo_api_key
        if not key or is_forex_symbol(symbol):
            return pd.DataFrame()
        provider_symbol = normalize_symbol_for_provider(symbol, "tiingo").lower()
        url = f"https://api.tiingo.com/tiingo/daily/{provider_symbol}/prices"
        params = {"startDate": start_date, "endDate": end_date, "format": "json"}
        headers = {"Authorization": f"Token {key}"}
        payload = self._request_json(url, params=params, headers=headers)
        return pd.DataFrame(payload).rename(columns={"adjClose": "Adjusted Close"})

    def _fetch_polygon(self, symbol: str, start_date: str, end_date: str) -> "pd.DataFrame":
        key = self.config.polygon_api_key
        if not key:
            return pd.DataFrame()
        provider_symbol = normalize_symbol_for_provider(symbol, "polygon")
        url = f"https://api.polygon.io/v2/aggs/ticker/{provider_symbol}/range/1/day/{start_date}/{end_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key}
        payload = self._request_json(url, params=params)
        if payload.get("status") not in {"OK", "DELAYED"} and not payload.get("results"):
            raise RuntimeError(payload.get("error", "Polygon error"))
        rows = []
        for item in payload.get("results", []):
            rows.append(
                {
                    "Date": pd.to_datetime(item.get("t"), unit="ms"),
                    "Open": item.get("o"),
                    "High": item.get("h"),
                    "Low": item.get("l"),
                    "Close": item.get("c"),
                    "Volume": item.get("v", 0.0),
                    "Adjusted Close": item.get("c"),
                }
            )
        return pd.DataFrame(rows)

    def _generate_offline_demo_data(self, symbol: str, start_date: str, end_date: str) -> "pd.DataFrame":
        dates = pd.bdate_range(start=start_date, end=end_date)
        if len(dates) == 0:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        seed = int(hashlib.sha256(f"{symbol}|offline_demo".encode("utf-8")).hexdigest()[:16], 16) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        base_prices = {"AAPL": 95.0, "MSFT": 155.0, "TSLA": 85.0, "SPY": 305.0, "QQQ": 205.0}
        base_price = base_prices.get(symbol.upper(), 75.0 + (seed % 120))
        drift = 0.00025 + ((seed % 11) - 5) * 0.00002
        vol = 0.018 + (seed % 9) * 0.001
        market_cycle = np.sin(np.linspace(0, 10 * math.pi, len(dates))) * 0.0025
        shocks = rng.normal(drift, vol, len(dates)) + market_cycle
        close = base_price * np.exp(np.cumsum(shocks))
        overnight = rng.normal(0.0, vol * 0.25, len(dates))
        open_price = close * np.exp(overnight)
        spread = np.abs(rng.normal(vol * 0.65, vol * 0.25, len(dates)))
        high = np.maximum(open_price, close) * (1.0 + spread)
        low = np.minimum(open_price, close) * (1.0 - spread)
        volume_base = 25_000_000 + (seed % 40_000_000)
        volume = volume_base * rng.lognormal(mean=0.0, sigma=0.35, size=len(dates))
        adjusted = close.copy()

        return pd.DataFrame(
            {
                "Date": dates,
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
                "Adjusted Close": adjusted,
                "Symbol": symbol.upper(),
                "Source": "offline_demo",
            }
        )

    def _request_json(self, url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if requests is None:
            raise RuntimeError("requests is not installed")
        response = requests.get(url, params=params, headers=headers, timeout=20)
        if response.status_code in {401, 403, 429}:
            raise RuntimeError(f"HTTP {response.status_code}: provider rejected or rate-limited the request")
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            rate_text = " ".join(str(payload.get(k, "")) for k in ["Note", "Information", "Error Message", "message"])
            if "rate" in rate_text.lower() or "premium" in rate_text.lower() or "invalid" in rate_text.lower():
                raise RuntimeError(rate_text)
        return payload


class FeatureEngineer:
    """Leak-aware feature computation for daily financial bars."""

    def __init__(self, logger: Any):
        self.logger = logger
        self.numeric_feature_columns: List[str] = []

    def transform(self, frame: "pd.DataFrame") -> Tuple["pd.DataFrame", List[str]]:
        frames = []
        for symbol, group in frame.groupby("Symbol", sort=False):
            engineered = self._transform_symbol(group.sort_values("Date").reset_index(drop=True))
            frames.append(engineered)
            self.logger.info(f"{symbol}: engineered {len(engineered.columns)} columns")
        result = pd.concat(frames, ignore_index=True).sort_values(["Symbol", "Date"]).reset_index(drop=True)
        excluded = {"Date", "Symbol", "Source", "label", "forward_return", "split"}
        numeric_feature_columns = [
            col
            for col in result.columns
            if col not in excluded
            and col not in {"Open", "High", "Low", "Close", "Volume", "Adjusted Close"}
            and pd.api.types.is_numeric_dtype(result[col])
        ]
        self.numeric_feature_columns = numeric_feature_columns
        return result, numeric_feature_columns

    def _transform_symbol(self, group: "pd.DataFrame") -> "pd.DataFrame":
        g = group.copy()
        for col in ["Open", "High", "Low", "Close", "Volume", "Adjusted Close"]:
            g[col] = pd.to_numeric(g[col], errors="coerce")

        close = g["Adjusted Close"].astype(float)
        raw_close = g["Close"].astype(float)
        high = g["High"].astype(float)
        low = g["Low"].astype(float)
        volume = g["Volume"].fillna(0.0).astype(float)

        g["log_return"] = np.log(close / close.shift(1))
        g["return_1d"] = close.pct_change()
        g["rolling_volatility_20"] = g["log_return"].rolling(20, min_periods=5).std()
        g["rolling_volatility_60"] = g["log_return"].rolling(60, min_periods=10).std()
        g["momentum_5"] = close / close.shift(5) - 1.0
        g["momentum_10"] = close / close.shift(10) - 1.0
        g["momentum_20"] = close / close.shift(20) - 1.0
        typical_price = (high + low + raw_close) / 3.0
        volume_sum = volume.rolling(20, min_periods=5).sum().replace(0.0, np.nan)
        g["vwap_20"] = (typical_price * volume).rolling(20, min_periods=5).sum() / volume_sum
        g["obv"] = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
        g["atr_14"] = self._safe_series(lambda: ta.volatility.AverageTrueRange(high, low, raw_close, window=14).average_true_range(), len(g))

        for window in [7, 14, 21]:
            g[f"rsi_{window}"] = self._safe_series(
                lambda window=window: ta.momentum.RSIIndicator(raw_close, window=window).rsi(), len(g)
            )

        macd = self._safe_indicator(lambda: ta.trend.MACD(raw_close, window_slow=26, window_fast=12, window_sign=9))
        if macd is not None:
            g["macd"] = macd.macd()
            g["macd_signal"] = macd.macd_signal()
            g["macd_hist"] = macd.macd_diff()
        else:
            g[["macd", "macd_signal", "macd_hist"]] = np.nan

        for window in [9, 21, 50, 200]:
            g[f"ema_{window}"] = self._safe_series(lambda window=window: ta.trend.EMAIndicator(raw_close, window=window).ema_indicator(), len(g))
        for window in [20, 50, 200]:
            g[f"sma_{window}"] = self._safe_series(lambda window=window: ta.trend.SMAIndicator(raw_close, window=window).sma_indicator(), len(g))

        bb = self._safe_indicator(lambda: ta.volatility.BollingerBands(raw_close, window=20, window_dev=2))
        if bb is not None:
            g["bb_high"] = bb.bollinger_hband()
            g["bb_mid"] = bb.bollinger_mavg()
            g["bb_low"] = bb.bollinger_lband()
            g["bb_width"] = bb.bollinger_wband()
        else:
            g[["bb_high", "bb_mid", "bb_low", "bb_width"]] = np.nan

        stoch = self._safe_indicator(lambda: ta.momentum.StochasticOscillator(high, low, raw_close, window=14, smooth_window=3))
        if stoch is not None:
            g["stoch_k"] = stoch.stoch()
            g["stoch_d"] = stoch.stoch_signal()
        else:
            g[["stoch_k", "stoch_d"]] = np.nan

        g["adx_14"] = self._safe_series(lambda: ta.trend.ADXIndicator(high, low, raw_close, window=14).adx(), len(g))
        g["cci_20"] = self._safe_series(lambda: ta.trend.CCIIndicator(high, low, raw_close, window=20).cci(), len(g))

        g = self._add_multi_timeframe(g)
        return g.replace([np.inf, -np.inf], np.nan)

    def _safe_indicator(self, builder):
        try:
            return builder()
        except Exception:
            return None

    def _safe_series(self, builder, length: int) -> "pd.Series":
        try:
            series = builder()
            return pd.Series(series).reset_index(drop=True)
        except Exception:
            return pd.Series([np.nan] * length)

    def _add_multi_timeframe(self, group: "pd.DataFrame") -> "pd.DataFrame":
        g = group.sort_values("Date").copy()
        indexed = g.set_index("Date")

        weekly_close = indexed["Adjusted Close"].resample("W-FRI").last()
        weekly = pd.DataFrame(
            {
                "Date": weekly_close.index,
                "weekly_return_lag": weekly_close.pct_change().shift(1),
                "weekly_volatility_lag": indexed["log_return"].resample("W-FRI").std().shift(1),
            }
        ).reset_index(drop=True).dropna(how="all", subset=["weekly_return_lag", "weekly_volatility_lag"])

        monthly_close = indexed["Adjusted Close"].resample("ME").last()
        monthly = pd.DataFrame(
            {
                "Date": monthly_close.index,
                "monthly_return_lag": monthly_close.pct_change().shift(1),
                "monthly_volatility_lag": indexed["log_return"].resample("ME").std().shift(1),
            }
        ).reset_index(drop=True).dropna(how="all", subset=["monthly_return_lag", "monthly_volatility_lag"])

        g = pd.merge_asof(g.sort_values("Date"), weekly.sort_values("Date"), on="Date", direction="backward")
        g = pd.merge_asof(g.sort_values("Date"), monthly.sort_values("Date"), on="Date", direction="backward")
        return g


class ClusteringAnalyzer:
    """Unsupervised learning component for clustering financial data and detecting patterns."""

    def __init__(self, config: PipelineConfig, logger: Any):
        self.config = config
        self.logger = logger
        try:
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            self.KMeans = KMeans
            self.DBSCAN = DBSCAN
            self.StandardScaler = StandardScaler
            self.PCA = PCA
            self.silhouette_score = silhouette_score
        except ImportError:
            self.logger.warning("scikit-learn not available for clustering")
            self.KMeans = None

    def perform_clustering(self, features: "pd.DataFrame", method: str = "kmeans", n_clusters: int = 3) -> Dict[str, Any]:
        """Perform clustering on the feature data."""
        if self.KMeans is None:
            return {"error": "Clustering unavailable - scikit-learn not installed"}

        # Select numerical features for clustering
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        data = features[numerical_cols].dropna()

        if data.empty:
            return {"error": "No numerical data available for clustering"}

        # Scale the data
        scaler = self.StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Optional: Reduce dimensionality with PCA
        if scaled_data.shape[1] > 10:
            pca = self.PCA(n_components=min(10, scaled_data.shape[1]))
            scaled_data = pca.fit_transform(scaled_data)
            explained_variance = pca.explained_variance_ratio_
        else:
            explained_variance = None

        # Perform clustering
        if method == "kmeans":
            cluster_model = self.KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = cluster_model.fit_predict(scaled_data)
            centroids = cluster_model.cluster_centers_
        elif method == "dbscan":
            cluster_model = self.DBSCAN(eps=0.5, min_samples=5)
            clusters = cluster_model.fit_predict(scaled_data)
            centroids = None  # DBSCAN doesn't have centroids
        else:
            return {"error": f"Unsupported clustering method: {method}"}

        # Calculate silhouette score if possible
        if len(set(clusters)) > 1 and len(set(clusters)) < len(clusters):
            try:
                silhouette_avg = self.silhouette_score(scaled_data, clusters)
            except:
                silhouette_avg = None
        else:
            silhouette_avg = None

        # Add cluster labels back to the dataframe
        result_df = data.copy()
        result_df['cluster'] = clusters

        return {
            "clusters": clusters.tolist(),
            "n_clusters": len(set(clusters)),
            "silhouette_score": silhouette_avg,
            "centroids": centroids.tolist() if centroids is not None else None,
            "explained_variance": explained_variance.tolist() if explained_variance is not None else None,
            "method": method,
            "data_shape": scaled_data.shape,
            "clustered_data": result_df.to_dict('records')
        }

    def detect_anomalies(self, features: "pd.DataFrame", contamination: float = 0.1) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            return {"error": "IsolationForest unavailable"}

        numerical_cols = features.select_dtypes(include=[np.number]).columns
        data = features[numerical_cols].dropna()

        if data.empty:
            return {"error": "No numerical data available for anomaly detection"}

        scaler = self.StandardScaler()
        scaled_data = scaler.fit_transform(data)

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_scores = iso_forest.fit_predict(scaled_data)

        # -1 for anomalies, 1 for normal
        anomalies = (anomaly_scores == -1)

        result_df = data.copy()
        result_df['anomaly_score'] = anomaly_scores
        result_df['is_anomaly'] = anomalies

        return {
            "anomaly_scores": anomaly_scores.tolist(),
            "n_anomalies": int(anomalies.sum()),
            "contamination": contamination,
            "anomalous_data": result_df[anomalies].to_dict('records')
        }


class SentimentFeatureGenerator:
    """Simulated sentiment now, FinBERT-ready later with a config switch."""

    SENTIMENT_COLUMNS = ["sentiment_score", "sentiment_positive", "sentiment_neutral", "sentiment_negative"]

    def __init__(self, config: PipelineConfig, logger: Any):
        self.config = config
        self.logger = logger
        self.vader = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None
        self.finbert = None
        if self.config.sentiment_mode == "finbert":
            self.finbert = self._load_finbert()
            if self.finbert is None:
                self.logger.warning("FinBERT is unavailable; falling back to deterministic random sentiment")

    def transform(self, frame: "pd.DataFrame") -> Tuple["pd.DataFrame", List[str]]:
        rows = []
        for _, row in frame.iterrows():
            if self.config.sentiment_mode == "vader" and self.vader is not None:
                rows.append(self._vader_features(row))
            elif self.config.sentiment_mode == "finbert" and self.finbert is not None:
                rows.append(self._finbert_features(row))
            else:
                rows.append(self._random_features(row))
        sentiment = pd.DataFrame(rows, index=frame.index)
        result = pd.concat([frame.reset_index(drop=True), sentiment.reset_index(drop=True)], axis=1)
        return result, self.SENTIMENT_COLUMNS

    def _row_rng(self, row: "pd.Series") -> "np.random.Generator":
        key = f"{self.config.random_seed}|{row.get('Symbol')}|{date_string(row.get('Date'))}"
        seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % (2**32 - 1)
        return np.random.default_rng(seed)

    def _random_features(self, row: "pd.Series") -> Dict[str, float]:
        rng = self._row_rng(row)
        score = float(np.tanh(rng.normal(0.0, 0.55)))
        positive = max(score, 0.0)
        negative = max(-score, 0.0)
        neutral = max(0.0, 1.0 - positive - negative)
        total = positive + neutral + negative
        return {
            "sentiment_score": score,
            "sentiment_positive": positive / total,
            "sentiment_neutral": neutral / total,
            "sentiment_negative": negative / total,
        }

    def _simulated_text(self, row: "pd.Series") -> str:
        score = self._random_features(row)["sentiment_score"]
        symbol = row.get("Symbol", "asset")
        if score > 0.35:
            return f"{symbol} investors see strong demand, upbeat guidance, resilient margins, and improving liquidity."
        if score < -0.35:
            return f"{symbol} faces weak demand, cautious guidance, margin pressure, and uncertain market liquidity."
        return f"{symbol} trading is mixed as investors weigh macro uncertainty, valuation, and earnings expectations."

    def _vader_features(self, row: "pd.Series") -> Dict[str, float]:
        if self.vader is None:
            return self._random_features(row)
        scores = self.vader.polarity_scores(self._simulated_text(row))
        return {
            "sentiment_score": float(scores.get("compound", 0.0)),
            "sentiment_positive": float(scores.get("pos", 0.0)),
            "sentiment_neutral": float(scores.get("neu", 1.0)),
            "sentiment_negative": float(scores.get("neg", 0.0)),
        }

    def _load_finbert(self):
        if hf_pipeline is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", local_files_only=self.config.finbert_local_only)
            model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert", local_files_only=self.config.finbert_local_only
            )
            device = 0 if torch is not None and torch.cuda.is_available() else -1
            return hf_pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
        except Exception as exc:
            self.logger.warning(f"FinBERT load failed: {exc}")
            return None

    def _finbert_features(self, row: "pd.Series") -> Dict[str, float]:
        try:
            result = self.finbert(self._simulated_text(row), truncation=True)[0]
            label = str(result.get("label", "")).lower()
            confidence = float(result.get("score", 0.0))
            features = {"sentiment_positive": 0.0, "sentiment_neutral": 0.0, "sentiment_negative": 0.0}
            if "positive" in label:
                features["sentiment_positive"] = confidence
                features["sentiment_neutral"] = 1.0 - confidence
                score = confidence
            elif "negative" in label:
                features["sentiment_negative"] = confidence
                features["sentiment_neutral"] = 1.0 - confidence
                score = -confidence
            else:
                features["sentiment_neutral"] = confidence
                score = 0.0
            features["sentiment_score"] = score
            return features
        except Exception:
            return self._random_features(row)


class LabelGenerator:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def transform(self, frame: "pd.DataFrame") -> "pd.DataFrame":
        frames = []
        for _, group in frame.groupby("Symbol", sort=False):
            g = group.sort_values("Date").copy()
            forward = g["Adjusted Close"].shift(-self.config.forward_days) / g["Adjusted Close"] - 1.0
            if self.config.label_mode == "fixed":
                threshold = pd.Series(self.config.fixed_threshold, index=g.index)
            else:
                daily_vol = np.log(g["Adjusted Close"] / g["Adjusted Close"].shift(1)).rolling(
                    self.config.rolling_vol_window, min_periods=5
                ).std()
                threshold = 0.5 * daily_vol * math.sqrt(max(self.config.forward_days, 1))
                threshold = threshold.fillna(self.config.fixed_threshold)

            g["forward_return"] = forward
            g["label"] = np.select([forward > threshold, forward < -threshold], [1, -1], default=0)
            g.loc[forward.isna(), "label"] = np.nan
            frames.append(g)
        return pd.concat(frames, ignore_index=True).sort_values(["Symbol", "Date"]).reset_index(drop=True)


def create_model_columns(
    frame: "pd.DataFrame", numeric_features: List[str], sentiment_features: List[str]
) -> Tuple["pd.DataFrame", List[str], List[str], List[str]]:
    df = frame.copy()
    all_features = numeric_features + sentiment_features
    model_numeric = []
    model_sentiment = []
    for col in all_features:
        model_col = f"model_{col}"
        df[model_col] = df.groupby("Symbol")[col].shift(1)
        if col in sentiment_features:
            model_sentiment.append(model_col)
        else:
            model_numeric.append(model_col)
    model_features = model_numeric + model_sentiment
    df[model_features] = df[model_features].replace([np.inf, -np.inf], np.nan)
    return df, model_features, model_numeric, model_sentiment


def assign_temporal_splits(frame: "pd.DataFrame", config: PipelineConfig, feature_columns: List[str], logger: Any) -> "pd.DataFrame":
    frames = []
    for symbol, group in frame.groupby("Symbol", sort=False):
        g = group.sort_values("Date").copy()
        valid_mask = g[feature_columns + ["label"]].notna().all(axis=1)
        valid_index = list(g.index[valid_mask])
        n = len(valid_index)
        if n < config.min_training_rows:
            logger.warning(f"{symbol}: only {n} model-ready rows; results may be unstable")
        train_end = max(int(n * 0.70), 1)
        val_end = max(int(n * 0.85), train_end + 1)
        split = pd.Series("ignore", index=g.index)
        split.loc[valid_index[:train_end]] = "train"
        split.loc[valid_index[train_end:val_end]] = "val"
        split.loc[valid_index[val_end:]] = "test"
        g["split"] = split
        frames.append(g)
    result = pd.concat(frames, ignore_index=True).sort_values(["Symbol", "Date"]).reset_index(drop=True)
    result["_row_id"] = np.arange(len(result))
    ready = result[result["split"].isin(["train", "val", "test"])]
    if ready.empty:
        raise RuntimeError("No model-ready rows after feature shifting and label generation.")
    return result


def fit_scaler(
    frame: "pd.DataFrame", feature_columns: List[str], logger: Any
) -> Tuple[Any, "pd.DataFrame", List[str]]:
    df = frame.copy()
    train = df[df["split"] == "train"]
    if train.empty:
        raise RuntimeError("Training split is empty after temporal split.")
    scaler = RobustScaler()
    train_values = train[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler.fit(train_values)
    scaled_columns = [f"scaled_{col}" for col in feature_columns]
    scaled_values = scaler.transform(df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    df[scaled_columns] = scaled_values
    logger.info(f"Fitted RobustScaler on {len(train)} training rows and {len(feature_columns)} features")
    return scaler, df, scaled_columns


def compute_class_weights(y: Sequence[int]) -> Optional[Any]:
    if torch is None:
        return None
    labels = [int(v) for v in y]
    if not labels:
        return torch.ones(3, dtype=torch.float32)
    total = float(len(labels))
    counts = {label: labels.count(label) for label in LABEL_ORDER}
    weights = []
    for label in LABEL_ORDER:
        count = counts.get(label, 0)
        weights.append(total / (len(LABEL_ORDER) * count) if count else 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def align_probabilities(proba: "np.ndarray", classes: Sequence[int]) -> "np.ndarray":
    aligned = np.zeros((len(proba), 3), dtype=float)
    for idx, cls in enumerate(classes):
        if int(cls) in LABEL_TO_INDEX:
            aligned[:, LABEL_TO_INDEX[int(cls)]] = proba[:, idx]
    row_sums = aligned.sum(axis=1)
    zero_rows = row_sums <= 0
    aligned[zero_rows, 1] = 1.0
    row_sums = aligned.sum(axis=1)
    aligned = aligned / row_sums[:, None]
    return aligned


def compute_prediction_metrics(y_true: Sequence[int], y_pred: Sequence[int], proba: "np.ndarray") -> Dict[str, Any]:
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    if len(y_true) == 0:
        return {
            "accuracy": None,
            "macro_f1": None,
            "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "classification_report": {},
        }
    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, target_names=CLASS_NAMES, zero_division=0, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),
        "classification_report": report,
    }


class MultimodalFusion(TorchModuleBase):
    def __init__(self, numeric_dim: int, sentiment_dim: int, fusion_dim: int, dropout: float):
        super().__init__()
        self.numeric_dim = numeric_dim
        self.sentiment_dim = sentiment_dim
        input_dim = numeric_dim + sentiment_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, numeric_x, sentiment_x=None):
        if sentiment_x is None or self.sentiment_dim == 0:
            x = numeric_x
        else:
            x = torch.cat([numeric_x, sentiment_x], dim=-1)
        return self.net(x)


class LSTMClassifier(TorchModuleBase):
    def __init__(
        self,
        numeric_dim: int,
        sentiment_dim: int,
        fusion_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()
        self.fusion = MultimodalFusion(numeric_dim, sentiment_dim, fusion_dim, dropout)
        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size * directions, 3))

    def forward(self, numeric_x, sentiment_x=None):
        fused = self.fusion(numeric_x, sentiment_x)
        output, _ = self.lstm(fused)
        return self.head(output[:, -1, :])


class PositionalEncoding(TorchModuleBase):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class AttentionBlock(TorchModuleBase):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.last_attention = None

    def forward(self, x):
        attn_output, weights = self.attention(x, x, x, need_weights=True, average_attn_weights=False)
        self.last_attention = weights.detach()
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.feed_forward(x))
        return x


class TransformerClassifier(TorchModuleBase):
    def __init__(
        self,
        numeric_dim: int,
        sentiment_dim: int,
        fusion_dim: int,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.fusion = MultimodalFusion(numeric_dim, sentiment_dim, fusion_dim, dropout)
        self.input_projection = nn.Linear(fusion_dim, d_model)
        self.positional = PositionalEncoding(d_model, max_len=max_len + 16)
        self.blocks = nn.ModuleList([AttentionBlock(d_model, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 3))

    def forward(self, numeric_x, sentiment_x=None):
        x = self.fusion(numeric_x, sentiment_x)
        x = self.input_projection(x)
        x = self.positional(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, -1, :])

    def get_attention_matrix(self) -> Optional["np.ndarray"]:
        if not self.blocks or self.blocks[-1].last_attention is None:
            return None
        weights = self.blocks[-1].last_attention.detach().cpu().numpy()
        return weights.mean(axis=(0, 1))


class WindowedDataset(Dataset):
    def __init__(
        self,
        frame: "pd.DataFrame",
        numeric_columns: List[str],
        sentiment_columns: List[str],
        target_splits: Iterable[str],
        sequence_length: int,
    ):
        self.numeric_sequences = []
        self.sentiment_sequences = []
        self.targets = []
        self.row_ids = []
        target_splits = set(target_splits)

        for _, group in frame.sort_values(["Symbol", "Date"]).groupby("Symbol", sort=False):
            g = group.reset_index(drop=True)
            for end_idx in range(sequence_length - 1, len(g)):
                end_row = g.iloc[end_idx]
                if end_row.get("split") not in target_splits:
                    continue
                window = g.iloc[end_idx - sequence_length + 1 : end_idx + 1]
                cols = numeric_columns + sentiment_columns
                if window[cols].isna().any().any() or pd.isna(end_row.get("label")):
                    continue
                self.numeric_sequences.append(window[numeric_columns].astype(float).values)
                if sentiment_columns:
                    self.sentiment_sequences.append(window[sentiment_columns].astype(float).values)
                else:
                    self.sentiment_sequences.append(np.zeros((sequence_length, 0), dtype=float))
                self.targets.append(LABEL_TO_INDEX[int(end_row["label"])])
                self.row_ids.append(int(end_row["_row_id"]))

        self.numeric_sequences = np.array(self.numeric_sequences, dtype=np.float32)
        self.sentiment_sequences = np.array(self.sentiment_sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.int64)
        self.row_ids = np.array(self.row_ids, dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.numeric_sequences[idx], dtype=torch.float32),
            torch.tensor(self.sentiment_sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.tensor(self.row_ids[idx], dtype=torch.long),
        )


def get_torch_device() -> "torch.device":
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_random_forest(
    frame: "pd.DataFrame",
    scaled_feature_columns: List[str],
    config: PipelineConfig,
    artifacts_dir: Path,
    logger: Any,
) -> Tuple["pd.DataFrame", Dict[str, Any], "pd.DataFrame", Any, None]:
    df = frame.copy()
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    X_train = train[scaled_feature_columns].fillna(0.0).values
    y_train = train["label"].astype(int).values
    X_test = test[scaled_feature_columns].fillna(0.0).values
    y_test = test["label"].astype(int).values

    if len(set(y_train)) < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        logger.warning("Training labels contain one class; using DummyClassifier")
    else:
        base = RandomForestClassifier(
            random_state=config.random_seed,
            class_weight="balanced_subsample",
            n_jobs=config.rf_n_jobs,
        )
        cv_splits = min(config.rf_cv_splits, max(2, len(train) // 80))
        param_grid = {
            "n_estimators": [200, 350],
            "max_depth": [None, 8, 14],
            "min_samples_leaf": [1, 3],
        }
        try:
            search = GridSearchCV(
                base,
                param_grid=param_grid,
                scoring="f1_macro",
                cv=TimeSeriesSplit(n_splits=cv_splits),
                n_jobs=config.rf_n_jobs,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            logger.info(f"RandomForest best params: {search.best_params_}")
        except Exception as exc:
            logger.warning(f"RandomForest grid search failed; fitting default model: {exc}")
            model = base.fit(X_train, y_train)

    if len(test) > 0:
        pred = model.predict(X_test).astype(int)
        if hasattr(model, "predict_proba"):
            proba = align_probabilities(model.predict_proba(X_test), model.classes_)
        else:
            proba = np.zeros((len(test), 3), dtype=float)
            proba[:, 1] = 1.0
    else:
        pred = np.array([], dtype=int)
        proba = np.empty((0, 3), dtype=float)

    df = write_predictions(df, test["_row_id"].values, pred, proba)
    metrics = compute_prediction_metrics(y_test, pred, proba)

    if hasattr(model, "feature_importances_"):
        importance = np.array(model.feature_importances_, dtype=float)
    else:
        importance = np.zeros(len(scaled_feature_columns), dtype=float)
    feature_importance = pd.DataFrame(
        {
            "feature": [col.replace("scaled_model_", "") for col in scaled_feature_columns],
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)

    artifact = {
        "model": model,
        "model_type": "random_forest",
        "feature_columns": scaled_feature_columns,
        "config": asdict(config),
    }
    with open(artifacts_dir / "model_artifact.pkl", "wb") as handle:
        pickle.dump(artifact, handle)
    return df, metrics, feature_importance, model, None


def train_torch_model(
    frame: "pd.DataFrame",
    scaled_numeric_columns: List[str],
    scaled_sentiment_columns: List[str],
    config: PipelineConfig,
    artifacts_dir: Path,
    logger: Any,
) -> Tuple["pd.DataFrame", Dict[str, Any], "pd.DataFrame", Any, Optional["np.ndarray"]]:
    train_ds = WindowedDataset(frame, scaled_numeric_columns, scaled_sentiment_columns, ["train"], config.sequence_length)
    val_ds = WindowedDataset(frame, scaled_numeric_columns, scaled_sentiment_columns, ["val"], config.sequence_length)
    test_ds = WindowedDataset(frame, scaled_numeric_columns, scaled_sentiment_columns, ["test"], config.sequence_length)
    if len(train_ds) == 0:
        logger.warning("No sequence windows available; falling back to RandomForest")
        return train_random_forest(
            frame,
            scaled_numeric_columns + scaled_sentiment_columns,
            config,
            artifacts_dir,
            logger,
        )

    device = get_torch_device()
    logger.info(f"Training {config.model_type} on {device} with {len(train_ds)} train windows")
    numeric_dim = len(scaled_numeric_columns)
    sentiment_dim = len(scaled_sentiment_columns)
    if config.model_type == "lstm":
        model = LSTMClassifier(
            numeric_dim=numeric_dim,
            sentiment_dim=sentiment_dim,
            fusion_dim=config.fusion_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout,
            bidirectional=config.lstm_bidirectional,
        )
    else:
        model = TransformerClassifier(
            numeric_dim=numeric_dim,
            sentiment_dim=sentiment_dim,
            fusion_dim=config.fusion_dim,
            d_model=config.transformer_d_model,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
            dropout=config.dropout,
            max_len=config.sequence_length,
        )
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False) if len(val_ds) else None

    class_weights = compute_class_weights([INDEX_TO_LABEL[int(y)] for y in train_ds.targets])
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    best_state = None
    best_val = float("inf")
    patience = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for numeric_x, sentiment_x, target, _ in train_loader:
            numeric_x = numeric_x.to(device)
            sentiment_x = sentiment_x.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(numeric_x, sentiment_x)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_torch_loss(model, val_loader, criterion, device) if val_loader is not None else float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(val_loss)})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"{config.model_type} epoch {epoch}: train_loss={np.mean(train_losses):.4f}, val_loss={val_loss:.4f}")
        if patience >= config.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    row_ids, pred, proba, attention = predict_torch(model, test_ds, config.batch_size, device)
    y_test = [INDEX_TO_LABEL[int(y)] for y in test_ds.targets]
    df = write_predictions(frame.copy(), row_ids, pred, proba)
    metrics = compute_prediction_metrics(y_test, pred, proba)
    metrics["training_history"] = history
    metrics["device"] = str(device)

    feature_importance = heuristic_feature_importance(df, scaled_numeric_columns + scaled_sentiment_columns)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": config.model_type,
            "numeric_columns": scaled_numeric_columns,
            "sentiment_columns": scaled_sentiment_columns,
            "config": asdict(config),
            "history": history,
        },
        artifacts_dir / "model_artifact.pt",
    )
    return df, metrics, feature_importance, model, attention


def evaluate_torch_loss(model, loader, criterion, device) -> float:
    if loader is None:
        return float("inf")
    model.eval()
    losses = []
    with torch.no_grad():
        for numeric_x, sentiment_x, target, _ in loader:
            numeric_x = numeric_x.to(device)
            sentiment_x = sentiment_x.to(device)
            target = target.to(device)
            losses.append(float(criterion(model(numeric_x, sentiment_x), target).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def predict_torch(model, dataset, batch_size: int, device) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", Optional["np.ndarray"]]:
    if len(dataset) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.empty((0, 3), dtype=float), None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_row_ids = []
    all_pred = []
    all_proba = []
    attention = None
    with torch.no_grad():
        for numeric_x, sentiment_x, _, row_ids in loader:
            numeric_x = numeric_x.to(device)
            sentiment_x = sentiment_x.to(device)
            logits = model(numeric_x, sentiment_x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred_indices = np.argmax(probs, axis=1)
            all_row_ids.extend(row_ids.detach().cpu().numpy().tolist())
            all_pred.extend([INDEX_TO_LABEL[int(idx)] for idx in pred_indices])
            all_proba.append(probs)
            if hasattr(model, "get_attention_matrix"):
                matrix = model.get_attention_matrix()
                if matrix is not None:
                    attention = matrix
    return np.array(all_row_ids, dtype=int), np.array(all_pred, dtype=int), np.vstack(all_proba), attention


def write_predictions(frame: "pd.DataFrame", row_ids: Sequence[int], pred: Sequence[int], proba: "np.ndarray") -> "pd.DataFrame":
    df = frame.copy()
    for col, default in [
        ("prediction", 0),
        ("prob_sell", 0.0),
        ("prob_hold", 1.0),
        ("prob_buy", 0.0),
        ("signal_name", "HOLD"),
    ]:
        if col not in df.columns:
            df[col] = default
    for i, row_id in enumerate(row_ids):
        mask = df["_row_id"] == int(row_id)
        df.loc[mask, "prediction"] = int(pred[i])
        df.loc[mask, "prob_sell"] = float(proba[i, 0])
        df.loc[mask, "prob_hold"] = float(proba[i, 1])
        df.loc[mask, "prob_buy"] = float(proba[i, 2])
    df["signal_name"] = df["prediction"].map({-1: "SELL", 0: "HOLD", 1: "BUY"}).fillna("HOLD")
    return df


def heuristic_feature_importance(frame: "pd.DataFrame", scaled_columns: List[str]) -> "pd.DataFrame":
    if frame.empty or "label" not in frame.columns:
        return pd.DataFrame({"feature": scaled_columns, "importance": np.zeros(len(scaled_columns))})
    train = frame[frame["split"] == "train"]
    scores = []
    y = train["label"].astype(float)
    for col in scaled_columns:
        x = train[col].astype(float)
        corr = abs(x.corr(y)) if x.std() > 0 and y.std() > 0 else 0.0
        scores.append(0.0 if pd.isna(corr) else float(corr))
    return pd.DataFrame(
        {"feature": [col.replace("scaled_model_", "") for col in scaled_columns], "importance": scores}
    ).sort_values("importance", ascending=False)


class Backtester:
    def __init__(self, config: PipelineConfig, logger: Any):
        self.config = config
        self.logger = logger

    def run(self, frame: "pd.DataFrame") -> Tuple["pd.DataFrame", "pd.DataFrame", Dict[str, Any]]:
        test = frame[frame["split"] == "test"].copy()
        if test.empty:
            equity = pd.DataFrame(
                {
                    "Date": [],
                    "equity": [],
                    "cash": [],
                    "invested": [],
                    "cash_ratio": [],
                    "invested_ratio": [],
                    "drawdown": [],
                    "buy_hold_equity": [],
                }
            )
            trades = pd.DataFrame(columns=self.trade_columns())
            return equity, trades, self._metrics(equity, trades)

        test = test.sort_values(["Date", "Symbol"])
        all_dates = sorted(test["Date"].dropna().unique())
        cash = float(self.config.initial_capital)
        positions: Dict[str, Dict[str, Any]] = {}
        equity_rows = []
        trades = []
        last_prices: Dict[str, float] = {}

        for current_date in all_dates:
            day = test[test["Date"] == current_date]
            for _, row in day.iterrows():
                last_prices[row["Symbol"]] = safe_float(row["Adjusted Close"])

            current_equity = cash + sum(pos["shares"] * last_prices.get(sym, pos["entry_price"]) for sym, pos in positions.items())

            for symbol in list(positions.keys()):
                row_match = day[day["Symbol"] == symbol]
                if row_match.empty:
                    continue
                row = row_match.iloc[0]
                price = safe_float(row["Adjusted Close"])
                pos = positions[symbol]
                holding_period = int(pos.get("holding_period", 0)) + 1
                pos["holding_period"] = holding_period
                trade_return = price / pos["entry_price"] - 1.0 if pos["entry_price"] else 0.0
                exit_reason = None
                if int(row.get("prediction", 0)) == -1 and safe_float(row.get("prob_sell"), 0.0) >= self.config.probability_threshold:
                    exit_reason = "sell_signal"
                elif trade_return <= -self.config.stop_loss:
                    exit_reason = "stop_loss"
                elif trade_return >= self.config.take_profit:
                    exit_reason = "take_profit"
                elif holding_period >= 10:
                    exit_reason = "max_holding_period"
                if exit_reason:
                    proceeds = pos["shares"] * price
                    cash += proceeds
                    pnl = proceeds - pos["entry_value"]
                    trades.append(
                        {
                            "symbol": symbol,
                            "entry_date": pos["entry_date"],
                            "exit_date": current_date,
                            "entry_price": pos["entry_price"],
                            "exit_price": price,
                            "shares": pos["shares"],
                            "entry_value": pos["entry_value"],
                            "exit_value": proceeds,
                            "pnl": pnl,
                            "return_pct": trade_return,
                            "holding_period": holding_period,
                            "exit_reason": exit_reason,
                        }
                    )
                    del positions[symbol]

            current_equity = cash + sum(pos["shares"] * last_prices.get(sym, pos["entry_price"]) for sym, pos in positions.items())
            entries = day[
                (day["prediction"].astype(int) == 1)
                & (day["prob_buy"].astype(float) >= self.config.probability_threshold)
            ].sort_values("prob_buy", ascending=False)
            for _, row in entries.iterrows():
                symbol = row["Symbol"]
                if symbol in positions:
                    continue
                price = safe_float(row["Adjusted Close"])
                if price <= 0:
                    continue
                allocation = min(cash, current_equity * self.config.position_size_pct)
                if allocation <= 0:
                    continue
                shares = allocation / price
                cash -= allocation
                positions[symbol] = {
                    "entry_date": current_date,
                    "entry_price": price,
                    "shares": shares,
                    "entry_value": allocation,
                    "holding_period": 0,
                }

            invested = sum(pos["shares"] * last_prices.get(sym, pos["entry_price"]) for sym, pos in positions.items())
            equity_value = cash + invested
            equity_rows.append(
                {
                    "Date": current_date,
                    "equity": equity_value,
                    "cash": cash,
                    "invested": invested,
                    "cash_ratio": cash / equity_value if equity_value else 0.0,
                    "invested_ratio": invested / equity_value if equity_value else 0.0,
                }
            )

        if positions and all_dates:
            final_date = all_dates[-1]
            for symbol, pos in list(positions.items()):
                price = last_prices.get(symbol, pos["entry_price"])
                proceeds = pos["shares"] * price
                cash += proceeds
                pnl = proceeds - pos["entry_value"]
                trades.append(
                    {
                        "symbol": symbol,
                        "entry_date": pos["entry_date"],
                        "exit_date": final_date,
                        "entry_price": pos["entry_price"],
                        "exit_price": price,
                        "shares": pos["shares"],
                        "entry_value": pos["entry_value"],
                        "exit_value": proceeds,
                        "pnl": pnl,
                        "return_pct": price / pos["entry_price"] - 1.0 if pos["entry_price"] else 0.0,
                        "holding_period": pos.get("holding_period", 0),
                        "exit_reason": "end_of_test",
                    }
                )

        equity = pd.DataFrame(equity_rows)
        if not equity.empty:
            equity["running_max"] = equity["equity"].cummax()
            equity["drawdown"] = equity["equity"] / equity["running_max"] - 1.0
            equity["buy_hold_equity"] = self._buy_and_hold_curve(test, equity["Date"], self.config.initial_capital)
        trades_df = pd.DataFrame(trades, columns=self.trade_columns())
        metrics = self._metrics(equity, trades_df)
        return equity, trades_df, metrics

    def _buy_and_hold_curve(self, test: "pd.DataFrame", dates: "pd.Series", initial_capital: float) -> "pd.Series":
        curves = []
        for symbol, group in test.groupby("Symbol"):
            prices = group.sort_values("Date").set_index("Date")["Adjusted Close"].astype(float)
            prices = prices.reindex(pd.to_datetime(dates)).ffill().bfill()
            if prices.empty or prices.iloc[0] == 0:
                continue
            curves.append(prices / prices.iloc[0])
        if not curves:
            return pd.Series(initial_capital, index=range(len(dates)))
        benchmark = pd.concat(curves, axis=1).mean(axis=1) * initial_capital
        return benchmark.reset_index(drop=True)

    def _metrics(self, equity: "pd.DataFrame", trades: "pd.DataFrame") -> Dict[str, Any]:
        if equity.empty:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "max_drawdown_duration_days": 0,
                "calmar_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_trade_return": 0.0,
                "annualized_volatility": 0.0,
                "number_of_trades": 0,
            }

        daily_returns = equity["equity"].pct_change().dropna()
        days = max((pd.to_datetime(equity["Date"].iloc[-1]) - pd.to_datetime(equity["Date"].iloc[0])).days, 1)
        total_return = equity["equity"].iloc[-1] / self.config.initial_capital - 1.0
        annualized_return = (1.0 + total_return) ** (365.0 / days) - 1.0 if total_return > -1.0 else -1.0
        risk_free_daily = 0.045 / 252.0
        excess = daily_returns - risk_free_daily
        sharpe = math.sqrt(252) * excess.mean() / excess.std() if len(excess) > 1 and excess.std() > 0 else 0.0
        downside = excess[excess < 0]
        sortino = math.sqrt(252) * excess.mean() / downside.std() if len(downside) > 1 and downside.std() > 0 else 0.0
        max_drawdown = float(equity["drawdown"].min()) if "drawdown" in equity else 0.0
        max_duration = self._max_drawdown_duration(equity.get("drawdown", pd.Series(dtype=float)))
        calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
        volatility = daily_returns.std() * math.sqrt(252) if len(daily_returns) > 1 else 0.0

        if trades.empty:
            win_rate = profit_factor = average_trade_return = 0.0
        else:
            wins = trades[trades["pnl"] > 0]
            losses = trades[trades["pnl"] < 0]
            win_rate = len(wins) / len(trades) if len(trades) else 0.0
            gross_profit = wins["pnl"].sum()
            gross_loss = abs(losses["pnl"].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
            average_trade_return = trades["return_pct"].mean()

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": float(max_drawdown),
            "max_drawdown_duration_days": int(max_duration),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "profit_factor": json_safe(profit_factor),
            "average_trade_return": float(average_trade_return),
            "annualized_volatility": float(volatility),
            "number_of_trades": int(len(trades)),
        }

    def _max_drawdown_duration(self, drawdown: "pd.Series") -> int:
        max_duration = 0
        current = 0
        for value in drawdown.fillna(0.0):
            if value < 0:
                current += 1
                max_duration = max(max_duration, current)
            else:
                current = 0
        return max_duration

    def trade_columns(self) -> List[str]:
        return [
            "symbol",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "shares",
            "entry_value",
            "exit_value",
            "pnl",
            "return_pct",
            "holding_period",
            "exit_reason",
        ]


class AsyncMarketDataFeed:
    """Placeholder structure for later websocket ingestion."""

    async def connect(self) -> None:
        await asyncio.sleep(0)

    async def subscribe(self, symbols: List[str]) -> None:
        await asyncio.sleep(0)


class Broker:
    """Abstract live/paper trading connector placeholder."""

    def submit_order(self, symbol: str, side: str, quantity: float, order_type: str = "market") -> Dict[str, Any]:
        raise NotImplementedError("Implement in a paper/live broker adapter.")


class PortfolioOptimizer:
    """Markowitz-ready placeholder for multi-asset allocation extensions."""

    def markowitz_weights(self, returns: "pd.DataFrame") -> "pd.Series":
        if returns.empty:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / returns.shape[1], index=returns.columns)


class DriftDetector:
    """Prediction distribution drift check for later monitoring hooks."""

    def compare_prediction_distribution(self, train_predictions: Sequence[int], recent_predictions: Sequence[int]) -> Dict[str, float]:
        train_counts = pd.Series(train_predictions).value_counts(normalize=True)
        recent_counts = pd.Series(recent_predictions).value_counts(normalize=True)
        divergence = 0.0
        for label in LABEL_ORDER:
            divergence += abs(float(train_counts.get(label, 0.0)) - float(recent_counts.get(label, 0.0)))
        return {"l1_prediction_distribution_shift": divergence}


class ExperimentLogger:
    """MLflow/W&B-ready no-op hook."""

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        # Hook point: mlflow.log_metrics(...) or wandb.log(...)
        return None


def empty_figure(title: str):
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white", height=420)
    return fig


def filter_dashboard_frame(frame: "pd.DataFrame", symbol: str, start_date: Optional[str], end_date: Optional[str]) -> "pd.DataFrame":
    df = frame[frame["Symbol"] == symbol].copy()
    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date)]
    return df.sort_values("Date")


def build_market_figure(frame: "pd.DataFrame", symbol: str, start_date: Optional[str], end_date: Optional[str]):
    if not symbol:
        return empty_figure("Market Analysis")
    df = filter_dashboard_frame(frame, symbol, start_date, end_date)
    if df.empty:
        return empty_figure(f"{symbol} Market Analysis")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    for col, name in [
        ("ema_9", "EMA 9"),
        ("ema_21", "EMA 21"),
        ("sma_50", "SMA 50"),
        ("bb_high", "BB High"),
        ("bb_low", "BB Low"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=name), row=1, col=1)
    buys = df[df["prediction"] == 1]
    sells = df[df["prediction"] == -1]
    fig.add_trace(
        go.Scatter(
            x=buys["Date"],
            y=buys["Low"] * 0.98,
            mode="markers",
            name="BUY",
            marker=dict(symbol="triangle-up", color="#1a9850", size=11),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sells["Date"],
            y=sells["High"] * 1.02,
            mode="markers",
            name="SELL",
            marker=dict(symbol="triangle-down", color="#d73027", size=11),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color="#6c757d"), row=2, col=1)
    fig.update_layout(title=f"{symbol} Market Analysis", template="plotly_white", height=720, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def build_model_figures(
    frame: "pd.DataFrame",
    equity: "pd.DataFrame",
    feature_importance: "pd.DataFrame",
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    equity_fig = empty_figure("Equity Curve")
    drawdown_fig = empty_figure("Underwater Plot")
    if not equity.empty:
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(x=equity["Date"], y=equity["equity"], mode="lines", name="Strategy"))
        if "buy_hold_equity" in equity:
            equity_fig.add_trace(go.Scatter(x=equity["Date"], y=equity["buy_hold_equity"], mode="lines", name="Buy & Hold"))
        equity_fig.update_layout(title="Equity Curve vs Buy-and-Hold", template="plotly_white", height=420)
        drawdown_fig = go.Figure(go.Scatter(x=equity["Date"], y=equity["drawdown"], fill="tozeroy", mode="lines", name="Drawdown"))
        drawdown_fig.update_layout(title="Underwater Plot", template="plotly_white", height=420)

    cm = np.array(metrics.get("model_performance", {}).get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    confusion_fig = px.imshow(cm, x=CLASS_NAMES, y=CLASS_NAMES, text_auto=True, color_continuous_scale="Blues")
    confusion_fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", height=420)

    roc_fig = go.Figure()
    perf_df = frame[frame["split"] == "test"].copy()
    for label, name, prob_col in [(-1, "SELL", "prob_sell"), (0, "HOLD", "prob_hold"), (1, "BUY", "prob_buy")]:
        y_true = (perf_df["label"].astype(int) == label).astype(int) if not perf_df.empty else pd.Series(dtype=int)
        if len(y_true) > 1 and y_true.nunique() == 2:
            fpr, tpr, _ = roc_curve(y_true, perf_df[prob_col].astype(float))
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} AUC={auc(fpr, tpr):.2f}"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    roc_fig.update_layout(
        title="One-vs-Rest ROC",
        template="plotly_white",
        height=420,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )

    top = feature_importance.head(15).sort_values("importance", ascending=True)
    importance_fig = px.bar(top, x="importance", y="feature", orientation="h", title="Top 15 Feature Importance")
    importance_fig.update_layout(template="plotly_white", height=420)
    return {
        "equity-chart": equity_fig,
        "drawdown-chart": drawdown_fig,
        "confusion-chart": confusion_fig,
        "roc-chart": roc_fig,
        "feature-importance-chart": importance_fig,
    }


def build_clustering_figures(metrics: Dict[str, Any]) -> Dict[str, Any]:
    clustering_results = metrics.get("clustering_results", {})
    anomaly_results = metrics.get("anomaly_detection", {})

    # Clustering visualization
    if "error" not in clustering_results and clustering_results.get("clustered_data"):
        clustered_data = pd.DataFrame(clustering_results["clustered_data"])
        if not clustered_data.empty and "cluster" in clustered_data.columns:
            # Simple scatter plot of first two features colored by cluster
            numerical_cols = clustered_data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) >= 2:
                cluster_fig = px.scatter(
                    clustered_data,
                    x=numerical_cols[0],
                    y=numerical_cols[1],
                    color="cluster",
                    title=f"Clustering Results (n_clusters={clustering_results.get('n_clusters', 'N/A')})",
                    labels={numerical_cols[0]: numerical_cols[0], numerical_cols[1]: numerical_cols[1]}
                )
                cluster_fig.update_layout(template="plotly_white", height=420)
            else:
                cluster_fig = empty_figure("Clustering Results - Insufficient numerical features")
        else:
            cluster_fig = empty_figure("Clustering Results - No cluster data")
    else:
        cluster_fig = empty_figure("Clustering Results - Error or unavailable")

    # Anomaly detection visualization
    if "error" not in anomaly_results and anomaly_results.get("anomalous_data"):
        anomalous_data = pd.DataFrame(anomaly_results["anomalous_data"])
        if not anomalous_data.empty:
            anomaly_fig = go.Figure()
            anomaly_fig.add_trace(go.Scatter(
                x=list(range(len(anomalous_data))),
                y=anomalous_data.get("anomaly_score", []),
                mode="markers",
                name="Anomalies",
                marker=dict(color="red", size=8)
            ))
            anomaly_fig.update_layout(
                title=f"Anomaly Detection (n_anomalies={anomaly_results.get('n_anomalies', 0)})",
                template="plotly_white",
                height=420,
                xaxis_title="Sample Index",
                yaxis_title="Anomaly Score"
            )
        else:
            anomaly_fig = empty_figure("Anomaly Detection - No anomalies detected")
    else:
        anomaly_fig = empty_figure("Anomaly Detection - Error or unavailable")

    return {
        "clustering-chart": cluster_fig,
        "anomaly-chart": anomaly_fig,
    }


def build_trade_figures(equity: "pd.DataFrame", trades: "pd.DataFrame") -> Dict[str, Any]:
    if equity.empty:
        monthly_fig = empty_figure("Monthly Returns")
        cumulative_fig = empty_figure("Cumulative Returns")
    else:
        eq = equity.copy().set_index("Date")
        monthly = eq["equity"].resample("ME").last().pct_change().dropna()
        if monthly.empty:
            monthly_fig = empty_figure("Monthly Returns")
        else:
            monthly_df = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.strftime("%b"), "return": monthly.values})
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            pivot = monthly_df.pivot(index="year", columns="month", values="return").reindex(columns=month_order)
            monthly_fig = px.imshow(pivot, text_auto=".1%", color_continuous_scale="RdYlGn", title="Monthly Returns")
        cumulative_fig = go.Figure()
        cumulative_fig.add_trace(go.Scatter(x=equity["Date"], y=equity["equity"] / equity["equity"].iloc[0] - 1, mode="lines", name="Strategy"))
        if "buy_hold_equity" in equity:
            cumulative_fig.add_trace(
                go.Scatter(
                    x=equity["Date"],
                    y=equity["buy_hold_equity"] / equity["buy_hold_equity"].iloc[0] - 1,
                    mode="lines",
                    name="Buy & Hold",
                )
            )
        cumulative_fig.update_layout(title="Cumulative Returns", template="plotly_white", height=420)

    if trades.empty:
        winloss_fig = empty_figure("Win/Loss Distribution")
    else:
        winloss_fig = px.histogram(trades, x="return_pct", nbins=30, title="Win/Loss Distribution")
        winloss_fig.update_layout(template="plotly_white", height=420)
    return {
        "monthly-returns-chart": monthly_fig,
        "winloss-chart": winloss_fig,
        "cumulative-chart": cumulative_fig,
    }


def build_multimodal_figures(
    frame: "pd.DataFrame",
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
    attention_matrix: Optional["np.ndarray"],
) -> Dict[str, Any]:
    df = filter_dashboard_frame(frame, symbol, start_date, end_date) if symbol else pd.DataFrame()
    if df.empty:
        sentiment_fig = empty_figure("Sentiment and Price")
        corr_fig = empty_figure("Feature Correlation")
    else:
        sentiment_fig = make_subplots(specs=[[{"secondary_y": True}]])
        sentiment_fig.add_trace(go.Scatter(x=df["Date"], y=df["Adjusted Close"], name="Adjusted Close"), secondary_y=False)
        sentiment_fig.add_trace(go.Scatter(x=df["Date"], y=df["sentiment_score"], name="Sentiment"), secondary_y=True)
        sentiment_fig.update_layout(title="Sentiment Score Overlay", template="plotly_white", height=420)
        numeric_cols = [
            col
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
            and col not in {"_row_id", "label", "prediction", "prob_sell", "prob_hold", "prob_buy"}
        ][:24]
        corr = df[numeric_cols].corr().fillna(0.0) if numeric_cols else pd.DataFrame()
        corr_fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Feature Correlation Matrix")
        corr_fig.update_layout(height=650)

    if attention_matrix is None:
        attention_fig = empty_figure("Transformer Attention")
        attention_fig.add_annotation(
            text="Attention is available when the Transformer model produces test windows.",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    else:
        labels = [
            f"t-{attention_matrix.shape[0] - i - 1}" if i < attention_matrix.shape[0] - 1 else "t"
            for i in range(attention_matrix.shape[0])
        ]
        attention_fig = px.imshow(attention_matrix, x=labels, y=labels, color_continuous_scale="Viridis", title="Transformer Attention Heatmap")
        attention_fig.update_layout(height=420)
    return {
        "sentiment-chart": sentiment_fig,
        "attention-chart": attention_fig,
        "correlation-chart": corr_fig,
    }


def trade_table_html(trades: "pd.DataFrame") -> str:
    display_trades = trades.copy()
    if display_trades.empty:
        return '<p class="muted">No closed trades were generated for the current prediction threshold.</p>'
    for col in ["entry_date", "exit_date"]:
        display_trades[col] = pd.to_datetime(display_trades[col]).dt.strftime("%Y-%m-%d")
    for col in ["entry_price", "exit_price", "shares", "entry_value", "exit_value", "pnl", "return_pct"]:
        display_trades[col] = pd.to_numeric(display_trades[col], errors="coerce").round(4)
    return display_trades.to_html(classes="trade-table sortable", index=False, border=0, escape=True)


def figures_to_payload(figures: Dict[str, Any]) -> str:
    payload = {chart_id: json.loads(fig.to_json()) for chart_id, fig in figures.items()}
    return json.dumps(json_safe(payload))


FLASK_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multimodal Financial Analysis</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #18212f;
      --muted: #667085;
      --line: #d9e0ea;
      --accent: #1565c0;
      --accent-dark: #0f4f96;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }
    .shell { width: min(1520px, 96vw); margin: 0 auto; padding: 22px 0 34px; }
    header { display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 16px; }
    h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: 0; }
    .muted { color: var(--muted); }
    .filters {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      display: grid;
      grid-template-columns: 180px 180px 180px auto;
      gap: 10px;
      align-items: end;
      margin-bottom: 16px;
    }
    label { font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }
    select, input, button {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
      font-size: 14px;
    }
    button { background: var(--accent); color: #fff; border-color: var(--accent); cursor: pointer; }
    button:hover { background: var(--accent-dark); }
    .metrics { display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap: 10px; margin-bottom: 16px; }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .metric span { color: var(--muted); font-size: 12px; display: block; }
    .metric strong { font-size: 20px; margin-top: 4px; display: block; }
    .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .tab-button {
      width: auto;
      background: #fff;
      color: var(--ink);
      border: 1px solid var(--line);
      min-width: 148px;
    }
    .tab-button.active { background: var(--accent); border-color: var(--accent); color: #fff; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .grid-2, .grid-3 {
      display: grid;
      gap: 12px;
      margin-bottom: 12px;
    }
    .grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .chart, .table-panel, .json-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      min-width: 0;
    }
    .chart { height: auto; }
    .trade-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      white-space: nowrap;
    }
    .trade-table th, .trade-table td {
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: right;
    }
    .trade-table th { cursor: pointer; color: var(--muted); background: #f9fafb; }
    .trade-table th:first-child, .trade-table td:first-child { text-align: left; }
    .table-scroll { overflow-x: auto; }
    pre { overflow-x: auto; margin: 0; font-size: 12px; color: #344054; }
    @media (max-width: 900px) {
      header { display: block; }
      .filters, .metrics, .grid-2, .grid-3 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>Transformer-Based Multimodal Financial Analysis</h1>
        <div class="muted">Flask deployment with interactive Plotly charts</div>
      </div>
      <div class="muted">Rows: {{ row_count }} | Model: {{ model_type }} | Sentiment: {{ sentiment_mode }}</div>
    </header>

    <form class="filters" method="get" action="/">
      <div>
        <label for="symbol">Symbol</label>
        <select id="symbol" name="symbol">
          {% for item in symbol_options %}
            <option value="{{ item.symbol }}" {% if item.selected %}selected{% endif %}>{{ item.symbol }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label for="start_date">Start date</label>
        <input id="start_date" name="start_date" type="date" min="{{ min_date }}" max="{{ max_date }}" value="{{ selected_start }}">
      </div>
      <div>
        <label for="end_date">End date</label>
        <input id="end_date" name="end_date" type="date" min="{{ min_date }}" max="{{ max_date }}" value="{{ selected_end }}">
      </div>
      <button type="submit">Update Dashboard</button>
    </form>

    <section class="metrics">
      {% for card in metric_cards %}
        <div class="metric"><span>{{ card.label }}</span><strong>{{ card.value }}</strong></div>
      {% endfor %}
    </section>

    <nav class="tabs">
      <button class="tab-button active" type="button" data-tab="market">Market Analysis</button>
      <button class="tab-button" type="button" data-tab="model">Model Performance</button>
      <button class="tab-button" type="button" data-tab="trades">Trade Analytics</button>
      <button class="tab-button" type="button" data-tab="multimodal">Multimodal Insights</button>
      <button class="tab-button" type="button" data-tab="clustering">Unsupervised Analysis</button>
    </nav>

    <section id="market" class="tab-panel active">
      <div class="chart" id="market-chart"></div>
    </section>

    <section id="model" class="tab-panel">
      <div class="grid-2">
        <div class="chart" id="equity-chart"></div>
        <div class="chart" id="drawdown-chart"></div>
      </div>
      <div class="grid-3">
        <div class="chart" id="confusion-chart"></div>
        <div class="chart" id="roc-chart"></div>
        <div class="chart" id="feature-importance-chart"></div>
      </div>
    </section>

    <section id="trades" class="tab-panel">
      <div class="table-panel table-scroll">{{ trade_table|safe }}</div>
      <div class="grid-3">
        <div class="chart" id="monthly-returns-chart"></div>
        <div class="chart" id="winloss-chart"></div>
        <div class="chart" id="cumulative-chart"></div>
      </div>
    </section>

    <section id="multimodal" class="tab-panel">
      <div class="grid-2">
        <div class="chart" id="sentiment-chart"></div>
        <div class="chart" id="attention-chart"></div>
      </div>
      <div class="chart" id="correlation-chart"></div>
      <div class="json-panel"><pre>{{ metrics_json }}</pre></div>
    </section>

    <section id="clustering" class="tab-panel">
      <div class="grid-2">
        <div class="chart" id="clustering-chart"></div>
        <div class="chart" id="anomaly-chart"></div>
      </div>
    </section>
  </main>

  <script>
    const figures = {{ plot_payload|safe }};
    Object.entries(figures).forEach(([id, fig]) => {
      Plotly.newPlot(id, fig.data, fig.layout, {responsive: true, displaylogo: false});
    });

    document.querySelectorAll(".tab-button").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab-button").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.tab).classList.add("active");
        setTimeout(() => window.dispatchEvent(new Event("resize")), 30);
      });
    });

    document.querySelectorAll("table.sortable th").forEach((header, colIndex) => {
      header.addEventListener("click", () => {
        const table = header.closest("table");
        const body = table.tBodies[0];
        const rows = Array.from(body.querySelectorAll("tr"));
        const asc = header.dataset.asc !== "true";
        rows.sort((a, b) => {
          const av = a.children[colIndex].innerText.trim();
          const bv = b.children[colIndex].innerText.trim();
          const an = Number(av);
          const bn = Number(bv);
          if (!Number.isNaN(an) && !Number.isNaN(bn)) return asc ? an - bn : bn - an;
          return asc ? av.localeCompare(bv) : bv.localeCompare(av);
        });
        header.dataset.asc = asc;
        rows.forEach((row) => body.appendChild(row));
      });
    });
  </script>
</body>
</html>
"""


def percent_text(value: Any) -> str:
    return f"{safe_float(value) * 100:.2f}%"


def number_text(value: Any, digits: int = 2) -> str:
    return f"{safe_float(value):.{digits}f}"


def build_flask_app(
    frame: "pd.DataFrame",
    equity: "pd.DataFrame",
    trades: "pd.DataFrame",
    feature_importance: "pd.DataFrame",
    metrics: Dict[str, Any],
    attention_matrix: Optional["np.ndarray"],
):
    global app, server
    app = Flask(__name__)
    symbols = sorted(frame["Symbol"].dropna().unique())
    min_date = date_string(frame["Date"].min()) if not frame.empty else date_string(pd.Timestamp.today() - pd.Timedelta(days=365))
    max_date = date_string(frame["Date"].max()) if not frame.empty else date_string(pd.Timestamp.today())

    @app.route("/")
    def index():
        selected_symbol = request.args.get("symbol") or (symbols[0] if symbols else "")
        if selected_symbol not in symbols and symbols:
            selected_symbol = symbols[0]
        selected_start = request.args.get("start_date") or min_date
        selected_end = request.args.get("end_date") or max_date

        figures = {"market-chart": build_market_figure(frame, selected_symbol, selected_start, selected_end)}
        figures.update(build_model_figures(frame, equity, feature_importance, metrics))
        figures.update(build_trade_figures(equity, trades))
        figures.update(build_multimodal_figures(frame, selected_symbol, selected_start, selected_end, attention_matrix))
        figures.update(build_clustering_figures(metrics))

        risk = metrics.get("risk_metrics", {})
        model = metrics.get("model_performance", {})
        metric_cards = [
            {"label": "Total Return", "value": percent_text(risk.get("total_return", 0.0))},
            {"label": "Sharpe", "value": number_text(risk.get("sharpe_ratio", 0.0))},
            {"label": "Max Drawdown", "value": percent_text(risk.get("max_drawdown_pct", 0.0))},
            {"label": "Accuracy", "value": percent_text(model.get("accuracy", 0.0))},
            {"label": "Trades", "value": str(int(safe_float(risk.get("number_of_trades", 0))))},
        ]
        symbol_options = [{"symbol": symbol, "selected": symbol == selected_symbol} for symbol in symbols]
        return render_template_string(
            FLASK_TEMPLATE,
            plot_payload=figures_to_payload(figures),
            trade_table=trade_table_html(trades),
            symbol_options=symbol_options,
            selected_start=selected_start,
            selected_end=selected_end,
            min_date=min_date,
            max_date=max_date,
            metric_cards=metric_cards,
            row_count=len(frame),
            model_type=metrics.get("model_type", ""),
            sentiment_mode=metrics.get("sentiment_mode", ""),
            metrics_json=json.dumps(json_safe(metrics), indent=2),
        )

    @app.route("/metrics.json")
    def metrics_json_route():
        return jsonify(json_safe(metrics))

    server = app
    return app


if Flask is not None:
    app = Flask(__name__)

    @app.route("/")
    def _uninitialized_flask_app():
        return "Run run_pipeline(config) to initialize the multimodal financial analysis Flask dashboard."

    server = app
else:
    app = None
    server = None


def save_artifacts(
    artifacts_dir: Path,
    metrics: Dict[str, Any],
    trades: "pd.DataFrame",
    feature_importance: "pd.DataFrame",
    scaler: Any,
    feature_columns: List[str],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "metrics_report.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(metrics), handle, indent=2)
    trades.to_csv(artifacts_dir / "backtest_trades.csv", index=False)
    feature_importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)
    with open(artifacts_dir / "scaler_artifact.pkl", "wb") as handle:
        pickle.dump({"scaler": scaler, "feature_columns": feature_columns}, handle)


def run_pipeline(config: dict) -> tuple:
    ensure_dependencies()
    pipeline_config = PipelineConfig.from_dict(config)
    logger = setup_logging()
    set_global_seed(pipeline_config.random_seed)
    artifacts_dir = Path(pipeline_config.artifacts_dir)
    logger.info("Starting multimodal financial analysis pipeline")
    logger.info(f"Configuration: {json.dumps(json_safe(asdict(pipeline_config)), indent=2)}")

    aggregator = DataAggregator(pipeline_config, logger)
    raw_data = aggregator.fetch(pipeline_config.symbols, pipeline_config.start_date, pipeline_config.end_date)
    if raw_data.empty:
        raise RuntimeError("No market data available.")

    features, numeric_features = FeatureEngineer(logger).transform(raw_data)
    sentiment_generator = SentimentFeatureGenerator(pipeline_config, logger)
    features, sentiment_features = sentiment_generator.transform(features)

    # Add unsupervised clustering analysis
    clustering_analyzer = ClusteringAnalyzer(pipeline_config, logger)
    clustering_results = clustering_analyzer.perform_clustering(features, method="kmeans", n_clusters=3)
    anomaly_results = clustering_analyzer.detect_anomalies(features, contamination=0.05)

    labeled = LabelGenerator(pipeline_config).transform(features)

    modeled, model_features, model_numeric, model_sentiment = create_model_columns(labeled, numeric_features, sentiment_features)
    modeled = assign_temporal_splits(modeled, pipeline_config, model_features, logger)
    scaler, modeled, scaled_feature_columns = fit_scaler(modeled, model_features, logger)
    scaled_numeric = [f"scaled_{col}" for col in model_numeric]
    scaled_sentiment = [f"scaled_{col}" for col in model_sentiment]

    modeled["prediction"] = 0
    modeled["prob_sell"] = 0.0
    modeled["prob_hold"] = 1.0
    modeled["prob_buy"] = 0.0
    modeled["signal_name"] = "HOLD"

    if pipeline_config.model_type == "random_forest":
        predicted, model_metrics, feature_importance, model, attention = train_random_forest(
            modeled,
            scaled_feature_columns,
            pipeline_config,
            artifacts_dir,
            logger,
        )
    else:
        predicted, model_metrics, feature_importance, model, attention = train_torch_model(
            modeled,
            scaled_numeric,
            scaled_sentiment,
            pipeline_config,
            artifacts_dir,
            logger,
        )

    backtester = Backtester(pipeline_config, logger)
    equity_curve, trades, risk_metrics = backtester.run(predicted)

    split_counts = predicted[predicted["split"].isin(["train", "val", "test"])].groupby(["split", "label"]).size().unstack(fill_value=0)
    metrics = {
        "project_title": "Transformer-Based Multimodal Financial Analysis for Trading Entry and Exit Prediction",
        "symbols": pipeline_config.symbols,
        "date_range": {"start": pipeline_config.start_date, "end": pipeline_config.end_date},
        "model_type": pipeline_config.model_type,
        "sentiment_mode": pipeline_config.sentiment_mode,
        "rows": int(len(predicted)),
        "split_label_counts": split_counts.to_dict(),
        "model_performance": model_metrics,
        "risk_metrics": risk_metrics,
        "clustering_results": clustering_results,
        "anomaly_detection": anomaly_results,
        "artifacts": {
            "model": "model_artifact.pt" if pipeline_config.model_type != "random_forest" else "model_artifact.pkl",
            "metrics": "metrics_report.json",
            "trades": "backtest_trades.csv",
            "feature_importance": "feature_importance.csv",
            "cache": pipeline_config.cache_path,
        },
    }
    if attention is not None:
        metrics["attention_shape"] = list(attention.shape)

    save_artifacts(artifacts_dir, metrics, trades, feature_importance, scaler, scaled_feature_columns)
    dashboard = build_flask_app(predicted, equity_curve, trades, feature_importance, metrics, attention)
    logger.info("Pipeline completed")
    return predicted, json_safe(metrics), dashboard


if __name__ == "__main__":
    config = {
        "symbols": ["AAPL", "MSFT", "TSLA"],
        "start_date": "2020-01-01",
        "end_date": "2026-05-01",
        "model_type": "transformer",  # Options: "random_forest", "lstm", "transformer"
        "sentiment_mode": "random",  # Options: "random", "finbert", "vader"
        "initial_capital": 10000,
        "position_size_pct": 0.10,
        "stop_loss": 0.02,
        "take_profit": 0.04,
        "sequence_length": 20,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "plotly_studio_mode": True,  # Enables Flask web app mode
    }
    try:
        df, metrics, app = run_pipeline(config)
        print(json.dumps(metrics, indent=2))
        if config.get("plotly_studio_mode"):
            print("Flask app available at http://127.0.0.1:8050")
            app.run(debug=True, port=8050, use_reloader=False)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
