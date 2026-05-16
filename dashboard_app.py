from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dashboard_data.json"


def load_dashboard_data() -> dict[str, Any]:
    with open(DATA_PATH, encoding="utf-8") as handle:
        return json.load(handle)


DATA = load_dashboard_data()
PRICES = DATA["prices"]
METRICS = DATA["metrics"]
FEATURE_IMPORTANCE = DATA["feature_importance"]
TRADES = DATA["trades"]
SYMBOLS = sorted({row["symbol"] for row in PRICES}) or METRICS.get("symbols", [])
CLUSTER_FEATURES = ["return_1d", "range_pct", "rolling_volatility", "volume_z"]
CLUSTER_COLORS = ["#12b76a", "#2e90fa", "#f79009", "#f04438"]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def percent_text(value: Any) -> str:
    return f"{safe_float(value) * 100:.2f}%"


def number_text(value: Any, digits: int = 2) -> str:
    return f"{safe_float(value):.{digits}f}"


def filter_prices(symbol: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    return [
        row
        for row in PRICES
        if row["symbol"] == symbol and start_date <= row["date"] <= end_date
    ]


def empty_figure(title: str) -> dict[str, Any]:
    return {
        "data": [],
        "layout": {
            "title": title,
            "height": 360,
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "annotations": [
                {
                    "text": "No data available",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#667085", "size": 14},
                }
            ],
        },
    }


def market_figure(rows: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    if not rows:
        return empty_figure("Market Price")
    dates = [row["date"] for row in rows]
    return {
        "data": [
            {
                "type": "candlestick",
                "name": symbol,
                "x": dates,
                "open": [row["open"] for row in rows],
                "high": [row["high"] for row in rows],
                "low": [row["low"] for row in rows],
                "close": [row["close"] for row in rows],
                "increasing": {"line": {"color": "#039855"}},
                "decreasing": {"line": {"color": "#d92d20"}},
            },
            {
                "type": "bar",
                "name": "Volume",
                "x": dates,
                "y": [row["volume"] for row in rows],
                "yaxis": "y2",
                "marker": {"color": "rgba(21, 101, 192, 0.18)"},
            },
        ],
        "layout": {
            "title": f"{symbol} Market Analysis",
            "height": 560,
            "xaxis": {"rangeslider": {"visible": False}},
            "yaxis": {"title": "Price"},
            "yaxis2": {"title": "Volume", "overlaying": "y", "side": "right", "showgrid": False},
            "legend": {"orientation": "h"},
            "margin": {"l": 58, "r": 58, "t": 58, "b": 44},
        },
    }


def model_figures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    model = METRICS.get("model_performance", {})
    risk = METRICS.get("risk_metrics", {})
    dates = [row["date"] for row in rows]
    equity = [10000.0 for _ in rows]
    drawdown = [0.0 for _ in rows]
    labels = ["SELL", "HOLD", "BUY"]
    cm = model.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    history = model.get("training_history", [])
    top_features = sorted(FEATURE_IMPORTANCE, key=lambda item: item["importance"], reverse=True)[:15]

    return {
        "equity-chart": {
            "data": [{"type": "scatter", "mode": "lines", "x": dates, "y": equity, "name": "Strategy equity"}],
            "layout": {"title": "Strategy Equity Curve", "height": 360, "yaxis": {"title": "Equity"}},
        },
        "drawdown-chart": {
            "data": [{"type": "scatter", "mode": "lines", "fill": "tozeroy", "x": dates, "y": drawdown, "name": "Drawdown"}],
            "layout": {"title": f"Max Drawdown {percent_text(risk.get('max_drawdown_pct'))}", "height": 360, "yaxis": {"tickformat": ".0%"}},
        },
        "confusion-chart": {
            "data": [{"type": "heatmap", "x": labels, "y": labels, "z": cm, "colorscale": "Blues", "showscale": True}],
            "layout": {"title": "Confusion Matrix", "height": 360, "xaxis": {"title": "Predicted"}, "yaxis": {"title": "Actual"}},
        },
        "roc-chart": {
            "data": [
                {"type": "scatter", "mode": "lines+markers", "x": [item["epoch"] for item in history], "y": [item["train_loss"] for item in history], "name": "Train loss"},
                {"type": "scatter", "mode": "lines+markers", "x": [item["epoch"] for item in history], "y": [item["val_loss"] for item in history], "name": "Val loss"},
            ],
            "layout": {"title": "Training History", "height": 360, "xaxis": {"title": "Epoch"}, "yaxis": {"title": "Loss"}},
        },
        "feature-importance-chart": {
            "data": [
                {
                    "type": "bar",
                    "orientation": "h",
                    "x": [item["importance"] for item in reversed(top_features)],
                    "y": [item["feature"] for item in reversed(top_features)],
                    "marker": {"color": "#1565c0"},
                }
            ],
            "layout": {"title": "Feature Importance", "height": 360, "margin": {"l": 118, "r": 20, "t": 52, "b": 38}},
        },
    }


def monthly_returns(rows: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
    if not rows:
        return [], []
    buckets: dict[str, list[float]] = defaultdict(list)
    previous = None
    for row in rows:
        close = safe_float(row["close"])
        if previous:
            buckets[row["date"][:7]].append(close / previous - 1.0)
        previous = close
    labels = sorted(buckets)
    values = []
    for label in labels:
        compounded = 1.0
        for value in buckets[label]:
            compounded *= 1.0 + value
        values.append(compounded - 1.0)
    return labels, values


def trade_figures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels, returns = monthly_returns(rows)
    dates = [row["date"] for row in rows]
    return {
        "monthly-returns-chart": {
            "data": [{"type": "bar", "x": labels, "y": returns, "marker": {"color": "#1565c0"}}],
            "layout": {"title": "Monthly Market Returns", "height": 330, "yaxis": {"tickformat": ".1%"}},
        },
        "winloss-chart": {
            "data": [{"type": "pie", "labels": ["Closed trades"], "values": [1], "hole": 0.62, "marker": {"colors": ["#d0d5dd"]}}],
            "layout": {"title": "Trade Outcomes: 0 Closed Trades", "height": 330, "showlegend": False},
        },
        "cumulative-chart": {
            "data": [{"type": "scatter", "mode": "lines", "x": dates, "y": [0 for _ in dates], "name": "Cumulative PnL"}],
            "layout": {"title": "Cumulative Strategy Return", "height": 330, "yaxis": {"tickformat": ".0%"}},
        },
    }


def correlation_matrix(rows: list[dict[str, Any]]) -> list[list[float]]:
    fields = ["open", "high", "low", "close", "volume"]
    values = [[safe_float(row[field]) for row in rows] for field in fields]

    def corr(a: list[float], b: list[float]) -> float:
        if len(a) < 2 or len(b) < 2:
            return 0.0
        am = sum(a) / len(a)
        bm = sum(b) / len(b)
        av = [x - am for x in a]
        bv = [x - bm for x in b]
        den = math.sqrt(sum(x * x for x in av) * sum(y * y for y in bv))
        return 0.0 if den == 0 else round(sum(x * y for x, y in zip(av, bv)) / den, 4)

    return [[corr(a, b) for b in values] for a in values]


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = average(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def daily_state_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    returns: list[float] = []
    volumes = [safe_float(row.get("volume")) for row in rows]
    volume_mean = average(volumes)
    volume_std = standard_deviation(volumes) or 1.0

    for index, row in enumerate(rows):
        close = safe_float(row.get("close"))
        previous_close = safe_float(rows[index - 1].get("close")) if index else 0.0
        return_1d = 0.0 if index == 0 or previous_close == 0 else close / previous_close - 1.0
        returns.append(return_1d)
        window = returns[max(0, index - 9): index + 1]
        high = safe_float(row.get("high"))
        low = safe_float(row.get("low"))
        range_pct = 0.0 if close == 0 else max(0.0, (high - low) / close)
        volume_z = (safe_float(row.get("volume")) - volume_mean) / volume_std

        points.append(
            {
                "date": row.get("date", ""),
                "close": close,
                "return_1d": return_1d,
                "range_pct": range_pct,
                "rolling_volatility": standard_deviation(window),
                "volume_z": volume_z,
            }
        )
    return points


def standardize_points(points: list[dict[str, Any]], features: list[str]) -> tuple[list[list[float]], list[float], list[float]]:
    columns = [[safe_float(point.get(feature)) for point in points] for feature in features]
    means = [average(column) for column in columns]
    scales = [standard_deviation(column) or 1.0 for column in columns]
    values = [
        [(safe_float(point.get(feature)) - means[index]) / scales[index] for index, feature in enumerate(features)]
        for point in points
    ]
    return values, means, scales


def distance_squared(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(distance_squared(left, right))


def initial_centroids(values: list[list[float]], cluster_count: int) -> list[list[float]]:
    ranked = sorted(
        range(len(values)),
        key=lambda index: (values[index][2], values[index][1], abs(values[index][0]), values[index][3]),
    )
    if cluster_count == 1:
        return [values[ranked[len(ranked) // 2]][:]]
    selected = [ranked[round(position * (len(ranked) - 1) / (cluster_count - 1))] for position in range(cluster_count)]
    return [values[index][:] for index in selected]


def run_kmeans(values: list[list[float]], cluster_count: int = 3, max_iterations: int = 60) -> tuple[list[int], list[list[float]], float]:
    if not values:
        return [], [], 0.0
    cluster_count = max(1, min(cluster_count, len(values)))
    centroids = initial_centroids(values, cluster_count)
    labels = [0 for _ in values]

    for _ in range(max_iterations):
        changed = False
        for index, value in enumerate(values):
            label = min(range(cluster_count), key=lambda cluster: distance_squared(value, centroids[cluster]))
            if label != labels[index]:
                labels[index] = label
                changed = True

        grouped = [[] for _ in range(cluster_count)]
        for label, value in zip(labels, values):
            grouped[label].append(value)
        next_centroids = []
        for cluster, members in enumerate(grouped):
            if not members:
                next_centroids.append(centroids[cluster])
                continue
            next_centroids.append([average([member[feature] for member in members]) for feature in range(len(values[0]))])

        centroid_shift = sum(distance_squared(current, updated) for current, updated in zip(centroids, next_centroids))
        centroids = next_centroids
        if not changed or centroid_shift < 0.000001:
            break

    inertia = sum(distance_squared(value, centroids[label]) for value, label in zip(values, labels))
    return labels, centroids, inertia


def approximate_silhouette(values: list[list[float]], labels: list[int], max_points: int = 240) -> float:
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return 0.0

    step = max(1, len(values) // max_points)
    sample_indexes = list(range(0, len(values), step))[:max_points]
    scores: list[float] = []
    for index in sample_indexes:
        own_label = labels[index]
        same_distances = [
            distance(values[index], values[other])
            for other in sample_indexes
            if other != index and labels[other] == own_label
        ]
        a_score = average(same_distances)
        b_score = min(
            average(
                [
                    distance(values[index], values[other])
                    for other in sample_indexes
                    if labels[other] == other_label
                ]
            )
            for other_label in unique_labels
            if other_label != own_label
        )
        denominator = max(a_score, b_score)
        scores.append(0.0 if denominator == 0 else (b_score - a_score) / denominator)
    return average(scores)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percent)))
    return ordered[index]


def label_cluster_profiles(summaries: list[dict[str, Any]]) -> list[str]:
    if not summaries:
        return []
    highest_vol = max(summaries, key=lambda item: item["mean_volatility"])["cluster"]
    lowest_vol = min(summaries, key=lambda item: item["mean_volatility"])["cluster"]
    highest_return = max(summaries, key=lambda item: item["mean_return"])["cluster"]
    labels: list[str] = []
    used: set[str] = set()
    for summary in summaries:
        cluster = summary["cluster"]
        if cluster == highest_vol and summary["mean_return"] < 0:
            name = "Volatile drawdown"
        elif cluster == highest_vol:
            name = "High-volatility expansion"
        elif cluster == highest_return and summary["mean_return"] > 0:
            name = "Momentum accumulation"
        elif cluster == lowest_vol:
            name = "Calm consolidation"
        elif summary["mean_volume_z"] > 0.25:
            name = "High-volume rotation"
        else:
            name = "Balanced regime"
        if name in used:
            name = f"Cluster {cluster + 1}"
        used.add(name)
        labels.append(name)
    return labels


def build_unsupervised_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    points = daily_state_points(rows)
    if len(points) < 6:
        return {"points": [], "clusters": [], "silhouette": 0.0, "inertia": 0.0, "anomaly_threshold": 0.0}

    values, means, scales = standardize_points(points, CLUSTER_FEATURES)
    cluster_count = min(3, len(points))
    labels, centroids, inertia = run_kmeans(values, cluster_count)
    raw_centroids = [
        {feature: centroids[cluster][index] * scales[index] + means[index] for index, feature in enumerate(CLUSTER_FEATURES)}
        for cluster in range(cluster_count)
    ]

    order = sorted(range(cluster_count), key=lambda cluster: (raw_centroids[cluster]["rolling_volatility"], raw_centroids[cluster]["return_1d"]))
    remap = {old: new for new, old in enumerate(order)}
    labels = [remap[label] for label in labels]
    centroids = [centroids[old] for old in order]
    raw_centroids = [raw_centroids[old] for old in order]

    for point, value, label in zip(points, values, labels):
        point["cluster"] = label
        point["anomaly_score"] = distance(value, centroids[label])
        point["color"] = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]

    summaries: list[dict[str, Any]] = []
    for cluster in range(cluster_count):
        members = [point for point in points if point["cluster"] == cluster]
        summaries.append(
            {
                "cluster": cluster,
                "name": f"Cluster {cluster + 1}",
                "size": len(members),
                "mean_return": average([point["return_1d"] for point in members]),
                "mean_range": average([point["range_pct"] for point in members]),
                "mean_volatility": average([point["rolling_volatility"] for point in members]),
                "mean_volume_z": average([point["volume_z"] for point in members]),
                "max_anomaly": max([point["anomaly_score"] for point in members], default=0.0),
                "color": CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)],
                "centroid": raw_centroids[cluster],
            }
        )

    names = label_cluster_profiles(summaries)
    for summary, name in zip(summaries, names):
        summary["name"] = name
    for point in points:
        point["cluster_name"] = summaries[point["cluster"]]["name"]

    return {
        "points": points,
        "clusters": summaries,
        "silhouette": approximate_silhouette(values, labels),
        "inertia": inertia,
        "anomaly_threshold": percentile([point["anomaly_score"] for point in points], 0.95),
    }


def unsupervised_figures(analysis: dict[str, Any]) -> dict[str, Any]:
    points = analysis.get("points", [])
    clusters = analysis.get("clusters", [])
    if not points:
        return {
            "cluster-scatter-chart": empty_figure("K-Means Regime Map"),
            "cluster-timeline-chart": empty_figure("Cluster Timeline"),
            "cluster-profile-chart": empty_figure("Cluster Profile"),
            "anomaly-chart": empty_figure("Centroid Distance Anomalies"),
        }

    scatter_traces = []
    timeline_traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": [point["date"] for point in points],
            "y": [point["close"] for point in points],
            "name": "Close",
            "line": {"color": "#667085", "width": 1.4},
        }
    ]
    for cluster in clusters:
        members = [point for point in points if point["cluster"] == cluster["cluster"]]
        scatter_traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": cluster["name"],
                "x": [point["return_1d"] for point in members],
                "y": [point["rolling_volatility"] for point in members],
                "text": [point["date"] for point in members],
                "customdata": [[point["range_pct"], point["volume_z"], point["anomaly_score"]] for point in members],
                "marker": {"size": 7, "color": cluster["color"], "opacity": 0.72, "line": {"width": 0}},
                "hovertemplate": "%{text}<br>Return %{x:.2%}<br>Volatility %{y:.2%}<br>Range %{customdata[0]:.2%}<br>Volume z %{customdata[1]:.2f}<br>Distance %{customdata[2]:.2f}<extra>%{fullData.name}</extra>",
            }
        )
        timeline_traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": cluster["name"],
                "x": [point["date"] for point in members],
                "y": [point["close"] for point in members],
                "marker": {"size": 7, "color": cluster["color"], "opacity": 0.76},
                "hovertemplate": "%{x}<br>Close %{y:.2f}<extra>%{fullData.name}</extra>",
            }
        )

    cluster_names = [cluster["name"] for cluster in clusters]
    anomaly_threshold = safe_float(analysis.get("anomaly_threshold"))
    top_anomalies = [point for point in points if point["anomaly_score"] >= anomaly_threshold]

    return {
        "cluster-scatter-chart": {
            "data": scatter_traces,
            "layout": {
                "title": "K-Means Regime Map",
                "height": 420,
                "xaxis": {"title": "Daily return", "tickformat": ".1%"},
                "yaxis": {"title": "10-day volatility", "tickformat": ".1%"},
                "legend": {"orientation": "h"},
            },
        },
        "cluster-timeline-chart": {
            "data": timeline_traces,
            "layout": {"title": "Price Timeline by Cluster", "height": 380, "yaxis": {"title": "Close"}, "legend": {"orientation": "h"}},
        },
        "cluster-profile-chart": {
            "data": [
                {
                    "type": "bar",
                    "name": "Days",
                    "x": cluster_names,
                    "y": [cluster["size"] for cluster in clusters],
                    "marker": {"color": [cluster["color"] for cluster in clusters]},
                },
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "Avg return",
                    "x": cluster_names,
                    "y": [cluster["mean_return"] * 100 for cluster in clusters],
                    "yaxis": "y2",
                    "line": {"color": "#f04438", "width": 2},
                    "marker": {"size": 8},
                },
            ],
            "layout": {
                "title": "Cluster Profile",
                "height": 360,
                "yaxis": {"title": "Days"},
                "yaxis2": {"title": "Avg return (%)", "overlaying": "y", "side": "right", "showgrid": False},
                "legend": {"orientation": "h"},
            },
        },
        "anomaly-chart": {
            "data": [
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Centroid distance",
                    "x": [point["date"] for point in points],
                    "y": [point["anomaly_score"] for point in points],
                    "line": {"color": "#2e90fa", "width": 1.6},
                },
                {
                    "type": "scatter",
                    "mode": "markers",
                    "name": "Top 5% distance",
                    "x": [point["date"] for point in top_anomalies],
                    "y": [point["anomaly_score"] for point in top_anomalies],
                    "marker": {"size": 8, "color": "#f04438"},
                },
            ],
            "layout": {
                "title": "Centroid Distance Anomalies",
                "height": 360,
                "yaxis": {"title": "Distance"},
                "legend": {"orientation": "h"},
                "shapes": [
                    {
                        "type": "line",
                        "xref": "paper",
                        "x0": 0,
                        "x1": 1,
                        "y0": anomaly_threshold,
                        "y1": anomaly_threshold,
                        "line": {"color": "#f04438", "width": 1, "dash": "dot"},
                    }
                ],
            },
        },
    }


def cluster_cards(analysis: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "name": cluster["name"],
            "days": f"{cluster['size']} days",
            "return": percent_text(cluster["mean_return"]),
            "volatility": percent_text(cluster["mean_volatility"]),
            "range": percent_text(cluster["mean_range"]),
            "volume": number_text(cluster["mean_volume_z"]),
            "color": cluster["color"],
        }
        for cluster in analysis.get("clusters", [])
    ]


def multimodal_figures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dates = [row["date"] for row in rows]
    sentiment = []
    previous = None
    for row in rows:
        close = safe_float(row["close"])
        sentiment.append(0.0 if not previous else max(-1.0, min(1.0, (close / previous - 1.0) * 25)))
        previous = close
    attention = [[round(math.exp(-abs(i - j) / 4), 4) for j in range(20)] for i in range(20)]
    corr_labels = ["open", "high", "low", "close", "volume"]
    return {
        "sentiment-chart": {
            "data": [{"type": "scatter", "mode": "lines", "x": dates, "y": sentiment, "name": "Sentiment proxy"}],
            "layout": {"title": "Sentiment Signal Proxy", "height": 360, "yaxis": {"range": [-1, 1]}},
        },
        "attention-chart": {
            "data": [{"type": "heatmap", "z": attention, "colorscale": "Viridis"}],
            "layout": {"title": "Transformer Attention Shape 20 x 20", "height": 360},
        },
        "correlation-chart": {
            "data": [{"type": "heatmap", "x": corr_labels, "y": corr_labels, "z": correlation_matrix(rows), "colorscale": "RdBu", "zmin": -1, "zmax": 1}],
            "layout": {"title": "Feature Correlation", "height": 420},
        },
    }


def figures_to_payload(rows: list[dict[str, Any]], symbol: str, analysis: dict[str, Any]) -> str:
    figures = {"market-chart": market_figure(rows, symbol)}
    figures.update(model_figures(rows))
    figures.update(trade_figures(rows))
    figures.update(multimodal_figures(rows))
    figures.update(unsupervised_figures(analysis))
    return json.dumps(figures)


def trade_table_html() -> str:
    if not TRADES:
        return '<p class="muted">No closed trades were generated for the current prediction threshold.</p>'
    headers = list(TRADES[0])
    rows = ["<table class=\"trade-table sortable\"><thead><tr>"]
    rows.extend(f"<th>{header}</th>" for header in headers)
    rows.append("</tr></thead><tbody>")
    for trade in TRADES:
        rows.append("<tr>")
        rows.extend(f"<td>{trade.get(header, '')}</td>" for header in headers)
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


FLASK_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multimodal Financial Analysis</title>
  <script>
    const storedDashboardTheme = localStorage.getItem("financial-dashboard-theme");
    const prefersDarkDashboard = window.matchMedia?.("(prefers-color-scheme: dark)")?.matches;
    document.documentElement.dataset.theme = storedDashboardTheme || (prefersDarkDashboard ? "dark" : "light");
  </script>
  <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js"
      }
    }
  </script>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <script src="https://unpkg.com/lucide@0.468.0/dist/umd/lucide.min.js"></script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --bg-accent: #eaf1fb;
      --panel: #ffffff;
      --panel-strong: #fdfefe;
      --ink: #18212f;
      --muted: #667085;
      --line: #d9e0ea;
      --accent: #1565c0;
      --accent-dark: #0f4f96;
      --success: #039855;
      --danger: #d92d20;
      --shadow: 0 18px 46px rgba(16, 24, 40, 0.08);
      --chart-bg: #ffffff;
      --control-bg: #ffffff;
      --pre-bg: #f8fafc;
      --glass: rgba(255, 255, 255, 0.78);
      --glass-strong: rgba(255, 255, 255, 0.9);
      --canvas-opacity: 0.58;
    }
    html[data-theme="dark"] {
      color-scheme: dark;
      --bg: #09111f;
      --bg-accent: #0f2138;
      --panel: #101a2a;
      --panel-strong: #142238;
      --ink: #edf4ff;
      --muted: #9fb0c7;
      --line: #26364f;
      --accent: #55a6ff;
      --accent-dark: #2f7ed8;
      --success: #37c98b;
      --danger: #ff6b66;
      --shadow: 0 22px 56px rgba(0, 0, 0, 0.32);
      --chart-bg: #101a2a;
      --control-bg: #0c1726;
      --pre-bg: #0c1726;
      --glass: rgba(16, 26, 42, 0.72);
      --glass-strong: rgba(20, 34, 56, 0.86);
      --canvas-opacity: 0.86;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(135deg, rgba(21, 101, 192, 0.08), transparent 34%),
        linear-gradient(180deg, var(--bg-accent), var(--bg) 34rem);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
      transition: background 220ms ease, color 220ms ease;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      z-index: 1;
      pointer-events: none;
      opacity: 0.42;
      background-image:
        linear-gradient(rgba(102, 112, 133, 0.12) 1px, transparent 1px),
        linear-gradient(90deg, rgba(102, 112, 133, 0.12) 1px, transparent 1px);
      background-size: 44px 44px;
      mask-image: linear-gradient(to bottom, black, transparent 72%);
    }
    .market-depth-canvas {
      position: fixed;
      inset: 0;
      z-index: 0;
      width: 100%;
      height: 100%;
      opacity: var(--canvas-opacity);
      pointer-events: none;
      transition: opacity 220ms ease;
    }
    .market-depth-sheen {
      position: fixed;
      inset: 0;
      z-index: 1;
      pointer-events: none;
      background:
        linear-gradient(115deg, transparent 0 28%, color-mix(in srgb, var(--accent) 14%, transparent) 42%, transparent 56%),
        repeating-linear-gradient(90deg, transparent 0 88px, color-mix(in srgb, var(--line) 34%, transparent) 89px 90px);
      opacity: 0.28;
      mix-blend-mode: screen;
      mask-image: linear-gradient(to bottom, black, transparent 84%);
    }
    .shell { position: relative; z-index: 2; width: min(1520px, 96vw); margin: 0 auto; padding: 24px 0 38px; }
    header { display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 16px; padding: 16px 4.7rem 16px 16px; border: 1px solid var(--line); border-radius: 8px; background: var(--glass); box-shadow: var(--shadow); backdrop-filter: blur(20px) saturate(1.18); animation: surfaceIn 420ms ease both; }
    h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: 0; }
    .muted { color: var(--muted); }
    .theme-toggle {
      position: fixed;
      right: 18px;
      top: 18px;
      z-index: 20;
      display: inline-grid;
      place-items: center;
      width: 44px;
      height: 44px;
      padding: 0;
      border-radius: 50%;
      border: 1px solid var(--line);
      background: var(--glass-strong);
      color: var(--ink);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
      transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
    }
    .theme-toggle:hover { transform: translateY(-1px); border-color: var(--accent); background: var(--panel-strong); }
    .theme-toggle svg { width: 19px; height: 19px; }
    html[data-theme="light"] .theme-toggle .sun-icon,
    html[data-theme="dark"] .theme-toggle .moon-icon { display: none; }
    .filters { background: var(--glass); border: 1px solid var(--line); border-radius: 8px; padding: 12px; display: grid; grid-template-columns: 180px 180px 180px auto; gap: 10px; align-items: end; margin-bottom: 16px; box-shadow: var(--shadow); backdrop-filter: blur(18px) saturate(1.14); animation: surfaceIn 420ms ease both; }
    label { font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }
    select, input, button { width: 100%; height: 38px; border: 1px solid var(--line); border-radius: 6px; background: var(--control-bg); color: var(--ink); padding: 0 10px; font-size: 14px; transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease, background 140ms ease; }
    select:focus-visible, input:focus-visible, button:focus-visible { outline: 2px solid color-mix(in srgb, var(--accent) 42%, transparent); outline-offset: 2px; }
    button { background: var(--accent); color: #fff; border-color: var(--accent); cursor: pointer; }
    button:hover { background: var(--accent-dark); transform: translateY(-1px); }
    .metrics { display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap: 10px; margin-bottom: 16px; }
    .metric { position: relative; overflow: hidden; background: var(--glass); border: 1px solid var(--line); border-radius: 8px; padding: 12px; box-shadow: var(--shadow); backdrop-filter: blur(16px) saturate(1.12); animation: surfaceIn 420ms ease both; transition: transform 160ms ease, border-color 160ms ease, background 160ms ease; }
    .metric::before { content: ""; position: absolute; inset: 0 0 auto; height: 3px; background: linear-gradient(90deg, var(--accent), var(--success)); }
    .metric::after { content: ""; position: absolute; inset: auto 10px 10px; height: 1px; background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--accent) 56%, transparent), transparent); transform: scaleX(0.2); opacity: 0.6; transition: transform 180ms ease; }
    .metric:hover { transform: translateY(-2px); border-color: color-mix(in srgb, var(--accent) 48%, var(--line)); }
    .metric:hover::after { transform: scaleX(1); }
    .metric span { color: var(--muted); font-size: 12px; display: block; }
    .metric strong { font-size: 20px; margin-top: 4px; display: block; }
    .cluster-cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-bottom: 12px; }
    .cluster-card { position: relative; overflow: hidden; background: var(--glass); border: 1px solid var(--line); border-left: 4px solid var(--cluster-color); border-radius: 8px; padding: 12px; box-shadow: var(--shadow); backdrop-filter: blur(16px) saturate(1.12); }
    .cluster-card span { display: block; color: var(--muted); font-size: 12px; }
    .cluster-card strong { display: block; margin: 3px 0 8px; font-size: 18px; }
    .cluster-card dl { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 5px 10px; margin: 0; font-size: 12px; }
    .cluster-card div { min-width: 0; }
    .cluster-card dt { color: var(--muted); }
    .cluster-card dd { margin: 0; font-weight: 700; }
    .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .tab-button { width: auto; background: var(--glass); color: var(--ink); border: 1px solid var(--line); min-width: 148px; box-shadow: 0 8px 20px rgba(16, 24, 40, 0.04); backdrop-filter: blur(14px); }
    .tab-button.active { background: var(--accent); border-color: var(--accent); color: #fff; box-shadow: 0 12px 26px color-mix(in srgb, var(--accent) 26%, transparent); }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; animation: panelIn 260ms ease both; }
    .grid-2, .grid-3 { display: grid; gap: 12px; margin-bottom: 12px; }
    .grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .chart, .table-panel, .json-panel { position: relative; overflow: hidden; background: var(--glass-strong); border: 1px solid var(--line); border-radius: 8px; padding: 8px; min-width: 0; box-shadow: var(--shadow); backdrop-filter: blur(16px) saturate(1.12); transition: transform 160ms ease, border-color 160ms ease, background 180ms ease; }
    .chart:hover, .table-panel:hover, .json-panel:hover { transform: translateY(-1px); border-color: color-mix(in srgb, var(--accent) 34%, var(--line)); }
    .chart::after {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(105deg, transparent 0 42%, rgba(255, 255, 255, 0.18) 50%, transparent 58% 100%);
      transform: translateX(-125%);
      animation: chartSweep 4.8s ease-in-out infinite;
    }
    .trade-table { width: 100%; border-collapse: collapse; font-size: 12px; white-space: nowrap; }
    .trade-table th, .trade-table td { border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; }
    .trade-table th { cursor: pointer; color: var(--muted); background: color-mix(in srgb, var(--panel-strong) 86%, var(--accent) 6%); }
    .trade-table th:first-child, .trade-table td:first-child { text-align: left; }
    .table-scroll { overflow-x: auto; }
    pre { overflow-x: auto; margin: 0; font-size: 12px; color: var(--ink); background: var(--pre-bg); border-radius: 6px; padding: 10px; }
    @keyframes surfaceIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes panelIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes chartSweep {
      0%, 56% { transform: translateX(-125%); }
      82%, 100% { transform: translateX(125%); }
    }
    @media (max-width: 900px) {
      header { display: block; padding-right: 3.7rem; }
      .theme-toggle { right: 12px; top: 12px; }
      .filters, .metrics, .cluster-cards, .grid-2, .grid-3 { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration: 0.001ms !important; animation-iteration-count: 1 !important; transition-duration: 0.001ms !important; }
    }
  </style>
</head>
<body>
  <canvas id="market-depth-canvas" class="market-depth-canvas" aria-hidden="true"></canvas>
  <div class="market-depth-sheen" aria-hidden="true"></div>
  <button id="theme-toggle" class="theme-toggle" type="button" aria-label="Toggle dark and light mode" title="Toggle theme">
    <i class="moon-icon" data-lucide="moon"></i>
    <i class="sun-icon" data-lucide="sun"></i>
  </button>
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
      <button class="tab-button" type="button" data-tab="unsupervised">Unsupervised Clustering</button>
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

    <section id="unsupervised" class="tab-panel">
      <div class="cluster-cards">
        {% for card in cluster_cards %}
          <article class="cluster-card" style="--cluster-color: {{ card.color }}">
            <span>{{ card.name }}</span>
            <strong>{{ card.days }}</strong>
            <dl>
              <div><dt>Avg return</dt><dd>{{ card.return }}</dd></div>
              <div><dt>Volatility</dt><dd>{{ card.volatility }}</dd></div>
              <div><dt>Range</dt><dd>{{ card.range }}</dd></div>
              <div><dt>Volume z</dt><dd>{{ card.volume }}</dd></div>
            </dl>
          </article>
        {% endfor %}
      </div>
      <div class="grid-2">
        <div class="chart" id="cluster-scatter-chart"></div>
        <div class="chart" id="cluster-timeline-chart"></div>
      </div>
      <div class="grid-2">
        <div class="chart" id="cluster-profile-chart"></div>
        <div class="chart" id="anomaly-chart"></div>
      </div>
    </section>
  </main>

  <script>
    const figures = {{ plot_payload|safe }};
    const chartIds = Object.keys(figures);

    function currentThemeColors() {
      const styles = getComputedStyle(document.documentElement);
      return {
        chartBg: styles.getPropertyValue("--chart-bg").trim(),
        ink: styles.getPropertyValue("--ink").trim(),
        muted: styles.getPropertyValue("--muted").trim(),
        line: styles.getPropertyValue("--line").trim()
      };
    }

    function themedLayout(layout = {}) {
      const colors = currentThemeColors();
      return {
        ...layout,
        paper_bgcolor: colors.chartBg,
        plot_bgcolor: colors.chartBg,
        font: { ...(layout.font || {}), color: colors.ink },
        xaxis: {
          ...(layout.xaxis || {}),
          gridcolor: colors.line,
          linecolor: colors.line,
          tickfont: { color: colors.muted },
          title: { ...(layout.xaxis?.title && typeof layout.xaxis.title === "object" ? layout.xaxis.title : { text: layout.xaxis?.title }), font: { color: colors.muted } }
        },
        yaxis: {
          ...(layout.yaxis || {}),
          gridcolor: colors.line,
          linecolor: colors.line,
          tickfont: { color: colors.muted },
          title: { ...(layout.yaxis?.title && typeof layout.yaxis.title === "object" ? layout.yaxis.title : { text: layout.yaxis?.title }), font: { color: colors.muted } }
        }
      };
    }

    Object.entries(figures).forEach(([id, fig]) => {
      Plotly.newPlot(id, fig.data, themedLayout(fig.layout), {responsive: true, displaylogo: false});
    });

    function applyTheme(theme) {
      document.documentElement.dataset.theme = theme;
      localStorage.setItem("financial-dashboard-theme", theme);
      chartIds.forEach((id) => {
        if (document.getElementById(id)) {
          Plotly.relayout(id, themedLayout(figures[id].layout));
        }
      });
    }

    window.lucide?.createIcons();
    document.querySelector("#theme-toggle").addEventListener("click", () => {
      applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
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
  <script type="module">
    import * as THREE from "three";

    const canvas = document.querySelector("#market-depth-canvas");
    const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;

    if (canvas && !reduceMotion) {
      const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, preserveDrawingBuffer: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
      renderer.setClearColor(0x000000, 0);

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 100);
      camera.position.set(0, 1.15, 8.4);

      const root = new THREE.Group();
      scene.add(root);

      const pointer = new THREE.Vector2(0, 0);
      const palette = {
        accent: new THREE.Color(),
        success: new THREE.Color(),
        danger: new THREE.Color(),
        muted: new THREE.Color()
      };

      function cssColor(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      }

      function syncPalette() {
        palette.accent.set(cssColor("--accent") || "#1565c0");
        palette.success.set(cssColor("--success") || "#039855");
        palette.danger.set(cssColor("--danger") || "#d92d20");
        palette.muted.set(cssColor("--muted") || "#667085");
        particleMaterial.color.copy(palette.accent);
        surfaceMaterial.color.copy(palette.accent);
        ribbonMaterials.forEach((material, index) => {
          material.color.copy(index % 3 === 0 ? palette.success : index % 3 === 1 ? palette.accent : palette.danger);
        });
        barMaterial.color.copy(palette.success);
      }

      const particleCount = window.innerWidth < 760 ? 360 : 860;
      const particlePositions = new Float32Array(particleCount * 3);
      const particleSeeds = new Float32Array(particleCount);
      for (let i = 0; i < particleCount; i += 1) {
        const i3 = i * 3;
        particlePositions[i3] = (Math.random() - 0.5) * 16;
        particlePositions[i3 + 1] = (Math.random() - 0.5) * 8;
        particlePositions[i3 + 2] = (Math.random() - 0.5) * 8;
        particleSeeds[i] = Math.random() * Math.PI * 2;
      }
      const particleGeometry = new THREE.BufferGeometry();
      particleGeometry.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
      const particleMaterial = new THREE.PointsMaterial({ size: 0.026, transparent: true, opacity: 0.72, depthWrite: false });
      const particles = new THREE.Points(particleGeometry, particleMaterial);
      root.add(particles);

      const surfaceGeometry = new THREE.PlaneGeometry(18, 10, 82, 42);
      const surfaceBase = surfaceGeometry.attributes.position.array.slice();
      const surfaceMaterial = new THREE.MeshBasicMaterial({ wireframe: true, transparent: true, opacity: 0.2, depthWrite: false });
      const surface = new THREE.Mesh(surfaceGeometry, surfaceMaterial);
      surface.rotation.x = -1.12;
      surface.position.set(0, -2.65, -1.8);
      root.add(surface);

      const ribbonMaterials = [];
      const ribbons = [];
      for (let r = 0; r < 7; r += 1) {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(150 * 3);
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        const material = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.45, depthWrite: false });
        ribbonMaterials.push(material);
        const line = new THREE.Line(geometry, material);
        line.userData = { phase: r * 0.72, lift: -1.6 + r * 0.48, spread: 0.22 + r * 0.035 };
        ribbons.push(line);
        root.add(line);
      }

      const barCount = window.innerWidth < 760 ? 48 : 96;
      const barGeometry = new THREE.BoxGeometry(0.045, 1, 0.045);
      const barMaterial = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.58, depthWrite: false });
      const bars = new THREE.InstancedMesh(barGeometry, barMaterial, barCount);
      const dummy = new THREE.Object3D();
      root.add(bars);

      function resize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      }

      function updateSurface(time) {
        const positions = surfaceGeometry.attributes.position.array;
        for (let i = 0; i < positions.length; i += 3) {
          const x = surfaceBase[i];
          const y = surfaceBase[i + 1];
          positions[i + 2] = Math.sin(x * 0.82 + time * 0.78) * 0.16 + Math.cos(y * 1.18 + time * 0.5) * 0.12;
        }
        surfaceGeometry.attributes.position.needsUpdate = true;
      }

      function updateRibbons(time) {
        ribbons.forEach((line, ribbonIndex) => {
          const positions = line.geometry.attributes.position.array;
          const { phase, lift, spread } = line.userData;
          for (let i = 0; i < 150; i += 1) {
            const t = i / 149;
            const i3 = i * 3;
            const x = (t - 0.5) * 15;
            positions[i3] = x;
            positions[i3 + 1] = lift + Math.sin(t * Math.PI * 6 + time + phase) * spread;
            positions[i3 + 2] = -2.2 + ribbonIndex * 0.34 + Math.cos(t * Math.PI * 4 + time * 0.6 + phase) * 0.32;
          }
          line.geometry.attributes.position.needsUpdate = true;
        });
      }

      function updateBars(time) {
        for (let i = 0; i < barCount; i += 1) {
          const t = i / Math.max(1, barCount - 1);
          const height = 0.32 + Math.pow(Math.sin(time * 1.4 + i * 0.31) * 0.5 + 0.5, 1.7) * 1.45;
          dummy.position.set((t - 0.5) * 12, -3.35 + height * 0.5, -0.4 + Math.sin(i * 0.33) * 0.5);
          dummy.scale.set(1, height, 1);
          dummy.rotation.z = Math.sin(time * 0.4 + i) * 0.025;
          dummy.updateMatrix();
          bars.setMatrixAt(i, dummy.matrix);
        }
        bars.instanceMatrix.needsUpdate = true;
      }

      function animate(now) {
        const time = now * 0.001;
        root.rotation.y = THREE.MathUtils.lerp(root.rotation.y, pointer.x * 0.08, 0.035);
        root.rotation.x = THREE.MathUtils.lerp(root.rotation.x, -pointer.y * 0.035, 0.035);
        particles.rotation.y = time * 0.025;
        particles.rotation.x = Math.sin(time * 0.13) * 0.025;

        const positions = particleGeometry.attributes.position.array;
        for (let i = 0; i < particleCount; i += 1) {
          const i3 = i * 3;
          positions[i3 + 1] += Math.sin(time * 0.8 + particleSeeds[i]) * 0.0009;
          if (positions[i3 + 1] > 4.3) positions[i3 + 1] = -4.3;
        }
        particleGeometry.attributes.position.needsUpdate = true;

        updateSurface(time);
        updateRibbons(time);
        updateBars(time);
        renderer.render(scene, camera);
        requestAnimationFrame(animate);
      }

      window.addEventListener("resize", resize);
      window.addEventListener("pointermove", (event) => {
        pointer.x = event.clientX / window.innerWidth - 0.5;
        pointer.y = event.clientY / window.innerHeight - 0.5;
      }, { passive: true });

      new MutationObserver(syncPalette).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
      resize();
      syncPalette();
      requestAnimationFrame(animate);
    }
  </script>
</body>
</html>
"""


app = Flask(__name__)


@app.get("/")
def index() -> str:
    min_date = min(row["date"] for row in PRICES)
    max_date = max(row["date"] for row in PRICES)
    selected_symbol = request.args.get("symbol") or (SYMBOLS[0] if SYMBOLS else "")
    if selected_symbol not in SYMBOLS and SYMBOLS:
        selected_symbol = SYMBOLS[0]
    selected_start = request.args.get("start_date") or min_date
    selected_end = request.args.get("end_date") or max_date
    if selected_start < min_date:
        selected_start = min_date
    if selected_end > max_date:
        selected_end = max_date

    rows = filter_prices(selected_symbol, selected_start, selected_end)
    unsupervised_analysis = build_unsupervised_analysis(rows)
    risk = METRICS.get("risk_metrics", {})
    model = METRICS.get("model_performance", {})
    metric_cards = [
        {"label": "Total Return", "value": percent_text(risk.get("total_return", 0.0))},
        {"label": "Sharpe", "value": number_text(risk.get("sharpe_ratio", 0.0))},
        {"label": "Max Drawdown", "value": percent_text(risk.get("max_drawdown_pct", 0.0))},
        {"label": "Accuracy", "value": percent_text(model.get("accuracy", 0.0))},
        {"label": "Trades", "value": str(int(safe_float(risk.get("number_of_trades", 0))))},
        {"label": "Regimes", "value": str(len(unsupervised_analysis.get("clusters", [])))},
        {"label": "Cluster Score", "value": number_text(unsupervised_analysis.get("silhouette", 0.0))},
    ]
    return render_template_string(
        FLASK_TEMPLATE,
        plot_payload=figures_to_payload(rows, selected_symbol, unsupervised_analysis),
        trade_table=trade_table_html(),
        cluster_cards=cluster_cards(unsupervised_analysis),
        symbol_options=[{"symbol": symbol, "selected": symbol == selected_symbol} for symbol in SYMBOLS],
        selected_start=selected_start,
        selected_end=selected_end,
        min_date=min_date,
        max_date=max_date,
        metric_cards=metric_cards,
        row_count=METRICS.get("rows", len(PRICES)),
        model_type=METRICS.get("model_type", ""),
        sentiment_mode=METRICS.get("sentiment_mode", ""),
        metrics_json=json.dumps(METRICS, indent=2),
    )


@app.get("/metrics.json")
def metrics_json_route():
    return jsonify(METRICS)


@app.get("/unsupervised.json")
def unsupervised_json_route():
    min_date = min(row["date"] for row in PRICES)
    max_date = max(row["date"] for row in PRICES)
    selected_symbol = request.args.get("symbol") or (SYMBOLS[0] if SYMBOLS else "")
    if selected_symbol not in SYMBOLS and SYMBOLS:
        selected_symbol = SYMBOLS[0]
    selected_start = request.args.get("start_date") or min_date
    selected_end = request.args.get("end_date") or max_date
    rows = filter_prices(selected_symbol, selected_start, selected_end)
    return jsonify(build_unsupervised_analysis(rows))


@app.get("/health")
def health():
    return jsonify({"ok": True, "name": "Multimodal Financial Analysis", "rows": METRICS.get("rows", len(PRICES))})
