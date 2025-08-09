# data_analyzer.py

import pandas as pd
import numpy as np
import json
import markdown
from typing import Tuple, Dict, Any
from io import StringIO

np.random.seed(42)  # deterministic

def statistical_profile(df: pd.DataFrame) -> Dict[str, Any]:
    profile = {}

    # Numeric columns profiling
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    profile['numeric'] = {}
    for col in num_cols:
        col_data = df[col].dropna()
        profile['numeric'][col] = {
            'mean': float(col_data.mean()) if not col_data.empty else None,
            'median': float(col_data.median()) if not col_data.empty else None,
            'variance': float(col_data.var()) if not col_data.empty else None,
            'std_dev': float(col_data.std()) if not col_data.empty else None,
            'missing_pct': round(df[col].isna().mean() * 100, 2),
            'correlations': {},  # fill later
        }

    # Correlation matrix (Pearson) for numeric cols
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr()
        for col in num_cols:
            profile['numeric'][col]['correlations'] = corr_matrix[col].drop(col).to_dict()

    # Categorical columns profiling
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    profile['categorical'] = {}
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        profile['categorical'][col] = {
            'unique_values': int(df[col].nunique(dropna=True)),
            'missing_pct': round(df[col].isna().mean() * 100, 2),
            'top_values': vc.head(10).to_dict()
        }

    return profile


def detect_trends_anomalies(df: pd.DataFrame, window:int=7) -> Dict[str, Any]:
    """
    Detect trends and anomalies using rolling mean, rolling std, and z-score for numeric cols.
    Simple seasonality and anomaly flags.
    """

    results = {}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        series = df[col].dropna()
        if len(series) < window * 3:  # Not enough data
            results[col] = {
                'trend': 'insufficient data',
                'anomalies_count': 0,
            }
            continue

        rolling_mean = series.rolling(window=window, min_periods=window).mean()
        rolling_std = series.rolling(window=window, min_periods=window).std()

        z_scores = (series - rolling_mean) / rolling_std
        # Mark anomaly points as z-score > 3 or < -3
        anomalies = z_scores[(z_scores.abs() > 3)]

        # Basic trend direction: compare mean first 10% vs last 10%
        n = len(series)
        start_mean = series.iloc[:max(1, n // 10)].mean()
        end_mean = series.iloc[-max(1, n // 10):].mean()
        trend = 'stable'
        if end_mean > start_mean * 1.1:
            trend = 'increasing'
        elif end_mean < start_mean * 0.9:
            trend = 'decreasing'

        results[col] = {
            'trend': trend,
            'anomalies_count': int(anomalies.count()),
            'anomalies_indices': anomalies.index.tolist()[:10],  # show up to 10 anomaly points index
        }

    return results


def generate_executive_summary(stat_profile: Dict[str, Any], trends_anomalies: Dict[str, Any]) -> str:
    """
    Generate a simple human-readable plain English summary for CEO level.
    """

    lines = []
    lines.append("# Executive Summary")

    # Numeric trends & anomalies
    lines.append("\n## Numeric Columns Summary:")
    for col, stats in stat_profile.get('numeric', {}).items():
        trend = trends_anomalies.get(col, {}).get('trend', 'unknown')
        anomalies = trends_anomalies.get(col, {}).get('anomalies_count', 0)
        mean = stats.get('mean')
        median = stats.get('median')
        missing = stats.get('missing_pct')

        lines.append(f"- **{col}**: Mean={mean:.2f} Median={median:.2f}, Missing={missing:.1f}%, Trend: {trend}, Anomalies detected: {anomalies}")

    # Categorical summary
    cat_cols = stat_profile.get('categorical', {})
    if cat_cols:
        lines.append("\n## Categorical Columns Summary:")
        for col, stats in cat_cols.items():
            missing = stats.get('missing_pct')
            unique = stats.get('unique_values')
            top_vals = stats.get('top_values', {})
            top_vals_summary = ", ".join([f\"{k} ({v})\" for k, v in list(top_vals.items())[:3]])
            lines.append(f"- **{col}**: Missing={missing:.1f}%, Unique Values={unique}, Top values: {top_vals_summary}")

    # Potential risks or opportunities - simplistic heuristics
    lines.append("\n## Potential Risks & Opportunities:")
    high_missing = [col for col, s in stat_profile.get('numeric', {}).items() if s.get('missing_pct', 0) > 20]
    high_missing += [col for col, s in cat_cols.items() if s.get('missing_pct', 0) > 20]
    if high_missing:
        lines.append("- Columns with >20% missing values might risk data quality issues: " + ", ".join(high_missing))
    else:
        lines.append("- No significant missing data issues detected.")

    high_anomalies = [col for col, d in trends_anomalies.items() if d.get('anomalies_count', 0) > 0]
    if high_anomalies:
        lines.append("- Anomalies detected in columns: " + ", ".join(high_anomalies))
    else:
        lines.append("- No major anomalies detected.")

    return "\n".join(lines)


def analyze_dataframe(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Main callable function.
    Returns tuple of (json_report_utf8_string, markdown_report_utf8_string)
    """

    stat_profile = statistical_profile(df)
    trends_anomalies = detect_trends_anomalies(df)
    summary_md = generate_executive_summary(stat_profile, trends_anomalies)

    # Compose JSON result
    result = {
        "statistical_profile": stat_profile,
        "trends_anomalies": trends_anomalies,
        "executive_summary": summary_md,
    }

    json_report = json.dumps(result, ensure_ascii=False, indent=2)
    return json_report, summary_md


if __name__ == "__main__":
    # simple CLI test
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python data_analyzer.py /path/to/cleaned.csv")
        sys.exit(1)

    df = pd.read_csv(path)
    json_report, markdown_report = analyze_dataframe(df)
    print(markdown_report)
