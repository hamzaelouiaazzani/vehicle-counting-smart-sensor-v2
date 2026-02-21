# Re-run with robust fallback: support common confidences (0.90,0.95,0.99).
# If other confidence requested, try scipy; if scipy missing, raise informative error.
import math
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
import ast
from pathlib import Path

def _get_z_for_confidence(confidence: float) -> float:
    """
    Return z critical value for two-sided confidence level (e.g. 0.95 -> 1.96).
    Supports common levels directly; otherwise tries scipy.stats.norm.ppf; 
    if scipy not present, raises an informative error.
    """
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1 (exclusive)")
    common = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    if confidence in common:
        return common[confidence]
    # compute two-tailed critical z using scipy if available
    try:
        from scipy.stats import norm
        alpha = 1 - confidence
        q = 1 - alpha/2
        return float(norm.ppf(q))
    except Exception:
        raise RuntimeError(
            "Requested confidence level is not a common one (0.90,0.95,0.99) and scipy is not available to compute the z-quantile. "
            "Please either use one of the common confidences or install scipy."
        )

def summary_stats_clt(series: pd.Series, confidence: float = 0.95) -> pd.Series:
    """
    Compute mean, median, variance (sample, ddof=1), std (sample), coefficient of variation (std/mean * 100 %),
    and CLT-based two-sided confidence interval for the mean using z quantile.
    
    Parameters
    ----------
    series : pd.Series or array-like
        Numeric data (can contain NaNs).
    confidence : float, default 0.95
        Confidence level for the interval (0 < confidence < 1). Uses z-quantile (CLT).
    
    Returns
    -------
    pd.Series
        With index: ['n','mean','median','variance','std','cv_percent','se','z','ci_lower','ci_upper']
    """
    # Convert to pandas Series and drop NaNs
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    x = series.dropna().astype(float)
    n = int(x.shape[0])
    if n == 0:
        return pd.Series({
            "n": 0,
            "mean": pd.NA,
            "median": pd.NA,
            "variance": pd.NA,
            "std": pd.NA,
            "cv_percent": pd.NA,
            "se": pd.NA,
            "z": pd.NA,
            "ci_lower": pd.NA,
            "ci_upper": pd.NA
        })
    
    mean = float(x.mean())
    median = float(x.median())
    if n > 1:
        var = float(x.var(ddof=1))
        std = float(x.std(ddof=1))
    else:
        var = float("nan")
        std = float("nan")
    
    cv_percent = (std / mean * 100.0) if (mean != 0 and not math.isnan(std)) else float("nan")
    
    if n > 1 and not math.isnan(std):
        se = std / math.sqrt(n)
        z = _get_z_for_confidence(confidence)
        margin = z * se
        ci_lower = mean - margin
        ci_upper = mean + margin
    else:
        se = float("nan")
        z = float("nan")
        ci_lower = float("nan")
        ci_upper = float("nan")
    
    return pd.Series({
        "n": n,
        "mean": mean,
        "median": median,
        "variance": var,
        "std": std,
        "cv_percent": cv_percent,
        "se": se,
        "z": z,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    })


def compute_detection_statistics(df):
    """
    Compute summary statistics (mean, median, variance, std, CV%, CI)
    for detection-related columns: pre, inf, post, n_detections, latency (if exists).

    Returns:
        DataFrame where each row corresponds to one metric.
    """
    wanted_cols = ["pre", "inf", "post", "n_detections", "latency"]

    stats_rows = []

    for col in wanted_cols:
        if col not in df.columns:
            continue  # skip missing columns

        series = df[col]
        stats = summary_stats_clt(series, confidence=0.95)
        stats["metric"] = col
        stats_rows.append(stats)

    # Combine into one DataFrame
    if stats_rows:
        df_stats = pd.DataFrame(stats_rows).set_index("metric")
    else:
        df_stats = pd.DataFrame()

    return df_stats

def _group_key_to_colname(key: Any) -> str:
    """Convert a group key (scalar or tuple) to a readable column name."""
    if isinstance(key, tuple):
        return "|".join(str(k) for k in key)
    return str(key)

def _compute_group_stats_for_latency(df: pd.DataFrame, groupby_cols: List[str], latency_col: str = "latency",
                                     confidence: float = 0.95) -> pd.DataFrame:
    """
    Generic helper: group df by `groupby_cols`, compute summary_stats_clt on `latency_col` for each group,
    and return DataFrame with rows=stats and columns=group names.
    """
    if not all(col in df.columns for col in groupby_cols):
        missing = [c for c in groupby_cols if c not in df.columns]
        raise ValueError(f"Missing groupby columns in dataframe: {missing}")
    if latency_col not in df.columns:
        raise ValueError(f"Latency column '{latency_col}' not found in dataframe")

    groups = df.groupby(groupby_cols)
    stats_by_group = {}
    for key, group in groups:
        # Extract latency series (dropna)
        latency_series = group[latency_col].dropna().astype(float)
        # If empty -> produce NaN-series with the same index as summary_stats_clt would produce
        if latency_series.shape[0] == 0:
            # use summary_stats_clt on an empty series to get index/structure
            empty_stats = summary_stats_clt(pd.Series([], dtype=float), confidence=confidence)
            stats_by_group[_group_key_to_colname(key)] = empty_stats
        else:
            stats = summary_stats_clt(latency_series, confidence=confidence)
            stats_by_group[_group_key_to_colname(key)] = stats

    if not stats_by_group:
        # no groups -> return empty DF with standard index
        return pd.DataFrame()

    # Combine into DataFrame: index = stat names, columns = group names
    combined = pd.DataFrame(stats_by_group)
    # Ensure a consistent row order
    desired_index = ["n","mean","median","variance","std","cv_percent","se","z","ci_lower","ci_upper"]
    # Keep only present indices and reindex in desired order if possible
    present = [i for i in desired_index if i in combined.index]
    combined = combined.reindex(present)
    return combined

def compute_tracking_statistics(tracking_df: pd.DataFrame,
                                groupby_cols: List[str] = ["tracker"],
                                latency_col: str = "latency",
                                confidence: float = 0.95) -> pd.DataFrame:
    """
    Compute CLT-based summary statistics of `latency` grouped by tracking category.
    Default grouping column: 'tracker'.

    Returns DataFrame with rows = stats and columns = tracker names.
    """
    return _compute_group_stats_for_latency(tracking_df, groupby_cols, latency_col, confidence)

def compute_counting_statistics(counting_df: pd.DataFrame,
                                groupby_cols: List[str] = ["counter_id"],
                                latency_col: str = "latency",
                                confidence: float = 0.95) -> pd.DataFrame:
    """
    Compute CLT-based summary statistics of `latency` grouped by counting category.
    Default grouping column: 'counter_id'.

    Returns DataFrame with rows = stats and columns = counter ids.
    """
    return _compute_group_stats_for_latency(counting_df, groupby_cols, latency_col, confidence)



def same_distribution(s1, s2, alpha=0.05, method='auto'):
    """
    Compare two pd.Series without assuming normality.
    Returns: {
      'same': bool,        # True => cannot reject same-distribution at significance alpha
      'test': str,         # 'ks_2samp' or 'mannwhitneyu'
      'statistic': float,
      'pvalue': float,
      'alpha': float
    }
    method: 'auto'|'ks'|'mw' to force choice
    """
    a = pd.Series(s1).dropna().reset_index(drop=True)
    b = pd.Series(s2).dropna().reset_index(drop=True)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both series must contain at least one non-NaN value.")
    # detect discreteness / many ties (heuristic)
    uniq_ratio = (a.nunique() / len(a) + b.nunique() / len(b)) / 2.0
    use_mw = (method == 'mw') or (method == 'auto' and uniq_ratio < 0.5)

    if method == 'ks' or (method == 'auto' and not use_mw):
        stat, p = stats.ks_2samp(a, b, alternative='two-sided', mode='auto')
        test_name = 'ks_2samp'
    else:
        # Mann-Whitney U is robust for ordinal/discrete data (but tests stochastic ordering)
        stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        test_name = 'mannwhitneyu'

    return {
        'same': bool(p >= float(alpha)),
        'test': test_name,
        'statistic': float(stat),
        'pvalue': float(p),
        'alpha': float(alpha)
    }

def dataset_report(csv_path: str) -> Dict[str, Any]:
    """
    Read dataset CSV and compute descriptive statistics useful for a research paper.

    Input:
      csv_path: path to the CSV file (e.g. r"C:\...\actual_counts.csv")

    Returns:
      {
        'summary': pandas.Series  # overall summary statistics (printable)
        'per_video': pandas.DataFrame  # one row per video with derived fields
        'vehicle_type_totals': pandas.Series # totals by vehicle class
      }
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(p, dtype=str)  # read as str first to be safe
    # standardize column names by stripping spaces
    df.columns = [c.strip() for c in df.columns]

    # convert numeric columns where possible
    numeric_cols = ['ID','total_frames','width','height'] + [c for c in df.columns if c.startswith('actual_')]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # parse datetime
    if 'recording_time' in df.columns:
        df['recording_time'] = pd.to_datetime(df['recording_time'], errors='coerce')

    # parse line_of_counting into coordinates and compute line length (euclidean)
    def parse_line(s):
        if pd.isna(s): 
            return None
        try:
            # some CSVs contain the string with double quotes and inner brackets; ast.literal_eval handles it
            return ast.literal_eval(s)
        except Exception:
            # fallback: try to clean common problems
            try:
                return ast.literal_eval(s.replace('“','"').replace('”','"'))
            except Exception:
                return None

    df['loc_parsed'] = df.get('line_of_counting', pd.NA).apply(parse_line)
    def line_length(coords):
        try:
            p0, p1 = coords
            dx = p0[0] - p1[0]
            dy = p0[1] - p1[1]
            return float((dx*dx + dy*dy) ** 0.5)
        except Exception:
            return np.nan
    df['line_length_px'] = df['loc_parsed'].apply(lambda x: line_length(x) if x is not None else np.nan)

    # total vehicles per video (sum across actual_* columns)
    vehicle_cols = [c for c in df.columns if c.startswith('actual_')]
    if len(vehicle_cols) == 0:
        raise ValueError("No columns prefixed with 'actual_' found.")
    df['total_vehicles'] = df[vehicle_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)

    # derived rates: vehicles per 1000 frames and vehicles per frame
    df['vehicles_per_frame'] = df['total_vehicles'] / df['total_frames']
    df['vehicles_per_1000_frames'] = df['vehicles_per_frame'] * 1000

    # resolution and aspect ratio
    df['resolution'] = df.apply(lambda r: f"{int(r['width'])}x{int(r['height'])}" if pd.notna(r['width']) and pd.notna(r['height']) else None, axis=1)
    def aspect_ratio(w,h):
        try:
            return float(w)/float(h)
        except Exception:
            return np.nan
    df['aspect_ratio'] = df.apply(lambda r: aspect_ratio(r['width'], r['height']), axis=1)

    # categorical counts (direction_relative_to_camera, road_directionality, lighting)
    categorical_cols = [c for c in ['direction_relative_to_camera','road_directionality','lighting'] if c in df.columns]

    # per-video table to return
    per_video_cols = ['video_name','ID','total_frames','recording_time','width','height','resolution','aspect_ratio',
                      'line_length_px','direction_relative_to_camera','road_directionality','lighting','total_vehicles',
                      'vehicles_per_frame','vehicles_per_1000_frames']
    per_video = df.reindex(columns=[c for c in per_video_cols if c in df.columns]).copy()

    # overall summary statistics
    summary = {}

    # counts and time span
    summary['n_videos'] = int(len(df))
    if 'recording_time' in df.columns:
        times = df['recording_time'].dropna()
        if len(times):
            summary['first_recording_time'] = times.min()
            summary['last_recording_time'] = times.max()
            summary['days_covered'] = (times.max().date() - times.min().date()).days
        else:
            summary['first_recording_time'] = pd.NaT
            summary['last_recording_time'] = pd.NaT
            summary['days_covered'] = np.nan
    # frames
    tf = df['total_frames'].dropna().astype(float)
    summary.update({
        'total_frames_sum': float(tf.sum()) if len(tf) else 0.0,
        'total_frames_mean': float(tf.mean()) if len(tf) else np.nan,
        'total_frames_median': float(tf.median()) if len(tf) else np.nan,
        'total_frames_min': float(tf.min()) if len(tf) else np.nan,
        'total_frames_max': float(tf.max()) if len(tf) else np.nan,
        'total_frames_std': float(tf.std()) if len(tf) > 1 else 0.0
    })

    # resolution distribution
    res_counts = df['resolution'].value_counts(dropna=True)
    summary['unique_resolutions'] = int(res_counts.size)
    summary['top_resolution'] = res_counts.index[0] if len(res_counts) else None
    summary['resolution_counts'] = res_counts.to_dict()

    # lighting / directionality distributions
    for c in categorical_cols:
        summary[f"{c}_counts"] = df[c].value_counts(dropna=True).to_dict()

    # line length stats
    ll = df['line_length_px'].dropna().astype(float)
    if len(ll):
        summary.update({
            'line_length_mean_px': float(ll.mean()),
            'line_length_median_px': float(ll.median()),
            'line_length_min_px': float(ll.min()),
            'line_length_max_px': float(ll.max())
        })
    else:
        summary.update({
            'line_length_mean_px': np.nan,
            'line_length_median_px': np.nan,
            'line_length_min_px': np.nan,
            'line_length_max_px': np.nan
        })

    # vehicle counts summary (per class and totals)
    totals_by_class = df[vehicle_cols].apply(pd.to_numeric, errors='coerce').sum(skipna=True)
    totals_by_class.index = [c.replace('actual_','') for c in totals_by_class.index]
    summary['total_vehicles_all_videos'] = float(totals_by_class.sum())
    # per-video total stats
    tv = df['total_vehicles'].dropna().astype(float)
    if len(tv):
        summary.update({
            'total_vehicles_mean': float(tv.mean()),
            'total_vehicles_median': float(tv.median()),
            'total_vehicles_min': float(tv.min()),
            'total_vehicles_max': float(tv.max()),
            'total_vehicles_std': float(tv.std()) if len(tv) > 1 else 0.0
        })
    else:
        summary.update({
            'total_vehicles_mean': np.nan,
            'total_vehicles_median': np.nan,
            'total_vehicles_min': np.nan,
            'total_vehicles_max': np.nan,
            'total_vehicles_std': np.nan
        })

    # vehicles per 1000 frames stats
    v1000 = df['vehicles_per_1000_frames'].dropna().astype(float)
    summary.update({
        'vehicles_per_1000_frames_mean': float(v1000.mean()) if len(v1000) else np.nan,
        'vehicles_per_1000_frames_median': float(v1000.median()) if len(v1000) else np.nan
    })

    # class proportions (global)
    class_props = (totals_by_class / totals_by_class.sum()).fillna(0)
    summary['class_proportions'] = class_props.to_dict()

    # top videos by total_vehicles
    summary['top_videos_by_count'] = df[['video_name','total_vehicles']].sort_values('total_vehicles', ascending=False).head(10).to_dict(orient='records')

    # missing data
    summary['missing_values_per_column'] = df.isna().sum().to_dict()

    # build return object
    result = {
        'summary': pd.Series(summary),
        'per_video': per_video.reset_index(drop=True),
        'vehicle_type_totals': totals_by_class
    }
    return result