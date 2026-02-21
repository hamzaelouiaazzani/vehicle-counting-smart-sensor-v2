"""
Generic evaluation/metrics module adapted from your original script.
- Works with any detector class map (auto-detects predicted_<class> columns or reads class names from meta).
- Keeps 'van' separate if present and includes van in 4-wheel aggregates.
- Produces the same error metrics (mae, signed_mae, mape, signed_mape) and supports algorithm-level aggregation.
- Backwards-compatible helpers: keep_min_vicinity, aggregate_metrics_by_algorithm, etc.
"""

import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Dict, Callable, List, Any, Optional, Sequence, Union
from pathlib import Path
import re

# Optional: user may provide parse_exp_id function; fallback below
def default_parse_exp_id(exp_id: str) -> Dict[str, Any]:
    parts = exp_id.split("__")
    return {
        "sequence": parts[0] if len(parts) > 0 else None,
        "counter": parts[1] if len(parts) > 1 else None,
        "stride": int(parts[2].lstrip("s")) if len(parts) > 2 else None,
        "tracker": None if len(parts) <= 3 or parts[3].split("tracker-")[1] == "none" else parts[3].split("tracker-")[1],
        "vicinity": None if len(parts) <= 4 or parts[4].split("vic-")[1] == "none" else float(parts[4].split("vic-")[1]),
    }

def get_algorithm_id_from_exp(exp_id: str) -> str:
    parts = exp_id.split("__", 1)
    return parts[1] if len(parts) > 1 else exp_id

# ------------------------
# Low-level errors (unchanged)
# ------------------------
def _to_aligned_series(actual: Iterable, predicted: Iterable) -> Tuple[pd.Series, pd.Series]:
    actual_s = pd.Series(actual).astype(float)
    pred_s = pd.Series(predicted).astype(float)
    if not actual_s.index.equals(pred_s.index):
        pred_s = pred_s.reindex(actual_s.index)
    return actual_s, pred_s

def mae_series(actual: Iterable, predicted: Iterable) -> pd.Series:
    a, p = _to_aligned_series(actual, predicted)
    return (p - a).abs()

def signed_mae_series(actual: Iterable, predicted: Iterable) -> pd.Series:
    a, p = _to_aligned_series(actual, predicted)
    return (p - a)

def mape_series(actual: Iterable, predicted: Iterable) -> pd.Series:
    a, p = _to_aligned_series(actual, predicted)
    abs_err = (p - a).abs()
    with np.errstate(divide='ignore', invalid='ignore'):
        out = abs_err / a
    out = out.replace([np.inf, -np.inf], np.nan)
    out[a == 0] = np.nan
    return out

def signed_mape_series(actual: Iterable, predicted: Iterable) -> pd.Series:
    a, p = _to_aligned_series(actual, predicted)
    err = (p - a)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = err / a
    out = out.replace([np.inf, -np.inf], np.nan)
    out[a == 0] = np.nan
    return out

# ------------------------
# Helpers to detect classes & columns
# ------------------------
def detect_predicted_class_names(df: pd.DataFrame, meta: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Detect class names from df by scanning columns prefixed with 'predicted_' (excluding group names).
    If meta contains 'detector_class_names' (list), prefer that.
    Returns a list of class names in the detected order.
    """
    if meta and isinstance(meta.get("detector_class_names"), list) and meta.get("detector_class_names"):
        return [str(x) for x in meta["detector_class_names"]]

    preds = []
    for col in df.columns:
        if not isinstance(col, str):
            continue
        if not col.startswith("predicted_"):
            continue
        name = col[len("predicted_"):]
        # skip aggregated columns '2_3_wheels' and '4_wheels' if present
        if name in ("2_3_wheels", "4_wheels", "total", "total_count"):
            continue
        # avoid including group columns that are not class names
        preds.append(name)
    # keep order as in DataFrame
    return preds

def _safe_num_series(df: pd.DataFrame, col: str, default_value: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype(float)
    return pd.Series(default_value, index=df.index, dtype=float)

def _sum_columns_if_present(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    parts = []
    for c in cols:
        if c in df.columns:
            parts.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float))
        else:
            parts.append(pd.Series(0.0, index=df.index))
    if not parts:
        return pd.Series(np.nan, index=df.index, dtype=float)
    s = parts[0].copy()
    for p in parts[1:]:
        s = s.add(p, fill_value=0.0)
    return s

# ------------------------
# Generic fetch actual/predicted series for a "type"
# ------------------------
def fetch_actual_predicted_series_generic(
    df: pd.DataFrame,
    t: str,
    class_names: Sequence[str],
    dataset: Optional[str] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    t can be:
      - "total"
      - a class name present in class_names (e.g., 'car', 'van')
      - "2_3_wheels" or "4_wheels"
    The function attempts to find sensible actual_* and predicted_* columns and falls back
    to sum-of-components if aggregated columns don't exist.
    """
    idx = df.index

    # ---------- TOTAL ----------
    if t == "total":
        # actual_total_vehicles OR sum of actual_*_counts
        if "actual_total_vehicles" in df.columns:
            actual_s = _safe_num_series(df, "actual_total_vehicles")
        else:
            actual_cols = [f"actual_{name}_counts" for name in class_names if f"actual_{name}_counts" in df.columns]
            actual_cols = actual_cols or [c for c in df.columns if c.startswith("actual_")]
            actual_s = _sum_columns_if_present(df, actual_cols) if actual_cols else pd.Series(np.nan, index=idx, dtype=float)

        # predicted_total (if exists) else sum predicted per-class
        if "predicted_total" in df.columns:
            pred_s = _safe_num_series(df, "predicted_total")
        else:
            pred_cols = [f"predicted_{name}" for name in class_names if f"predicted_{name}" in df.columns]
            pred_cols = pred_cols or [c for c in df.columns if c.startswith("predicted_")]
            pred_s = _sum_columns_if_present(df, pred_cols) if pred_cols else pd.Series(np.nan, index=idx, dtype=float)

        return actual_s.reindex(idx), pred_s.reindex(idx)

    # ---------- GROUPS ----------
    if t == "2_3_wheels":
        # prefer explicit aggregated columns if present
        actual_s = _safe_num_series(df, "actual_2_3_wheels")
        if actual_s.isna().all():
            # sum bicycle + motorcycle actuals (if present)
            actual_s = _sum_columns_if_present(df, ["actual_bicycle_counts", "actual_motorcycle_counts"])
        pred_s = _safe_num_series(df, "predicted_2_3_wheels")
        if pred_s.isna().all():
            pred_s = _sum_columns_if_present(df, ["predicted_bicycle", "predicted_motorcycle"])
        return actual_s.reindex(idx), pred_s.reindex(idx)

    if t == "4_wheels":
        actual_s = _safe_num_series(df, "actual_4_wheels")
        if actual_s.isna().all():
            # include van if present in actual columns (actual_van_counts), else car + bus + truck
            actual_s = _sum_columns_if_present(df, ["actual_car_counts", "actual_bus_counts", "actual_truck_counts", "actual_van_counts"])
        pred_s = _safe_num_series(df, "predicted_4_wheels")
        if pred_s.isna().all():
            # predicted columns: car,bus,truck,van if present
            pred_s = _sum_columns_if_present(df, ["predicted_car", "predicted_bus", "predicted_truck", "predicted_van"])
        return actual_s.reindex(idx), pred_s.reindex(idx)

    # ---------- PER-CLASS ----------
    # class name expected like 'car' or 'van'
    name = t
    actual_col1 = f"actual_{name}_counts"
    actual_col2 = f"actual_{name}"  # alternative naming
    if actual_col1 in df.columns:
        actual_s = _safe_num_series(df, actual_col1)
    elif actual_col2 in df.columns:
        actual_s = _safe_num_series(df, actual_col2)
    else:
        # missing actual -> zeros or NaN? choose zeros (consistent with prior code)
        actual_s = pd.Series(np.nan, index=idx, dtype=float)

    pred_col = f"predicted_{name}"
    if pred_col in df.columns:
        pred_s = _safe_num_series(df, pred_col)
    else:
        # maybe predictions are stored by class index (unlikely here) or missing -> NaN
        pred_s = pd.Series(np.nan, index=idx, dtype=float)

    return actual_s.reindex(idx), pred_s.reindex(idx)





def sort_metrics(df_metrics: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Return a new DataFrame sorted by the given metric column (ascending).

    Args:
        df_metrics : DataFrame produced by compute_accuracy_metrics
        metric_col : string name of the metric column to sort by
                     (e.g., 'mae_total', 'mape_cars', 'signed_mape_buses', ...)

    Returns:
        A sorted copy of df_metrics.
    """
    if metric_col not in df_metrics.columns:
        raise ValueError(f"Column '{metric_col}' not found in df_metrics.")

    return df_metrics.sort_values(by=metric_col, ascending=True).reset_index(drop=True)


# ------------------------
# Main builder for df_metrics (generic)
# ------------------------
def build_error_df_generic(
    df_counts: pd.DataFrame,
    class_names: Optional[Sequence[str]] = None,
    dataset: Optional[str] = None,
    parse_exp_id_fn: Optional[Callable[[str], Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Build df_metrics from df_counts. Auto-detects classes if class_names is None.
    Returns DataFrame with per-exp rows and error columns for:
      mae_<type>, signed_mae_<type>, mape_<type>, signed_mape_<type>
    where <type> includes: total, each class, 2_3_wheels, 4_wheels
    """
    df = df_counts.copy().reset_index(drop=True)
    meta = {}

    # detect class names
    detected = class_names or detect_predicted_class_names(df, meta)
    if not detected:
        # fallback to canonical COCO-like list (best-effort)
        detected = ["bicycle", "car", "motorcycle", "bus", "truck"]

    # create types list
    types = ["total"] + list(detected) + ["2_3_wheels", "4_wheels"]

    # Ensure required meta columns exist in df_metrics: if missing, add with NaN
    # Build base meta-like columns: try to preserve previous names where possible
    meta_cols = [
        "exp_id", "algorithm_id" , "sequence", "counter_family", "tracker", "vicinity", "stride",
        "direction_relative_to_camera", "road_directionality", "lighting",
        "actual_total_vehicles"
    ]
    # Add common actual per-class columns to ensure selection later
    for c in detected:
        col_actual = f"actual_{c}_counts"
        col_pred = f"predicted_{c}"
    
        if col_actual not in df.columns:
            df[col_actual] = np.nan
    
        if col_pred not in df.columns:
            df[col_pred] = np.nan


    # ensure meta cols present
    for c in meta_cols:
        if c not in df.columns:
            df[c] = np.nan

    # prepare result frame with meta subset
    base_meta_cols = [c for c in meta_cols if c in df.columns]
    df_metrics = df[base_meta_cols].copy()

    # compute metrics for each type
    for t in types:
        actual_s, pred_s = fetch_actual_predicted_series_generic(df, t, class_names=detected, dataset=dataset)
        # align
        actual_s = actual_s.reindex(df.index).astype(float)
        pred_s = pred_s.reindex(df.index).astype(float)
        df_metrics[f"mae_{t}"] = mae_series(actual_s, pred_s).values
        df_metrics[f"signed_mae_{t}"] = signed_mae_series(actual_s, pred_s).values
        df_metrics[f"mape_{t}"] = mape_series(actual_s, pred_s).values
        df_metrics[f"signed_mape_{t}"] = signed_mape_series(actual_s, pred_s).values

    # attach algorithm_id (if not present)
    if "algorithm_id" not in df_metrics.columns:
        df_metrics["algorithm_id"] = df_metrics["exp_id"].astype(str).apply(get_algorithm_id_from_exp)

    return df_metrics

# ------------------------
# Weighted helpers & aggregation (adapted from your original)
# ------------------------
def _infer_actual_col_for_metric(metric_col: str) -> str:
    """
    Map metric column to appropriate actual-count column name used as weight.
    """
    if "_" not in metric_col:
        return "actual_total_vehicles"
    suffix = metric_col.split("_", 1)[1]
    if suffix == "total":
        return "actual_total_vehicles"
    if suffix in ("2_3_wheels", "4_wheels"):
        return f"actual_{suffix}"
    return f"actual_{suffix}_counts"

def _weighted_mean(values: pd.Series, weights: pd.Series, eps: float = 1e-9) -> Optional[float]:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = (~v.isna()) & (~w.isna())
    if not mask.any():
        return float(np.nan)
    v_sel = v[mask].astype(float)
    w_sel = w[mask].astype(float)
    wsum = w_sel.sum()
    if wsum <= eps:
        return float(np.nan)
    return float((v_sel * w_sel).sum() / wsum)

def aggregate_metrics_by_algorithm_generic(
    df_metrics: pd.DataFrame,
    metrics_cols: Optional[List[str]] = None,
    aggregator: str = "mean",
    weight_mode: Optional[str] = None,
    parse_exp_id_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    eps: float = 1e-9
) -> pd.DataFrame:
    """
    Aggregates df_metrics per algorithm_id (same semantics as your original aggregate_metrics_by_algorithm).
    """
    if 'exp_id' not in df_metrics.columns:
        raise ValueError("df_metrics must contain an 'exp_id' column.")

    if aggregator not in ("mean", "weighted"):
        raise ValueError("aggregator must be 'mean' or 'weighted'.")

    if aggregator == "weighted" and weight_mode not in ("actual", "inverse_actual"):
        raise ValueError("For weighted aggregation, weight_mode must be 'actual' or 'inverse_actual'.")

    df = df_metrics.copy()
    df["algorithm_id"] = df["exp_id"].astype(str).map(get_algorithm_id_from_exp)

    if metrics_cols is None:
        prefixes = ("mae_", "mape_", "signed_mae_", "signed_mape_")
        metrics_cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
        metrics_cols = sorted(metrics_cols)
    else:
        missing = [c for c in metrics_cols if c not in df.columns]
        if missing:
            raise ValueError(f"The following metric columns were not found in df_metrics: {missing}")

    rows: List[Dict[str, Any]] = []
    groupby_obj = df.groupby("algorithm_id", sort=True)
    for algorithm_id, group in groupby_obj:
        row: Dict[str, Any] = {"algorithm_id": algorithm_id, "n_videos": int(len(group))}
        # metadata via parse_exp_id_fn if provided
        if parse_exp_id_fn is not None:
            meta = {}
            # derive parse metadata by calling parse_exp_id on a dummy exp_id prefixed by a sequence
            try:
                dummy = "DUMMY__" + algorithm_id
                parsed = parse_exp_id_fn(dummy)
                row.update({
                    "counter_family": parsed.get("counter"),
                    "stride": parsed.get("stride"),
                    "tracker": parsed.get("tracker"),
                    "vicinity": parsed.get("vicinity")
                })
            except Exception:
                row.update({"counter_family": None, "stride": None, "tracker": None, "vicinity": None})
        else:
            row.update({"counter_family": None, "stride": None, "tracker": None, "vicinity": None})

        for metric in metrics_cols:
            if aggregator == "mean":
                vals = pd.to_numeric(group[metric], errors="coerce")
                agg_val = float(vals.mean()) if vals.notna().any() else float(np.nan)
            else:
                weight_col = _infer_actual_col_for_metric(metric)
                if weight_col not in group.columns:
                    if "actual_total_vehicles" in group.columns:
                        w = group["actual_total_vehicles"]
                    else:
                        w = pd.Series(1.0, index=group.index)
                else:
                    w = group[weight_col]
                if weight_mode == "actual":
                    weights = pd.to_numeric(w, errors="coerce").fillna(0.0)
                else:
                    weights = 1.0 / (pd.to_numeric(w, errors="coerce").fillna(0.0) + eps)
                agg_val = _weighted_mean(group[metric], weights, eps=eps)
            row[metric] = agg_val
        rows.append(row)

    if not rows:
        cols = ["algorithm_id", "n_videos", "counter_family", "stride", "tracker", "vicinity"]
        return pd.DataFrame(columns=cols)

    result = pd.DataFrame(rows)
    meta_cols = ["counter_family", "stride", "tracker", "vicinity"]
    ordered_cols = ["algorithm_id", "n_videos"] + [c for c in meta_cols if c in result.columns] + metrics_cols
    result = result[ordered_cols].reset_index(drop=True)
    return result

# ------------------------
# keep_min_vicinity (same semantics)
# ------------------------
def keep_min_vicinity(df: pd.DataFrame,
                      metric_col: str,
                      counter_families: List[str],
                      keep_other: bool = False) -> pd.DataFrame:
    dfc = df.copy()
    dfc[metric_col] = pd.to_numeric(dfc[metric_col], errors="coerce")
    dfc['stride'] = pd.to_numeric(dfc['stride'], errors="coerce")

    selected_parts = []
    for cf in counter_families:
        sub = dfc[dfc['counter_family'] == cf]
        if sub.empty:
            continue
        idx = sub.groupby(['tracker', 'stride'])[metric_col].idxmin().dropna().astype(int)
        selected_parts.append(dfc.loc[idx])

    if selected_parts:
        result = pd.concat(selected_parts, ignore_index=True)
    else:
        result = pd.DataFrame(columns=dfc.columns)

    if keep_other:
        others = dfc[~dfc['counter_family'].isin(counter_families)]
        if not others.empty:
            result = pd.concat([result, others], ignore_index=True)

    return result.reset_index(drop=True)

# ------------------------
# Convenience wrappers
# ------------------------
def compute_overall_simple_mean(df_metrics: pd.DataFrame, parse_exp_id_fn: Optional[Callable[[str], dict]] = None) -> pd.DataFrame:
    return aggregate_metrics_by_algorithm_generic(df_metrics=df_metrics, metrics_cols=None, aggregator="mean", weight_mode=None, parse_exp_id_fn=parse_exp_id_fn)

def compute_overall_weighted_by_actual(df_metrics: pd.DataFrame, parse_exp_id_fn: Optional[Callable[[str], dict]] = None) -> pd.DataFrame:
    return aggregate_metrics_by_algorithm_generic(df_metrics=df_metrics, metrics_cols=None, aggregator="weighted", weight_mode="actual", parse_exp_id_fn=parse_exp_id_fn)

def compute_overall_weighted_by_inverse_actual(df_metrics: pd.DataFrame, parse_exp_id_fn: Optional[Callable[[str], dict]] = None, eps: float = 1e-9) -> pd.DataFrame:
    return aggregate_metrics_by_algorithm_generic(df_metrics=df_metrics, metrics_cols=None, aggregator="weighted", weight_mode="inverse_actual", parse_exp_id_fn=parse_exp_id_fn, eps=eps)







# Helper: strip leading counter prefix from an algorithm_id
def _strip_counter_prefix(algorithm_id: str) -> str:
    """
    Remove a leading 'counter_X__' or 'counter-X__' prefix from algorithm_id.
    If no counter prefix found, returns algorithm_id unchanged.
    Example:
      'counter_0__s1__tracker-none__vic-0.01' -> 's1__tracker-none__vic-0.01'
    """
    if not isinstance(algorithm_id, str):
        return algorithm_id
    # match either counter_0__ or counter-0__ (robust)
    return re.sub(r'^counter[_-]\d+__', '', algorithm_id)

# Helper: produce candidate algorithm_id forms for a given counter index and algorithm_minus_counter_id
def _make_counter_algorithm_candidates(counter_idx: int, algorithm_minus_counter_id: str) -> list:
    """
    Return likely algorithm_id strings for this counter index.
    We try a couple of reasonable variants to be robust.
    """
    return [
        f"counter_{counter_idx}__{algorithm_minus_counter_id}",
        f"counter-{counter_idx}__{algorithm_minus_counter_id}",
        # sometimes 'counter0' (no underscore) — unlikely but included:
        f"counter{counter_idx}__{algorithm_minus_counter_id}"
    ]


def build_counter_err_pivot(
    df_overall: pd.DataFrame,
    metric_col: str,
    num_counters: int = 6,
    parse_exp_id_fn: Optional[Callable[[str], dict]] = None
) -> pd.DataFrame:
    """
    Build a pivoted dataframe where each row is one unique algorithm_minus_counter_id (the triplet
    e.g. 's1__tracker-none__vic-0.01') and columns contain the requested metric for counters 0..num_counters-1.

    Args:
      df_overall: aggregated per-algorithm dataframe (must include an 'algorithm_id' column and the requested metric_col)
      metric_col: string name of the metric column to extract (e.g., 'mae_total', 'mape_total', ...)
      num_counters: number of counters to include (default 6 -> counter_0 .. counter_5)
      parse_exp_id_fn: optional function to parse exp_id-like strings into metadata; if None we use default_parse_exp_id

    Returns:
      DataFrame with columns:
        ['algorithm_minus_counter_id', 'tracker', 'vicinity', 'stride',
         'counter_0_err', 'counter_1_err', ..., 'counter_{num_counters-1}_err']
    """
    # basic validation
    if 'algorithm_id' not in df_overall.columns:
        raise ValueError("df_overall must contain an 'algorithm_id' column.")

    if metric_col not in df_overall.columns:
        raise ValueError(f"metric_col '{metric_col}' not found in df_overall columns.")

    # default parse fn
    if parse_exp_id_fn is None:
        # default_parse_exp_id should exist in the same script (provided earlier)
        try:
            parse_exp_id_fn = default_parse_exp_id  # type: ignore
        except NameError:
            raise ValueError("No parse_exp_id_fn provided and default_parse_exp_id not found in scope.")

    # build lookup: algorithm_id -> metric value
    metric_map = dict(zip(df_overall['algorithm_id'].astype(str), df_overall[metric_col].astype(float)))

    # collect unique algorithm_minus_counter_id values
    alg_ids = [str(x) for x in df_overall['algorithm_id'].astype(str).unique()]
    algo_minus_set = set(_strip_counter_prefix(aid) for aid in alg_ids)

    rows = []
    for algo_minus in sorted(algo_minus_set):
        row = {"algorithm_minus_counter_id": algo_minus}

        # derive metadata by parsing a dummy exp id that includes a counter (we pick counter_0)
        # format: DUMMY__counter_0__<algo_minus>
        dummy = f"DUMMY__counter_0__{algo_minus}"
        try:
            parsed = parse_exp_id_fn(dummy)
            # parsed expected to provide 'tracker','vicinity','stride' keys (see default_parse_exp_id)
            row["tracker"] = parsed.get("tracker")
            row["vicinity"] = parsed.get("vicinity")
            row["stride"] = parsed.get("stride")
        except Exception:
            row["tracker"] = None
            row["vicinity"] = None
            row["stride"] = None

        # for each counter, try to find its aggregated metric in metric_map (try multiple candidate forms)
        for i in range(num_counters):
            colname = f"counter_{i}_err"
            val = np.nan
            candidates = _make_counter_algorithm_candidates(i, algo_minus)
            for cand in candidates:
                if cand in metric_map:
                    val = metric_map[cand]
                    break
            # fallback: sometimes algorithm_id in df_overall might be missing the counter prefix but include other formatting;
            # also try to find by regex matching end-with algorithm_minus (less strict)
            if np.isnan(val):
                # search for any key that endswith the algo_minus and contains the counter index
                pattern = re.compile(rf'counter[_-]{i}__{re.escape(algo_minus)}$', re.IGNORECASE)
                for k in metric_map:
                    if pattern.search(k):
                        val = metric_map[k]
                        break
            row[colname] = float(val) if not (val is None or (isinstance(val, float) and np.isnan(val))) else np.nan

        rows.append(row)

    out_df = pd.DataFrame(rows)

    # ensure desired column ordering
    counter_cols = [f"counter_{i}_err" for i in range(num_counters)]
    cols = ["algorithm_minus_counter_id", "tracker", "vicinity", "stride"] + counter_cols
    cols = [c for c in cols if c in out_df.columns]
    return out_df[cols].sort_values(by="algorithm_minus_counter_id").reset_index(drop=True)
