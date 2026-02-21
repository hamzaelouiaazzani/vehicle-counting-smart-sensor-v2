import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable, Union
from matplotlib.ticker import MaxNLocator

def plot_metric_by_tracker_stride(
    df: pd.DataFrame,
    metric: str = "mape_4_wheels",
    aggfunc: Union[str, Callable] = "mean",
    figsize: Tuple[int,int] = (10,6),
    return_fig: bool = False,
    ax: Optional[plt.Axes] = None,
    show_lines: bool = True,
    metric_str: Optional[str] = None,
    counter: Optional[str] = None,
    detector: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Grouped bar plot of `metric` for each (tracker, stride) pair.
    - Metric values are multiplied by 100 (to display as percent).
    - metric_str: optional label string shown on y-axis and used in the title.
    - counter, detector: optional strings included in the title when provided.
    - save_path: optional path to save the figure at 300 dpi (PNG/PDF etc).
    """

    # validate required columns
    required = {"stride", "tracker", metric}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # select relevant columns and drop rows with missing essentials
    df_keep = df[["stride", "tracker", metric]].copy()
    df_keep = df_keep.dropna(subset=["stride", "tracker", metric])

    if df_keep.empty:
        raise ValueError("After selecting required columns and dropping NaNs, dataframe is empty.")

    # aggregate metric by (stride, tracker)
    grouped = df_keep.groupby(["stride", "tracker"])[metric].agg(aggfunc).reset_index()

    # pivot to have strides as rows and trackers as columns
    pivot = grouped.pivot(index="stride", columns="tracker", values=metric)

    # attempt to sort strides numerically if possible, otherwise lexicographically
    try:
        idx_numeric = pd.to_numeric(pivot.index, errors="coerce")
        if not idx_numeric.isna().all():
            pivot.index = idx_numeric
            pivot = pivot.sort_index()
        else:
            pivot = pivot.sort_index()
    except Exception:
        pivot = pivot.sort_index()

    # convert metric to percent for plotting (do not modify original df)
    pivot = pivot * 100.0

    trackers = list(pivot.columns)
    n_trackers = len(trackers)
    x = np.arange(len(pivot.index))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # grouped bar positions
    total_width = 0.8
    if n_trackers == 0:
        raise ValueError("No trackers found after pivoting.")
    width = total_width / n_trackers
    offsets = (np.arange(n_trackers) - (n_trackers - 1) / 2.0) * width

    # draw bars and collect bar containers for color reuse
    bar_containers = []
    for i, tracker in enumerate(trackers):
        values = pivot[tracker].values
        bars = ax.bar(x + offsets[i], values, width=width, label=str(tracker))
        bar_containers.append(bars)

    # optionally draw dashed lines connecting centers for each tracker (skip through NaNs)
    if show_lines:
        for i, tracker in enumerate(trackers):
            centers = x + offsets[i]
            values = pivot[tracker].values
            values_masked = np.ma.masked_invalid(values)
            # try to get color from the first bar patch; fallback to None
            color = None
            if len(bar_containers[i]) > 0:
                try:
                    color = bar_containers[i][0].get_facecolor()
                except Exception:
                    color = None
            # use label='_nolegend_' to avoid creating extra legend entries
            ax.plot(centers, values_masked, linestyle='--', marker='o', linewidth=1, markersize=4,
                    color=color, alpha=0.9, label='_nolegend_')

    # labels, ticks, legend, grid
    ylabel = metric_str if metric_str is not None else f"{metric} (%)"
    ax.set_xlabel("stride")
    ax.set_ylabel(ylabel)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in pivot.index], rotation=0)

    # title logic: if neither detector nor counter given, use default title
    title_metric = metric_str if metric_str is not None else f"{metric} (%)"
    if detector is None and counter is None:
        title = f"{title_metric} by stride and tracker"
    else:
        detector_text = detector if detector is not None else "None"
        counter_text = counter if counter is not None else "None"
        title = f"{title_metric} for config: ({detector_text},{counter_text}) by tracker and stride"

    ax.set_title(title)
    ax.legend(title="tracker", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # optionally save figure at publication quality (300 dpi)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if return_fig:
        return fig, ax
    else:
        plt.show()
        return None


def plot_metric_for_tracker_vicinity(
    df: pd.DataFrame,
    tracker,
    metric: str = "mape_4_wheels",
    aggfunc: Union[str, Callable] = "mean",
    figsize: Tuple[int,int] = (10,6),
    return_fig: bool = False,
    ax: Optional[plt.Axes] = None,
    show_lines: bool = True,
    metric_str: Optional[str] = None,
    counter: Optional[str] = None,
    detector: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    For a fixed `tracker`, plot `metric` grouped by (vicinity, stride).
    - Metric values are multiplied by 100 (to display as percent).
    - metric_str: optional label string shown on y-axis and used in the title.
    - counter, detector: optional strings included in the title when provided.
    - save_path: optional path to save the figure at 300 dpi (PNG/PDF etc).
    """

    # required columns check
    required = {"stride", "vicinity", "tracker", metric}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # select only needed columns and drop NaNs EXCEPT tracker
    df_filtered = df[["stride", "vicinity", "tracker", metric]].copy()
    # DO NOT dropna on "tracker" because tracker may be None/NaN intentionally
    df_filtered = df_filtered.dropna(subset=["stride", "vicinity", metric])

    # filter by tracker robustly (handles None and np.nan)
    if pd.isna(tracker):
        df_filtered = df_filtered[df_filtered["tracker"].isna()]
    else:
        df_filtered = df_filtered[df_filtered["tracker"] == tracker]

    if df_filtered.empty:
        # helpful debug info for the user
        uniq = df["tracker"].unique()
        # convert unique values to readable strings
        uniq_readable = [("None" if pd.isna(x) else str(x)) for x in uniq]
        raise ValueError(
            f"No data left after filtering for tracker={tracker!r}.\n"
            f"Available tracker values (sample): {uniq_readable[:20]}"
        )

    # aggregate metric by (vicinity, stride)
    grouped = df_filtered.groupby(["vicinity", "stride"], as_index=False)[metric].agg(aggfunc)

    # pivot so vicinities are rows (index) and strides are columns
    pivot = grouped.pivot(index="vicinity", columns="stride", values=metric)

    # sort vicinities numerically if possible, otherwise lexicographically
    try:
        vic_numeric = pd.to_numeric(pivot.index, errors="coerce")
        if not vic_numeric.isna().all():
            pivot.index = vic_numeric
            pivot = pivot.sort_index()
        else:
            pivot = pivot.sort_index()
    except Exception:
        pivot = pivot.sort_index()

    # sort strides (columns) numerically if possible for consistent legend order
    strides = list(pivot.columns)
    try:
        strides_numeric = pd.to_numeric(strides, errors="coerce")
        if not np.all(np.isnan(strides_numeric)):
            stride_order = [s for _, s in sorted(zip(strides_numeric, strides),
                                                  key=lambda x: (np.nan_to_num(x[0], nan=np.inf), x[1]))]
            pivot = pivot[stride_order]
            strides = stride_order
        else:
            strides = sorted(strides, key=lambda x: str(x))
            pivot = pivot[strides]
    except Exception:
        strides = sorted(strides, key=lambda x: str(x))
        pivot = pivot[strides]

    # convert metric to percent for plotting
    pivot = pivot * 100.0

    n_strides = len(strides)
    x = np.arange(len(pivot.index))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if n_strides == 0:
        raise ValueError("No unique strides available for plotting after pivoting.")

    # compute bar width and offsets for grouped bars
    total_width = 0.8
    width = total_width / n_strides
    offsets = (np.arange(n_strides) - (n_strides - 1) / 2.0) * width

    # draw bars for each stride and store containers (for color reuse)
    bar_containers = []
    for i, stride in enumerate(strides):
        values = pivot[stride].values
        bars = ax.bar(x + offsets[i], values, width=width, label=str(stride))
        bar_containers.append(bars)

    # optionally draw dashed lines connecting centers for each stride (skip missing values)
    if show_lines:
        for i, stride in enumerate(strides):
            centers = x + offsets[i]
            values = pivot[stride].values
            values_masked = np.ma.masked_invalid(values)
            color = None
            if len(bar_containers[i]) > 0:
                try:
                    color = bar_containers[i][0].get_facecolor()
                except Exception:
                    color = None
            ax.plot(centers, values_masked, linestyle='--', marker='o', linewidth=1, markersize=4,
                    color=color, alpha=0.9, label='_nolegend_')

    # labels, ticks, legend and grid
    ylabel = metric_str if metric_str is not None else f"{metric} (%)"
    ax.set_xlabel("vicinity")
    ax.set_ylabel(ylabel)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in pivot.index], rotation=45, ha='right')

    # build title using new config format if either detector/counter present
    tracker_label = "None" if pd.isna(tracker) else str(tracker)
    title_metric = metric_str if metric_str is not None else f"{metric} (%)"
    if detector is None and counter is None:
        title = f"{title_metric} for tracker '{tracker_label}' by vicinity and stride"
    else:
        detector_text = detector if detector is not None else "None"
        counter_text = counter if counter is not None else "None"
        title = f"{title_metric} for config: ({detector_text},{tracker_label},{counter_text}) by vicinity and stride"

    ax.set_title(title)

    ax.legend(title="stride", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # optionally save figure at publication quality (300 dpi)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if return_fig:
        return fig, ax
    else:
        plt.show()
        return None