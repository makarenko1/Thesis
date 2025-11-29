import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator
from opacus import PrivacyEngine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mutual_information import MutualInformation
from proxy_mutual_information_tvd import ProxyMutualInformationTVD
from proxy_repair_maxsat import ProxyRepairMaxSat
from tuple_contribution import TupleContribution
from unused_measures.proxy_mutual_information_privbayes import ProxyMutualInformationPrivbayes


adult_criteria = [["sex", "income>50K", "education-num"], ["sex", "income>50K", "hours-per-week"],
                  ["race", "income>50K", "education-num"], ["race", "income>50K", "hours-per-week"]]

census_criteria = [["HEALTH", "INCTOT", "EDUC"], ["HEALTH", "OCC", "EDUC"], ["HEALTH", "MARST", "AGE"],
                   ["HEALTH", "INCTOT", "AGE"]]

stackoverflow_criteria = [["Country", "RemoteWork", "Employment"], ["Age", "PurchaseInfluence", "OrgSize"],
                          ["Country", "MainBranch", "YearsCodePro"], ["Age", "MainBranch", "EdLevel"]]

compas_criteria = [["race", "is_recid", "age_cat"], ["sex", "is_recid", "priors_count"],
                   ["race", "decile_score", "c_charge_degree"], ["sex", "v_decile_score", "age_cat"]]

healthcare_criteria = [["race", "complications", "age_group"], ["smoker", "complications", "age_group"],
                       ["race", "income", "county"], ["smoker", "income", "num_children"]]

datasets = {
    "Adult": {
        "path": "data/adult.csv",
        "criteria": adult_criteria,
    },
    "IPUMS-CPS": {
        "path": "data/census.csv",
        "criteria": census_criteria,
    },
    "Stackoverflow": {
        "path": "data/stackoverflow.csv",
        "criteria": stackoverflow_criteria,
    },
    "Compas": {
        "path": "data/compas.csv",
        "criteria": compas_criteria,
    },
    "Healthcare": {
        "path": "data/healthcare.csv",
        "criteria": healthcare_criteria,
    },
}

datasets_shortened = {
    "Adult": {
        "path": "data/adult.csv",
        "criteria": adult_criteria,
    },
    "Stackoverflow": {
        "path": "data/stackoverflow.csv",
        "criteria": stackoverflow_criteria,
    },
    "Compas": {
        "path": "data/compas.csv",
        "criteria": compas_criteria,
    }
}


from matplotlib.ticker import FuncFormatter

# Formatter: round to 3 decimals, then strip trailing zeros and dot
def _yfmt(y, pos):
    s = f"{y:.3f}"
    s = s.rstrip('0').rstrip('.')
    return s
y_formatter = FuncFormatter(_yfmt)


def create_plot_1():
    # --- config ---------------------------------------------------
    # Row order: MI, PrivBayes Original, PrivBayes with offset, TVD
    PROXIES = [
        ("Mutual\nInformation", "MI"),
        ("PrivBayes Proxy", "PRIV_ORIG"),
        ("PrivBayes Proxy\nwith offset", "PRIV_OFFSET"),
        ("TVD Proxy", "TVD"),
    ]

    PROXY_COLOR = {
        "MI": "#1f77b4",
        "PRIV_ORIG": "#ff7f0e",
        "PRIV_OFFSET": "#9467bd",
        "TVD": "#2ca02c",
    }

    # Bigger fonts everywhere
    TITLE_FS = 30  # column titles
    ROWLAB_FS = 28  # row (y-axis) labels
    TICK_FS = 28  # tick labels (bottom axis + y-ticks)
    ANNOT_FS = 22  # numbers above bars

    # bar layout
    BAR_SPACING = 0.55  # <--- tighter spacing (was effectively 0.65)
    BAR_WIDTH = 0.5

    # --- compute values
    vals = {k: {} for _, k in PROXIES}
    for ds_name, ds_config in datasets_shortened.items():
        path, attrs = ds_config["path"], ds_config["criteria"]
        mi_scores, priv_scores_orig, priv_scores_offset, tvd_scores = [], [], [], []
        for s_col, o_col, a_col in attrs:
            mi_scores.append(MutualInformation(datapath=path).calculate([[s_col, o_col, a_col]],
                                                                        encode_and_clean=True))
            priv_scores_orig.append(ProxyMutualInformationPrivbayes(datapath=path).calculate(
                s_col, o_col, a_col, add_offset=False
            ))
            priv_scores_offset.append(ProxyMutualInformationPrivbayes(datapath=path).calculate(
                s_col, o_col, a_col
            ))
            tvd_scores.append(ProxyMutualInformationTVD(datapath=path).calculate(
                [[s_col, o_col, a_col]], encode_and_clean=True
            ))
        vals["MI"][ds_name] = mi_scores
        vals["PRIV_ORIG"][ds_name] = priv_scores_orig
        vals["PRIV_OFFSET"][ds_name] = priv_scores_offset
        vals["TVD"][ds_name] = tvd_scores

    # labels per dataset
    ds_labels = {
        ds_name: [str(i) for i in range(1, len(ds_config["criteria"]) + 1)]
        for ds_name, ds_config in datasets_shortened.items()
    }

    # --- figure
    n_rows, n_cols = len(PROXIES), len(datasets_shortened)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 14), constrained_layout=True)

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    def annotate_bar(ax, rect):
        h = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2.0
        va = 'bottom' if h >= 0 else 'top'
        offset = 6 if h >= 0 else -8
        ax.annotate(
            f"{h:.3f}", xy=(x, h), xytext=(0, offset),
            textcoords="offset points", ha="center", va=va, fontsize=ANNOT_FS
        )

    # column titles
    for c, ds_name in enumerate(datasets_shortened):
        axes[0, c].set_title(ds_name, fontsize=TITLE_FS)

    # draw bars
    for r, (proxy_title, proxy_key) in enumerate(PROXIES):
        color = PROXY_COLOR[proxy_key]

        # common y-range per row
        row_vals = []
        for ds_name in datasets_shortened:
            row_vals.extend(vals[proxy_key][ds_name])
        row_vals = np.asarray(row_vals, dtype=float)

        for c, ds_name in enumerate(datasets_shortened):
            ax = axes[r, c]
            y = vals[proxy_key][ds_name]
            labels = ds_labels[ds_name]

            if row_vals.size == 0:
                row_min, row_max = -1.0, 1.0
            else:
                row_min, row_max = float(np.min(row_vals)), float(np.max(row_vals))

            pad = 0.05 * max(1.0, abs(row_max - row_min))  # 5% padding

            if proxy_key in ("MI", "PRIV_OFFSET", "TVD"):
                ymin, ymax = 0.0, row_max + pad
            else:
                ymin, ymax = row_min - pad, row_max + pad

            ax.set_ylim(ymin, ymax)

            # tighter spacing
            x = np.arange(len(y)) * BAR_SPACING
            bars = ax.bar(x, y, color=color, width=BAR_WIDTH)
            for rect in bars:
                annotate_bar(ax, rect)

            ax.set_xticks(x)
            if r == n_rows - 1:
                ax.set_xticklabels(labels, ha="right", fontsize=TICK_FS)
                # x label only for bottom row
                ax.set_xlabel("criterion", fontsize=ROWLAB_FS)
            else:
                ax.set_xticklabels([])

            if c == 0:
                ax.set_ylabel(proxy_title, fontsize=ROWLAB_FS, labelpad=20)

            ax.yaxis.grid(True, linestyle=":", linewidth=0.9, alpha=0.65)
            ax.tick_params(axis='y', labelsize=TICK_FS)
            # <<< format y tick labels here
            ax.yaxis.set_major_formatter(y_formatter)

    plt.savefig("plots/plot1.png", dpi=220)
    plt.show()


def create_plot_2():
    # --- config ---------------------------------------------------
    MEASURES = [
        ("TVD Proxy", "TVD"),
        ("Tuple Contribution", "AUC"),
    ]

    MEASURE_COLOR = {
        "TVD": "#1f77b4",   # blue
        "AUC": "#ff7f0e",   # orange
    }

    TITLE_FS = 32  # column titles
    ROWLAB_FS = 32  # row (y-axis) labels
    TICK_FS = 28  # tick labels (bottom axis + y-ticks)
    ANNOT_FS = 22  # numbers above bars

    BAR_SPACING = 0.55  # <--- tighter spacing (was 0.65)
    BAR_WIDTH = 0.5

    # --- compute values ------------------------------------------
    vals = {k: {} for _, k in MEASURES}
    for ds_name, ds_config in datasets_shortened.items():
        path, attrs = ds_config["path"], ds_config["criteria"]
        tvd_scores, auc_scores = [], []
        for s_col, o_col, a_col in attrs:
            tvd_scores.append(ProxyMutualInformationTVD(datapath=path).calculate(
                [[s_col, o_col, a_col]], encode_and_clean=True
            ))
            auc_scores.append(TupleContribution(datapath=path).calculate(
                [[s_col, o_col, a_col]], encode_and_clean=True
            ))
        vals["TVD"][ds_name] = tvd_scores
        vals["AUC"][ds_name] = auc_scores

    ds_labels = {
        ds_name: [str(i) for i in range(1, len(ds_config["criteria"]) + 1)]
        for ds_name, ds_config in datasets_shortened.items()
    }

    # --- figure ---------------------------------------------------
    n_rows, n_cols = len(MEASURES), len(datasets_shortened)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), constrained_layout=True)

    def annotate_bar(ax, rect):
        h = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2.0
        va = 'bottom' if h >= 0 else 'top'
        offset = 6 if h >= 0 else -8
        ax.annotate(
            f"{h:.3f}", xy=(x, h), xytext=(0, offset),
            textcoords="offset points", ha="center", va=va, fontsize=ANNOT_FS
        )

    # Column titles
    for c, ds_name in enumerate(datasets_shortened):
        axes[0, c].set_title(ds_name, fontsize=TITLE_FS)

    # Draw bars ----------------------------------------------------
    for r, (measure_title, measure_key) in enumerate(MEASURES):
        color = MEASURE_COLOR[measure_key]
        row_vals = []
        for ds_name in datasets_shortened:
            row_vals.extend(vals[measure_key][ds_name])
        row_vals = np.asarray(row_vals, dtype=float)
        row_min, row_max = float(np.min(row_vals)), float(np.max(row_vals))
        pad = 0.05 * max(1.0, abs(row_max - row_min))
        ymin, ymax = 0.0, row_max + pad

        for c, ds_name in enumerate(datasets_shortened):
            ax = axes[r, c]
            y = vals[measure_key][ds_name]
            labels = ds_labels[ds_name]

            ax.set_ylim(ymin, ymax)
            x = np.arange(len(y)) * BAR_SPACING
            bars = ax.bar(x, y, color=color, width=BAR_WIDTH)
            for rect in bars:
                annotate_bar(ax, rect)

            ax.set_xticks(x)
            if r == n_rows - 1:
                ax.set_xticklabels(labels, ha="right", fontsize=TICK_FS)
                # x label only for bottom row
                ax.set_xlabel("criterion", fontsize=ROWLAB_FS)
            else:
                ax.set_xticklabels([])

            if c == 0:
                ax.set_ylabel(measure_title, fontsize=ROWLAB_FS, labelpad=20)

            ax.yaxis.grid(True, linestyle=":", linewidth=0.9, alpha=0.65)
            ax.tick_params(axis='y', labelsize=TICK_FS)
            ax.yaxis.set_major_formatter(y_formatter)  # <<< apply formatter

    plt.savefig("plots/plot2.png", dpi=220)
    plt.show()


######################################### Experiments ##########################################

measures = {
    "Proxy Mutual Information TVD": ProxyMutualInformationTVD,
    "Proxy RepairMaxSat": ProxyRepairMaxSat,
    "Tuple Contribution": TupleContribution,
}

timeout_seconds = 2 * 60 * 60

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import numpy as np

def _encode_and_clean(data_path, cols):
    """
    Read CSV, clean missing values, normalize numeric columns, and label-encode
    categorical columns in `cols`.

    - Replace ["NA", "N/A", ""] with NaN and drop rows with missing values in `cols`.
    - For numeric columns in `cols`, replace negative values with 0.
    - For data/census.csv:
        * Bin AGE into buckets of size 10 (e.g., 1–10 -> 10, 11–20 -> 20, ...).
        * Drop rows with INCTOT > 200000.
        * Discretize INCTOT into 10,000-wide buckets (0–9999 -> 0, 10000–19999 -> 10000, ...).
    - For categorical columns in `cols`, apply LabelEncoder.
    """
    df = pd.read_csv(data_path)
    df = df.replace(["NA", "N/A", ""], pd.NA).dropna(subset=cols).copy()

    # 1) Numeric: replace negative values with 0 (only in selected cols)
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df.loc[df[c] < 0, c] = 0

    # 2) Special handling for IPUMS-CPS census data
    if data_path == "data/census.csv":
        # Bin AGE into buckets of 10: 1–10 -> 10, 11–20 -> 20, ...
        if "AGE" in df.columns:
            age = pd.to_numeric(df["AGE"], errors="coerce")
            # Clamp to at least 1 so that 0 or invalids go into the first bucket
            age = age.fillna(1)
            age = np.clip(age, 1, None)
            # (age-1)//10 gives 0 for 1–10, 1 for 11–20, etc.; then +1 and *10 -> 10, 20, ...
            df["AGE"] = (((age - 1) // 10) + 1) * 10

        # Remove INCTOT > 200000 and discretize into 10k buckets
        if "INCTOT" in df.columns:
            inctot = pd.to_numeric(df["INCTOT"], errors="coerce")
            df = df[inctot <= 200000].copy()
            inctot = pd.to_numeric(df["INCTOT"], errors="coerce").fillna(0)
            # Bucket size 10,000; adjust if you want different granularity
            df["INCTOT"] = (inctot // 10000) * 10000

    # 3) Categorical: label-encode only non-numeric columns in `cols`
    for c in cols:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    return df


def plot_legend(outfile="plots/legend_proxies.png"):
    """Creating a standalone legend figure for Experiment 1 with the four measures arranged in a single horizontal row,
    and save it to `outfile`.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(8, 1.6))
    handles = []
    for i, label in enumerate(measures.keys()):
        from matplotlib.lines import Line2D as MplLine2D
        line = MplLine2D(
            [2, 3], [2, 2],
            color=colors[i % len(colors)],
            marker="o",
            linestyle="-",
            linewidth=2,
            label=label,
        )
        ax.add_line(line)
        handles.append(line)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        handles,
        measures.keys(),
        loc="center",
        ncol=len(measures.keys()),
        frameon=True,
        fontsize=10,
    )

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.show()


def run_experiment_1(
    epsilon=None,
    repetitions=5,
    outfile="plots/experiment1.png"
):
    """Plotting average runtimes over 'repetitions' repetitions per measure and dataset as function of
    the number of tuples."""
    num_tupless_per_dataset = {
        "Adult": [1000, 5000, 10000, 15000, 30000],
        "IPUMS-CPS": [5000, 10000, 50000, 100000, 300000, 600000, 1000000],
        "Stackoverflow": [5000, 10000, 20000, 40000, 60000],
        "Compas": [1000, 1500, 3000, 7000, 10000],
        "Healthcare": [100, 200, 400, 700, 1000],
    }

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []

        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        num_tuples_this_dataset = num_tupless_per_dataset[ds_name]
        results = {
            measure_name: {"mean": [], "min": [], "max": []}
            for measure_name in measures.keys()
        }

        for measure_name, measure_cls in measures.items():
            if measure_name == "Proxy RepairMaxSat" and path == "data/census.csv":  # this dataset timeouted whole
                for _ in num_tuples_this_dataset:
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                continue

            flag_timeout = False
            for num_tuples in num_tuples_this_dataset:
                if flag_timeout:
                    print("Skipping iteration due to timeout.")
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                    continue

                runtimes_rep = []
                for _ in range(repetitions):
                    n = min(num_tuples, len(data))
                    sample = data.sample(n=n, replace=False)
                    m = measure_cls(data=sample)
                    start_time = time.time()
                    with ThreadPoolExecutor() as executor:
                        try:
                            _ = executor.submit(m.calculate, criteria, epsilon=epsilon).result(
                                timeout=timeout_seconds)
                            elapsed_time = time.time() - start_time
                            runtimes_rep.append(elapsed_time)
                        except TimeoutError:
                            print("Skipping the iteration due to timeout.")
                            runtimes_rep.append(np.nan)
                            flag_timeout = True
                            break

                vals = np.array(runtimes_rep, dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    mean_v = min_v = max_v = np.nan
                else:
                    mean_v = vals.mean()
                    min_v = vals.min()
                    max_v = vals.max()
                results[measure_name]["mean"].append(mean_v)
                results[measure_name]["min"].append(min_v)
                results[measure_name]["max"].append(max_v)

        xs = np.arange(len(num_tuples_this_dataset))
        tick_labels = []

        for num_tuples in num_tuples_this_dataset:
            if num_tuples >= 1_000_000:
                tick_labels.append(f"{num_tuples // 1_000_000}M")
            elif num_tuples >= 1_000:
                tick_labels.append(f"{num_tuples // 1_000}K")
            else:
                tick_labels.append(str(num_tuples))

        if ds_name == "IPUMS-CPS":
            if len(num_tuples_this_dataset) >= 7:
                show_idx = [0, 2, 4, 6]
            else:
                show_idx = list(range(len(num_tuples_this_dataset)))
            ax.set_xticks(np.array(show_idx))
            ax.set_xticklabels([tick_labels[i] for i in show_idx])
        else:
            ax.set_xticks(xs)
            ax.set_xticklabels(tick_labels)

        for measure_name, stats in results.items():
            means = np.array(stats["mean"])
            lows  = np.array(stats["min"])
            highs = np.array(stats["max"])
            line, = ax.plot(xs, means, marker="o", linewidth=2, label=measure_name)
            mask = ~np.isnan(means) & ~np.isnan(lows) & ~np.isnan(highs)
            if mask.any():
                ax.fill_between(
                    xs[mask],
                    lows[mask],
                    highs[mask],
                    alpha=0.2,
                    color=line.get_color(),
                    linewidth=0,
                )

        ax.set_xlabel("number of tuples")
        ax.set_yscale('log')
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("runtime (s), log scale")
    fig.suptitle("Runtime as Function of Number of Tuples", y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_2(
    epsilon=None,
    num_tuples=100000,
    repetitions=5,
    outfile="plots/experiment2.png"
):
    """Plot average runtimes over `repetitions` per measure and dataset as function of the number of criteria."""
    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []

        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        results = {
            measure_name: {"mean": [], "min": [], "max": []}
            for measure_name in measures.keys()
        }

        for measure_name, measure_cls in measures.items():
            flag_timeout = False

            for num_criteria in range(1, len(criteria) + 1):
                if measure_name == "Proxy RepairMaxSat" and path == "data/census.csv":  # this dataset timeouted
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                    continue

                if flag_timeout:
                    print("Skipping next iterations because got timeout for smaller number of criteria.")
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                    continue

                runtimes_rep = []
                for _ in range(repetitions):
                    n = min(num_tuples, len(data))
                    sample = data.sample(n=n, replace=False)
                    m = measure_cls(data=sample)
                    start_time = time.time()
                    with ThreadPoolExecutor() as executor:
                        try:
                            _ = executor.submit(
                                m.calculate,
                                criteria[:num_criteria],
                                epsilon=epsilon
                            ).result(timeout=timeout_seconds)
                            elapsed_time = time.time() - start_time
                            runtimes_rep.append(elapsed_time)
                        except TimeoutError:
                            print("Skipping iteration due to timeout.")
                            runtimes_rep.append(np.nan)
                            flag_timeout = True
                            break

                vals = np.array(runtimes_rep, dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    mean_v = min_v = max_v = np.nan
                else:
                    mean_v = vals.mean()
                    min_v = vals.min()
                    max_v = vals.max()

                results[measure_name]["mean"].append(mean_v)
                results[measure_name]["min"].append(min_v)
                results[measure_name]["max"].append(max_v)

        xs = np.arange(1, len(criteria) + 1)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(k)) for k in xs])

        for measure_name, stats in results.items():
            means = np.array(stats["mean"])
            lows  = np.array(stats["min"])
            highs = np.array(stats["max"])
            line, = ax.plot(xs, means, marker="o", linewidth=2, label=measure_name)
            mask = ~np.isnan(means) & ~np.isnan(lows) & ~np.isnan(highs)
            if mask.any():
                ax.fill_between(
                    xs[mask],
                    lows[mask],
                    highs[mask],
                    alpha=0.2,
                    color=line.get_color(),
                    linewidth=0,
                )

        ax.set_xlabel("number of criteria")
        ax.set_yscale('log')
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("runtime (s), log scale")
    fig.suptitle(
        f"Runtime as Function of Number of Criteria, number of tuples at most {round(num_tuples / 1000)}K",
        y=1.02
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_3(
    epsilons=(0.1, 1, 5, 10),
    num_tuples=100000,
    repetitions=5,
    outfile="plots/experiment3.png"
):
    """Relative L1 error as function of epsilon."""

    def _rel_error(x, y, tiny=1e-100):
        denom = max(abs(y), tiny)  # ensure we do not divide by 0
        return abs(x - y) / denom

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        cols_list = list(dict.fromkeys(cols_list))
        data_full = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(data_full))
        results = {
            measure_name: {"mean": [], "min": [], "max": []}
            for measure_name in measures.keys()
        }

        for measure_name, measure_cls in measures.items():
            if measure_name == "Proxy RepairMaxSat" and path == "data/census.csv":
                for _ in epsilons:
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                continue

            errs_per_eps = [[] for _ in epsilons]
            for _ in range(repetitions):
                if n < len(data_full):
                    sample = data_full.sample(n=n, replace=False)
                else:
                    sample = data_full
                m = measure_cls(data=sample)
                with ThreadPoolExecutor() as executor:
                    try:
                        non_private_result = executor.submit(
                            m.calculate, criteria, epsilon=None
                        ).result(timeout=timeout_seconds)
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")
                        continue

                for j, eps in enumerate(epsilons):
                    with ThreadPoolExecutor() as executor:
                        try:
                            private_result = executor.submit(
                                m.calculate, criteria, epsilon=eps
                            ).result(timeout=timeout_seconds)
                            err = _rel_error(private_result, non_private_result)
                            errs_per_eps[j].append(err)
                        except TimeoutError:
                            print("Skipping iteration due to timeout.")
                            continue

            for j in range(len(epsilons)):
                vals = np.array(errs_per_eps[j], dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    mean_v = min_v = max_v = np.nan
                else:
                    mean_v = vals.mean()
                    min_v = vals.min()
                    max_v = vals.max()
                results[measure_name]["mean"].append(mean_v)
                results[measure_name]["min"].append(min_v)
                results[measure_name]["max"].append(max_v)

        x = np.array(epsilons, dtype=float)
        for measure_name, stats in results.items():
            means = np.array(stats["mean"])
            lows  = np.array(stats["min"])
            highs = np.array(stats["max"])
            line, = ax.plot(x, means, marker="o", linewidth=2, label=measure_name)
            mask = ~np.isnan(means) & ~np.isnan(lows) & ~np.isnan(highs)
            if mask.any():
                ax.fill_between(
                    x[mask],
                    lows[mask],
                    highs[mask],
                    alpha=0.2,
                    color=line.get_color(),
                    linewidth=0,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([str(eps) for eps in epsilons])
        ax.set_xlabel("privacy budget ε")
        ax.set_yscale('log')
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("relative L1 error, log scale")
    fig.suptitle(
        f"Relative L1 Error as Function of Privacy Budget, number of tuples at most {round(num_tuples / 1000)}K",
        y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_4(
        epsilon: float = None,
        num_tuples: int = 100000,
        repetitions: int = 5,
        outfile: str = "plots/experiment4.png",
):
    """
    For each dataset: histogram with X = fairness criteria (indexed 1..k), Y = value.
    For each criterion, show two bars: MutualInformation and its proxy
    ProxyMutualInformationTVD, averaged over `repetitions`.
    """

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(30, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        cols_list = list(dict.fromkeys(cols_list))
        df_full = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(df_full))
        mi_sums = {}
        tvd_sums = {}
        mi_counts = {}
        tvd_counts = {}

        for criterion in criteria:
            if len(criterion) == 3:
                protected, response, admissible = criterion
                crit_label = f"{protected} , {response} | {admissible}"
            else:
                protected, response = criterion[0], criterion[1]
                crit_label = f"{protected} , {response}"

            mi_sums[crit_label] = 0.0
            tvd_sums[crit_label] = 0.0
            mi_counts[crit_label] = 0
            tvd_counts[crit_label] = 0

        for _ in range(repetitions):
            if n < len(df_full):
                df = df_full.sample(n=n, replace=False)
            else:
                df = df_full
            mi_measure = MutualInformation(data=df)
            tvd_measure = ProxyMutualInformationTVD(data=df)

            for criterion in criteria:
                if len(criterion) == 3:
                    protected, response, admissible = criterion
                    crit_label = f"{protected} , {response} | {admissible}"
                else:
                    protected, response = criterion[0], criterion[1]
                    crit_label = f"{protected} , {response}"
                with ThreadPoolExecutor() as executor:
                    try:
                        mi_val = executor.submit(
                            mi_measure.calculate, [criterion], epsilon=epsilon
                        ).result(timeout=timeout_seconds)
                        mi_sums[crit_label] += float(mi_val)
                        mi_counts[crit_label] += 1
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")
                with ThreadPoolExecutor() as executor:
                    try:
                        tvd_val = executor.submit(
                            tvd_measure.calculate, [criterion], epsilon=epsilon
                        ).result(timeout=timeout_seconds)
                        tvd_sums[crit_label] += float(tvd_val)
                        tvd_counts[crit_label] += 1
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")

        crit_labels = sorted(mi_sums.keys())
        x = np.arange(len(crit_labels), dtype=float)
        width = 0.35
        mi_vals = []
        tvd_vals = []
        for cl in crit_labels:
            mi_mean = mi_sums[cl] / mi_counts[cl] if mi_counts[cl] > 0 else np.nan
            tvd_mean = tvd_sums[cl] / tvd_counts[cl] if tvd_counts[cl] > 0 else np.nan
            mi_vals.append(mi_mean)
            tvd_vals.append(tvd_mean)
        mi_vals = np.array(mi_vals, dtype=float)
        tvd_vals = np.array(tvd_vals, dtype=float)
        mi_bars = ax.bar(x - width / 2, mi_vals, width, label="MutualInformation")
        ax.bar(x + width / 2, tvd_vals, width, label="ProxyMutualInformationTVD")
        ax.set_xlabel("criterion")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(1, len(crit_labels) + 1)])

        for rect, val in zip(mi_bars, mi_vals):
            if not np.isnan(val):
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=18,
                )

        ax.set_title(ds_name)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        f"Comparison of MutualInformation and ProxyMutualInformationTVD, at most "
        f"{round(num_tuples / 1000)}K tuples",
        y=1.03,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_5(
    num_tuples=100000,
    repetitions=10,
    epsilon=None,
    outfile="plots/experiment5.png",
):
    """TupleContribution value as function of k, sampling separately for each repetition."""

    ks_per_dataset = {
        "Adult": [100, 500, 1000, 5000, 10000, 15000, 30000],
        "IPUMS-CPS": [1000, 5000, 10000, 50000, 100000, 300000, 600000, 1000000],
        "Stackoverflow": [1000, 5000, 10000, 20000, 40000, 60000],
        "Compas": [100, 500, 1000, 1500, 3000, 7000, 10000],
        "Healthcare": [100, 200, 400, 700, 1000],
    }

    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 28,
        "xtick.labelsize": 19,
        "ytick.labelsize": 22,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    tiny = 1e-100  # to avoid issues with log(0)

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        ks_this_dataset = ks_per_dataset[ds_name]

        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        cols_list = list(dict.fromkeys(cols_list))

        data = _encode_and_clean(path, cols_list)
        stats = {"mean": [], "min": [], "max": []}

        flag_timeout = False
        for k in ks_this_dataset:
            if flag_timeout:
                print("Skipping the iteration due to timeout.")
                stats["mean"].append(np.nan)
                stats["min"].append(np.nan)
                stats["max"].append(np.nan)
                continue

            values_rep = []
            for _ in range(repetitions):
                n = min(num_tuples, len(data))
                sample = data.sample(n=n, replace=False)
                m = TupleContribution(data=sample)
                with ThreadPoolExecutor() as executor:
                    try:
                        val = executor.submit(
                            m.calculate,
                            criteria,
                            k=k,
                            epsilon=epsilon,
                        ).result(timeout=timeout_seconds)
                        values_rep.append(float(val))
                    except TimeoutError:
                        print("Skipping the iteration due to timeout.")
                        values_rep.append(np.nan)
                        flag_timeout = True
                        break

            vals = np.array(values_rep, dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                mean_v = min_v = max_v = np.nan
            else:
                mean_v = vals.mean()
                min_v = vals.min()
                max_v = vals.max()
            stats["mean"].append(mean_v)
            stats["min"].append(min_v)
            stats["max"].append(max_v)

        xs = np.arange(len(ks_this_dataset))
        means = np.array(stats["mean"], dtype=float)
        lows  = np.array(stats["min"], dtype=float)
        highs = np.array(stats["max"], dtype=float)
        means = np.clip(means, tiny, None)
        lows  = np.clip(lows, tiny, None)
        highs = np.clip(highs, tiny, None)
        line, = ax.plot(xs, means, marker="o", linewidth=2,
                        label="TupleContribution value")
        mask = ~np.isnan(means) & ~np.isnan(lows) & ~np.isnan(highs)
        if mask.any():
            ax.fill_between(
                xs[mask],
                lows[mask],
                highs[mask],
                alpha=0.2,
                color=line.get_color(),
                linewidth=0,
            )
        tick_labels = []
        for k in ks_this_dataset:
            if k >= 1_000_000:
                tick_labels.append(f"{k // 1_000_000}M")
            elif k >= 1_000:
                tick_labels.append(f"{k // 1_000}K")
            else:
                tick_labels.append(str(k))
        if ds_name == "IPUMS-CPS":
            show_idx = [0, 2, 4, 6] if len(ks_this_dataset) >= 7 else list(range(len(ks_this_dataset)))
            ax.set_xticks(np.array(show_idx))
            ax.set_xticklabels([tick_labels[i] for i in show_idx])
        else:
            ax.set_xticks(xs)
            ax.set_xticklabels(tick_labels)
        ax.set_xlabel("k (top-k tuples)")
        ax.set_yscale('log')
        if ds_name == "Healthcare":
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("TupleContribution (log scale)")
    fig.suptitle(
        f"TupleContribution value as function of k, at most {round(num_tuples / 1000)}K tuples",
        y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_6(
    num_tuples=100000,
    repetitions=10,
    epsilon=1.0,
    outfile="plots/experiment6.png",
):
    """Relative L1 error of TupleContribution as function of k."""

    ks_per_dataset = {
        "Adult": [500, 1000, 5000, 10000, 15000, 30000],
        "IPUMS-CPS": [1000, 5000, 10000, 50000, 100000, 300000, 600000, 1000000],
        "Stackoverflow": [1000, 5000, 10000, 20000, 40000, 60000],
        "Compas": [500, 1000, 1500, 3000, 7000, 10000],
        "Healthcare": [100, 200, 400, 700, 1000],
    }

    def _rel_error(x, y, tiny=1e-100):
        denom = max(abs(y), tiny)  # ensure we do not divide by 0
        return abs(x - y) / denom

    plt.rcParams.update({
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        ks = ks_per_dataset[ds_name]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        cols_list = list(dict.fromkeys(cols_list))
        data = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(data))
        stats = {"mean": [], "min": [], "max": []}

        for k in ks:
            errs = []

            for _ in range(repetitions):
                sample = data.sample(n=n, replace=False)
                m = TupleContribution(data=sample)
                with ThreadPoolExecutor() as executor:
                    try:
                        non_private_result = executor.submit(
                            m.calculate,
                            criteria,
                            k=k,
                            epsilon=None
                        ).result(timeout=timeout_seconds)
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")
                        errs.append(np.nan)
                with ThreadPoolExecutor() as executor:
                    try:
                        private_result = executor.submit(
                            m.calculate,
                            criteria,
                            k=k,
                            epsilon=epsilon
                        ).result(timeout=timeout_seconds)
                        errs.append(_rel_error(private_result, non_private_result))
                    except TimeoutError:
                        print("Skipping iteration due to timeout.")
                        errs.append(np.nan)
                        break

            vals = np.array(errs, dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                mean_v = min_v = max_v = np.nan
            else:
                mean_v = vals.mean()
                min_v = vals.min()
                max_v = vals.max()
            stats["mean"].append(mean_v)
            stats["min"].append(min_v)
            stats["max"].append(max_v)

        x = np.arange(len(ks))
        means = np.array(stats["mean"])
        lows  = np.array(stats["min"])
        highs = np.array(stats["max"])
        line, = ax.plot(x, means, marker="o", linewidth=2,
                        label="TupleContribution L1 error")
        mask = ~np.isnan(means) & ~np.isnan(lows) & ~np.isnan(highs)
        if mask.any():
            ax.fill_between(
                x[mask],
                lows[mask],
                highs[mask],
                alpha=0.2,
                color=line.get_color(),
                linewidth=0,
            )
        full_tick_labels = [str(k) if k % 1000 != 0 else f"{k // 1000}K" for k in ks]
        if ds_name == "IPUMS-CPS":
            show_idx = [0, 2, 4, 6] if len(ks) >= 7 else list(range(len(ks)))
            ax.set_xticks(np.array(show_idx))
            ax.set_xticklabels([full_tick_labels[i] for i in show_idx])
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(full_tick_labels)
        ax.set_xlabel("k (top-k tuples)")
        ax.set_yscale('log')
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("relative L1 error (log scale)")
    fig.suptitle(
        f"Relative L1 Error of TupleContribution as Function of k, at most {round(num_tuples / 1000)}K tuples, "
        f"ε = {epsilon}",
        y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_7(
        epsilon: Optional[float] = None,
        num_tuples: int = 5000,
        repetitions: int = 1,
        outfile: str = "plots/experiment7.png",
):
    """Values of measures for IPUMS-CPS (for criterions with more unfairness we expect higher values)."""

    all_rows = []
    path = "data/census.csv"
    criteria = [["HEALTH", "INCTOT"], ["INCTOT", "AGE"], ["SEX", "AGE"]]

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    for criterion in criteria:
        data = _encode_and_clean(path, criterion)
        n = min(num_tuples, len(data))
        sum_tvd = 0.0
        sum_repair = 0.0
        sum_tc = 0.0

        for rep in range(repetitions):
            sample = data.sample(n=n, replace=False)
            tvd_proxy = ProxyMutualInformationTVD(data=sample)
            sum_tvd += float(tvd_proxy.calculate([criterion], epsilon=epsilon))
            repair_proxy = ProxyRepairMaxSat(data=sample)
            sum_repair += float(repair_proxy.calculate([criterion], epsilon=epsilon))
            tc_proxy = TupleContribution(data=sample)
            sum_tc += float(tc_proxy.calculate([criterion], epsilon=epsilon))

        tvd_avg = sum_tvd / repetitions
        repair_avg = sum_repair / repetitions
        tc_avg = sum_tc / repetitions
        all_rows.append([
            round(tvd_avg, 4),
            repair_avg,
            round(tc_avg, 4),
        ])

    num_criteria = len(criteria)
    x = np.arange(num_criteria)
    criterion_numbers = [str(i) for i in range(1, num_criteria + 1)]
    measure_labels = ["Proxy\nMutualInformationTVD", "Proxy\nRepairMaxSAT", "TupleContribution"]

    # one distinct color per subplot
    subplot_colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(max(8, num_criteria * 1.5), 3.5),
        sharex=True
    )
    all_rows_np = np.array(all_rows, dtype=float)

    for ax, mlabel, col_idx, color in zip(axes, measure_labels, [0, 1, 2], subplot_colors):
        vals = all_rows_np[:, col_idx]
        ax.bar(x, vals, color=color)
        ax.set_yscale('log')
        ax.set_title(mlabel)
        ax.set_xticks(x)
        ax.set_xticklabels(criterion_numbers)

    for ax in axes:
        ax.set_xlabel("criterion")

    plt.tight_layout(rect=[0, 0, 1, 0.9])

    dir_name = os.path.dirname(outfile)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # create_plot_1()
    # create_plot_2()
    # plot_legend()
    # run_experiment_1()
    # run_experiment_2()
    # run_experiment_3()
    # run_experiment_4()
    # run_experiment_5()
    # run_experiment_6()
    run_experiment_7()
    # run_experiment_7_make_less_unfair()
    # run_experiment_8_unconditional()
    # run_experiment_8_conditional()

