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

    import os
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.show()


def run_experiment_1(
    epsilon=None,
    repetitions=10,
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

    rng = np.random.RandomState()
    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        num_tuples_this_dataset = num_tupless_per_dataset[ds_name]

        # store mean / min / max per measure
        results = {
            measure_name: {"mean": [], "min": [], "max": []}
            for measure_name in measures.keys()
        }

        for measure_name, measure_cls in measures.items():
            if measure_name == "Proxy RepairMaxSat" and path == "data/census.csv":  # this dataset timeouted whole
                # no data -> all NaNs
                for _ in num_tuples_this_dataset:
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                continue

            flag_timeout = False
            for num_tuples in num_tuples_this_dataset:
                if flag_timeout:
                    print("Skipping next iterations because got timeout for smaller number of tuples.")
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                    continue

                runtimes_rep = []
                for _ in range(repetitions):
                    n = min(num_tuples, len(data))
                    sample = data.sample(n=n, replace=False, random_state=rng.randint(0, 1_000_000))
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

        # numeric x for plotting + nice tick labels
        xs = np.arange(len(num_tuples_this_dataset))
        tick_labels = []
        for num_tuples in num_tuples_this_dataset:
            if num_tuples >= 1_000_000:
                # 1,000,000 -> "1M"
                tick_labels.append(f"{num_tuples // 1_000_000}M")
            elif num_tuples >= 1_000:
                # 5,000 -> "5K"
                tick_labels.append(f"{num_tuples // 1_000}K")
            else:
                # < 1,000 -> exact number
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

    import os
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_2(
    epsilon=None,
    num_tuples=100000,
    repetitions=10,
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

    rng = np.random.RandomState()
    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)

        # store mean / min / max per measure
        results = {
            measure_name: {"mean": [], "min": [], "max": []}
            for measure_name in measures.keys()
        }

        for measure_name, measure_cls in measures.items():
            flag_timeout = False
            for num_criteria in range(1, len(criteria) + 1):
                # cases we skip entirely
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
                    sample = data.sample(n=n, replace=False,
                                         random_state=rng.randint(0, 1_000_000))
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

        # numeric x for plotting + nice tick labels
        xs = np.arange(1, len(criteria) + 1)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(k)) for k in xs])

        for measure_name, stats in results.items():
            means = np.array(stats["mean"])
            lows  = np.array(stats["min"])
            highs = np.array(stats["max"])

            # main line
            line, = ax.plot(xs, means, marker="o", linewidth=2, label=measure_name)

            # shadow band between min and max
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

    import os
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_3(
    epsilons=(0.1, 1, 5, 10),
    num_tuples=100000,
    repetitions=10,
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

    rng = np.random.RandomState()
    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(data))
        sample = data.sample(n=n, replace=False,
                             random_state=rng.randint(0, 1_000_000))

        # store mean / min / max per measure
        results = {
            measure_name: {"mean": [], "min": [], "max": []}
            for measure_name in measures.keys()
        }

        for measure_name, measure_cls in measures.items():
            m = measure_cls(data=sample)

            # skip RepairMaxSat on huge datasets (as before)
            if measure_name == "Proxy RepairMaxSat" and path == "data/census.csv":
                for _ in epsilons:
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                continue

            # non-private baseline
            with ThreadPoolExecutor() as executor:
                try:
                    non_private_result = executor.submit(
                        m.calculate, criteria, epsilon=None
                    ).result(timeout=timeout_seconds)
                except TimeoutError:
                    print("Skipping the measure due to baseline timeout.")
                    for _ in epsilons:
                        results[measure_name]["mean"].append(np.nan)
                        results[measure_name]["min"].append(np.nan)
                        results[measure_name]["max"].append(np.nan)
                    continue

            # DP runs for each epsilon
            for eps in epsilons:
                errs = []
                for _ in range(repetitions):
                    with ThreadPoolExecutor() as executor:
                        try:
                            private_result = executor.submit(
                                m.calculate, criteria, epsilon=eps
                            ).result(timeout=timeout_seconds)
                            errs.append(_rel_error(private_result, non_private_result))
                        except TimeoutError:
                            print("Skipping this epsilon due to timeout.")
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

                results[measure_name]["mean"].append(mean_v)
                results[measure_name]["min"].append(min_v)
                results[measure_name]["max"].append(max_v)

        # plotting with shadow bands
        x = np.array(epsilons, dtype=float)
        for measure_name, stats in results.items():
            means = np.array(stats["mean"])
            lows  = np.array(stats["min"])
            highs = np.array(stats["max"])

            # line for the mean
            line, = ax.plot(x, means, marker="o", linewidth=2, label=measure_name)

            # shadow band between min and max
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

    import os
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

    # Loop over datasets (assumes `datasets` and `_encode_and_clean` are defined globally)
    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]

        # Collect all needed columns once for this dataset
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        cols_list = list(dict.fromkeys(cols_list))  # dedupe

        # Preprocess data and subsample
        df = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(df))
        if n < len(df):
            df = df.sample(n=n, replace=False, random_state=0)

        # Per-criterion accumulators: sums over repetitions
        mi_sums = {}
        tvd_sums = {}
        mi_counts = {}
        tvd_counts = {}

        for criterion in criteria:
            # Human-readable label (used only for internal ordering now)
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

        # Repeat experiment to average over randomness / DP noise
        for _ in range(repetitions):
            mi_measure = MutualInformation(data=df)
            tvd_measure = ProxyMutualInformationTVD(data=df)

            for criterion in criteria:
                if len(criterion) == 3:
                    protected, response, admissible = criterion
                    crit_label = f"{protected} , {response} | {admissible}"
                else:
                    protected, response = criterion[0], criterion[1]
                    crit_label = f"{protected} , {response}"

                # MutualInformation
                with ThreadPoolExecutor() as executor:
                    try:
                        mi_val = executor.submit(
                            mi_measure.calculate, [criterion], epsilon=epsilon
                        ).result(timeout=timeout_seconds)
                        mi_sums[crit_label] += float(mi_val)
                        mi_counts[crit_label] += 1
                    except TimeoutError:
                        print("Skipping MutualInformation iteration due to timeout.")

                # ProxyMutualInformationTVD
                with ThreadPoolExecutor() as executor:
                    try:
                        tvd_val = executor.submit(
                            tvd_measure.calculate, [criterion], epsilon=epsilon
                        ).result(timeout=timeout_seconds)
                        tvd_sums[crit_label] += float(tvd_val)
                        tvd_counts[crit_label] += 1
                    except TimeoutError:
                        print("Skipping TVD iteration due to timeout.")

        # Build arrays for plotting (mean over repetitions)
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

        # Plot grouped bars
        mi_bars = ax.bar(x - width / 2, mi_vals, width, label="MutualInformation")
        ax.bar(x + width / 2, tvd_vals, width, label="ProxyMutualInformationTVD")

        # X-ticks as 1,2,3,... instead of criterion strings
        ax.set_xlabel("criterion")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(1, len(crit_labels) + 1)])

        # Write values on top of MutualInformation bars only
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

    import os
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

    rng = np.random.RandomState()
    EPS = 1e-12  # to avoid issues with log(0)

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]

        ks_this_dataset = ks_per_dataset[ds_name]

        # all columns needed for this dataset
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
                sample = data.sample(
                    n=n,
                    replace=False,
                    random_state=rng.randint(0, 1_000_000),
                )
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

        # clip to avoid log(0)
        means = np.clip(means, EPS, None)
        lows  = np.clip(lows,  EPS, None)
        highs = np.clip(highs, EPS, None)

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

    import os
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
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 34,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    rng = np.random.RandomState()

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]

        # --- use dataset-specific ks ---
        ks = ks_per_dataset[ds_name]

        # Collect all columns used by any criterion for this dataset
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        cols_list = list(dict.fromkeys(cols_list))  # deduplicate while preserving order

        # Encode and clean once per dataset
        data = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(data))
        sample = data.sample(n=n, replace=False,
                             random_state=rng.randint(0, 1_000_000))

        # We store mean/min/max L1 errors for each k
        stats = {"mean": [], "min": [], "max": []}

        # One TupleContribution instance on the sampled data
        m = TupleContribution(data=sample)

        for k in ks:
            # --- Non-private baseline for this k ---
            with ThreadPoolExecutor() as executor:
                try:
                    non_private_result = executor.submit(
                        m.calculate,
                        criteria,
                        k=k,
                        epsilon=None
                    ).result(timeout=timeout_seconds)
                except TimeoutError:
                    print("Skipping the iteration due to timeout.")
                    stats["mean"].append(np.nan)
                    stats["min"].append(np.nan)
                    stats["max"].append(np.nan)
                    continue

            # --- DP runs for this k ---
            errs = []
            for _ in range(repetitions):
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
                        print("Skipping the iteration due to timeout.")
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

        # ---- Plotting for this dataset ----
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

    import os
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_7(
    epsilon: Optional[float] = None,
    num_tuples: int = 60000,
    repetitions_model: int = 20,
    repetitions_measures: int = 0,
    outfile: str = "plots/experiment7.png",
):
    """
    Mixed real-data experiment.

    We use one criterion per dataset:
      - IPUMS-CPS    : census_criteria[3]
      - Stackoverflow: stackoverflow_criteria[3]
      - Compas       : compas_criteria[0]
      - Healthcare   : healthcare_criteria[1]

    For each (dataset, criterion), we gradually INCREASE dependence between
    protected and response (within each admissible stratum) by editing a growing
    fraction of rows.

    The model is trained on *all* columns (all features except the response),
    while the proxy measures operate only on the projection to
    (protected, response, admissible).
    """

    plt.rcParams.update(
        {
            "axes.titlesize": 21,
            "axes.labelsize": 21,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "figure.titlesize": 26,
        }
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 20
    LR = 1e-3
    EPS = 1e-8

    # -------------------- model -------------------- #
    class FairMLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),  # logits
            )

        def forward(self, x):
            return self.net(x)  # raw logits

    # -------------------- encoding / cleaning -------------------- #
    def _encode_and_clean(data_path: str) -> pd.DataFrame:
        """
        Read CSV, clean missing values and make all columns numeric.

        Steps:
        - Read CSV and map ["NA", "N/A", ""] to NaN.
        - (For census only) convert AGE/INCTOT to numeric and apply special
          AGE binning and INCTOT clipping/bucketing.
        - For *all numeric* columns:
            * convert to numeric,
            * replace negative values with 0,
            * fill NaNs with the column mean (or 0 if all NaN).
        - For *all non-numeric* columns:
            * replace NaN with the string "MISSING",
            * label-encode to integers.
        """
        df = pd.read_csv(data_path)
        df = df.replace(["NA", "N/A", ""], pd.NA)

        # --- special handling for IPUMS-CPS census data (before imputation) ---
        if data_path == "data/census.csv":
            # Convert AGE to numeric first
            if "AGE" in df.columns:
                df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")

            # Convert INCTOT to numeric, clip > 200000, keep NaN for now
            if "INCTOT" in df.columns:
                inct = pd.to_numeric(df["INCTOT"], errors="coerce")
                # Keep rows with INCTOT <= 200000; NaN <= 200000 is False, so keep NaNs
                mask_inct = (inct <= 200000) | inct.isna()
                df = df[mask_inct].copy()
                df["INCTOT"] = pd.to_numeric(df["INCTOT"], errors="coerce")

        # --- numeric columns: clip negatives, fill NaNs with mean ---
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                col = pd.to_numeric(df[c], errors="coerce")
                # replace negative values with 0
                col = col.mask(col < 0, 0)
                mean_val = col.mean(skipna=True)
                if np.isnan(mean_val):
                    mean_val = 0.0
                col = col.fillna(mean_val)
                df[c] = col

        # --- now apply census-specific binning after numeric cleanup ---
        if data_path == "data/census.csv":
            if "AGE" in df.columns:
                age = df["AGE"]
                age = np.clip(age, 1, None)
                df["AGE"] = (((age - 1) // 10) + 1) * 10  # 1–10 -> 10, 11–20 -> 20, ...

            if "INCTOT" in df.columns:
                inctot = df["INCTOT"]
                df["INCTOT"] = (inctot // 10000) * 10000  # 10k buckets

        # --- non-numeric columns: fill with "MISSING" and label-encode ---
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                col = df[c].astype("string")
                col = col.fillna("MISSING")
                le = LabelEncoder()
                df[c] = le.fit_transform(col.astype(str))

        return df

    # -------------------- CSP -------------------- #
    def _conditional_statistical_parity(y_hat_prob, protected, admissible):
        """
        For each admissible value a:
            gap(a) = max_s,s' E[ŷ | S=s, A=a] - E[ŷ | S=s', A=a]
        Return sum_a P(A=a) * gap(a).
        """
        A_vals, A_counts = np.unique(admissible, return_counts=True)
        n = len(admissible)
        weighted_gaps = []
        for a, c in zip(A_vals, A_counts):
            mask_a = (admissible == a)
            if mask_a.sum() == 0:
                continue
            rates = []
            for s in np.unique(protected[mask_a]):
                mask_sa = mask_a & (protected == s)
                rates.append(y_hat_prob[mask_sa].mean() if mask_sa.sum() > 0 else 0.0)
            if not rates:
                continue
            gap = float(np.max(rates) - np.min(rates))
            weighted_gaps.append((c / n) * gap)
        return float(np.sum(weighted_gaps)) if weighted_gaps else 0.0

    # -------------------- unfairness injection -------------------- #
    def make_unfair(
        df: pd.DataFrame,
        protected: str,
        response: str,
        admissible: str,
        frac: float,
        rng,
        mode: str = "max",  # "max" -> group with largest mean Y, "min" -> smallest
    ) -> pd.DataFrame:
        """
        Modify a fraction `frac` of rows within each admissible stratum A=a.

        For each a:
          - compute group means E[Y | S=s, A=a],
          - pick privileged group (max or min mean),
          - among ~ frac * |{i : A_i = a}| random rows:
                privileged rows -> label = local y_max
                non-privileged  -> label = local y_min
        """
        df = df.copy()
        if frac <= 0.0 or len(df) == 0:
            return df

        for a in df[admissible].unique():
            mask_a = (df[admissible] == a)
            idx_a = df.index[mask_a]
            n_a = len(idx_a)
            if n_a == 0:
                continue

            num_rows = int(frac * n_a)
            if num_rows == 0:
                continue

            df_a = df.loc[idx_a]

            group_means = df_a.groupby(protected)[response].mean()
            if mode == "max":
                priv_val = group_means.idxmax()
            else:
                priv_val = group_means.idxmin()

            y_min = df_a[response].min()
            y_max = df_a[response].max()

            chosen_idx = rng.choice(idx_a, size=num_rows, replace=False)
            is_priv = df.loc[chosen_idx, protected] == priv_val

            df.loc[chosen_idx[is_priv], response] = y_max
            df.loc[chosen_idx[~is_priv], response] = y_min

        return df

    # -------------------- helpers -------------------- #
    def _to_torch(X: np.ndarray, y: np.ndarray) -> tuple[TensorDataset, int]:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        return TensorDataset(X_t, y_t), X_t.shape[1]

    def _dependency_score(df_proj: pd.DataFrame, criterion) -> float:
        """Use ProxyMutualInformationTVD (non-private) as a direction score."""
        m = ProxyMutualInformationTVD(data=df_proj)
        return float(m.calculate([criterion], epsilon=None))

    # -------------------- experiment config -------------------- #
    chosen_specs = [
        ("IPUMS-CPS", datasets["IPUMS-CPS"]["path"], census_criteria[3]),
        # ("Stackoverflow", datasets["Stackoverflow"]["path"], stackoverflow_criteria[3]),
        ("Compas", datasets["Compas"]["path"], compas_criteria[0]),
        ("Healthcare", datasets["Healthcare"]["path"], healthcare_criteria[1]),
    ]

    fractions = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    rng_global = np.random.RandomState(42)

    measure_classes = {
        "ProxyMutualInformationTVD": ProxyMutualInformationTVD,
        "ProxyRepairMaxSat": ProxyRepairMaxSat,
        "TupleContribution": TupleContribution,
    }

    fig, axes = plt.subplots(1, len(chosen_specs), figsize=(24, 5), sharey=False)

    # -------------------- per-dataset loop -------------------- #
    for ax, (ds_name, path, criterion) in zip(axes, chosen_specs):
        protected, response, admissible = criterion

        # Prepare a base encoded dataset with *all* columns
        df_all = _encode_and_clean(path)
        df_all = df_all.sample(
            n=min(num_tuples, len(df_all)),
            replace=False,
            random_state=rng_global.randint(0, 1_000_000),
        )

        print(f"\nDataset: {ds_name}, criterion: {criterion}")

        # ---- choose direction (max vs min) so that dependency at frac=1 is larger ----
        base_proj = df_all[[protected, response, admissible]].copy()
        dep0 = _dependency_score(base_proj, criterion)

        df_dep_max = make_unfair(
            df_all,
            protected=protected,
            response=response,
            admissible=admissible,
            frac=1.0,
            rng=np.random.RandomState(123),
            mode="max",
        )
        dep_max = _dependency_score(df_dep_max[[protected, response, admissible]], criterion)

        df_dep_min = make_unfair(
            df_all,
            protected=protected,
            response=response,
            admissible=admissible,
            frac=1.0,
            rng=np.random.RandomState(456),
            mode="min",
        )
        dep_min = _dependency_score(df_dep_min[[protected, response, admissible]], criterion)

        chosen_mode = "max" if dep_max >= dep_min else "min"

        print(
            f"  dep(0)={dep0:.4g}, dep(1,max)={dep_max:.4g}, "
            f"dep(1,min)={dep_min:.4g} -> using mode='{chosen_mode}'"
        )

        # ---- main results container for this dataset ----
        results = {
            key: {"mean": []}
            for key in list(measure_classes.keys()) + ["CSP"]
        }

        # -------------------- loop over unfairness fractions -------------------- #
        for frac in fractions:
            measure_vals = {key: [] for key in results.keys()}

            # --- proxy measures --- #
            for _ in range(repetitions_measures):
                df_dep = make_unfair(
                    df_all,
                    protected=protected,
                    response=response,
                    admissible=admissible,
                    frac=frac,
                    rng=np.random.RandomState(rng_global.randint(0, 1_000_000)),
                    mode=chosen_mode,
                )
                df_proj = df_dep[[protected, response, admissible]].copy()

                for name, cls in measure_classes.items():
                    m = cls(data=df_proj)
                    val = m.calculate([criterion], epsilon=epsilon)
                    measure_vals[name].append(float(val))

            # --- model CSP (model sees ALL columns) --- #
            for _ in range(repetitions_model):
                df_dep = make_unfair(
                    df_all,
                    protected=protected,
                    response=response,
                    admissible=admissible,
                    frac=frac,
                    rng=np.random.RandomState(rng_global.randint(0, 1_000_000)),
                    mode=chosen_mode,
                )

                # All columns except the response as features
                X = df_dep.drop(columns=[response]).to_numpy(dtype=float)
                y = df_dep[response].to_numpy(dtype=float)

                # scale y to [0,1] for safety
                y = (y - y.min()) / (y.max() - y.min() + 1e-100)

                prot = df_dep[protected].to_numpy()
                adm = df_dep[admissible].to_numpy()

                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    prot_train,
                    prot_test,
                    adm_train,
                    adm_test,
                ) = train_test_split(
                    X,
                    y,
                    prot,
                    adm,
                    test_size=0.3,
                    random_state=12345,
                )

                ds_train, in_dim = _to_torch(X_train, y_train)
                train_loader = DataLoader(
                    ds_train,
                    batch_size=min(num_tuples, len(ds_train)),
                    shuffle=True,
                )

                model = FairMLP(in_dim).to(DEVICE)
                optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0)
                loss_fn = nn.BCEWithLogitsLoss()

                model.train()
                for _epoch in range(EPOCHS):
                    for xb, yb in train_loader:
                        xb = xb.to(DEVICE)
                        yb = yb.to(DEVICE)
                        optimizer.zero_grad()
                        logits = model(xb).squeeze(1)
                        loss = loss_fn(logits, yb.squeeze(1))
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    logits_test = model(
                        torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
                    ).squeeze(1)
                    probs_test = torch.sigmoid(logits_test).cpu().numpy()

                csp = _conditional_statistical_parity(probs_test, prot_test, adm_test)
                measure_vals["CSP"].append(csp)

            # aggregate over repetitions
            for key in results.keys():
                arr = np.array(measure_vals[key], dtype=float)
                arr = arr[~np.isnan(arr)]
                results[key]["mean"].append(arr.mean() if arr.size else np.nan)

        # -------------------- plotting for this dataset -------------------- #
        for key, vals in results.items():
            means = np.array(vals["mean"], dtype=float)
            means = np.clip(means, EPS, None)
            ax.plot(fractions, means, marker="o", linewidth=2, label=key)

        ax.set_xlabel("fraction of rows edited")
        ax.set_title(ds_name)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("measure / unfairness (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.suptitle(
        "Fairness Measures and Model Unfairness vs. Increasing Unfairness (mixed datasets)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    import os
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_7_make_less_unfair(
    epsilon: Optional[float] = None,
    num_tuples: int = 60000,
    repetitions: int = 5,
    outfile: str = "plots/experiment7_make_less_unfair.png",
):
    """
    For the Census dataset and each fairness criterion, this experiment tests
    how the fairness measures and model unfairness respond to controlled degradation
    of dependency between the protected and response attributes.
    """
    path = "data/census.csv"

    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.titlesize": 24,
    })

    # ==== helper: fairness (CSP) ====
    def _conditional_statistical_parity(y_hat, protected, admissible, threshold=0.5):
        y_pos = (y_hat >= threshold).astype(int)
        A_vals, A_counts = np.unique(admissible, return_counts=True)
        n = len(admissible)
        weighted_gaps = []
        for a, c in zip(A_vals, A_counts):
            mask_a = (admissible == a)
            if mask_a.sum() == 0:
                continue
            rates = []
            for s in np.unique(protected[mask_a]):
                mask_sa = mask_a & (protected == s)
                rates.append(y_pos[mask_sa].mean() if mask_sa.sum() > 0 else 0.0)
            if len(rates) == 0:
                continue
            gap = float(np.max(rates) - np.min(rates))
            weighted_gaps.append((c / n) * gap)
        return float(np.sum(weighted_gaps)) if weighted_gaps else 0.0

    # ==== helper: permutation ====
    def permute_column(df, col, fraction, rng):
        df = df.copy()
        n = len(df)
        num_swap = int(fraction * n)
        if num_swap <= 0:
            return df

        idx_pos = rng.choice(n, size=num_swap, replace=False)
        permuted_vals = df[col].iloc[idx_pos].sample(frac=1, random_state=rng).values
        df.iloc[idx_pos, df.columns.get_loc(col)] = permuted_vals
        return df

    # ==== helper: torch ====
    def _to_torch(X: np.ndarray, y: np.ndarray) -> tuple[TensorDataset, int]:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        return TensorDataset(X_t, y_t), X_t.shape[1]

    # ==== setup ====
    # more permutation levels: 0.1, 0.2, ..., 1.0
    fractions = np.linspace(0, 1, 11)[1:]
    rng = np.random.RandomState(42)

    # measures to include (MutualInformation removed)
    measure_classes = {
        "ProxyMutualInformationTVD": ProxyMutualInformationTVD,
        "ProxyRepairMaxSat": ProxyRepairMaxSat,
        "TupleContribution": TupleContribution,
    }

    fig, axes = plt.subplots(
        1, len(census_criteria), figsize=(22, 5), sharey=False
    )

    # ==== per criterion ====
    for ax, criterion in zip(axes, census_criteria):
        protected, response, admissible = criterion
        cols = [protected, response, admissible]
        df_full = _encode_and_clean(path, cols)
        df_full = df_full.sample(
            n=min(num_tuples, len(df_full)), random_state=rng
        )

        print(f"\nRunning on criterion: {criterion}")

        results = {
            key: {"mean": []}
            for key in list(measure_classes.keys()) + ["CSP", "L1"]
        }

        # vary permutation intensity
        for frac in fractions:
            measure_vals = {key: [] for key in results.keys()}
            for _ in range(repetitions):
                df_perm = permute_column(df_full, response, frac, rng)

                # --- compute measures ---
                for name, cls in measure_classes.items():
                    m = cls(data=df_perm)
                    val = m.calculate([criterion], epsilon=epsilon)
                    measure_vals[name].append(float(val))

                # --- train model ---
                X = df_perm[[protected, admissible]].to_numpy(dtype=float)
                y = df_perm[response].to_numpy(dtype=float)
                # scale y to [0,1]
                y = (y - y.min()) / (y.max() - y.min() + 1e-100)

                prot = df_perm[protected].to_numpy()
                adm = df_perm[admissible].to_numpy()

                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    prot_train,
                    prot_test,
                    adm_train,
                    adm_test,
                ) = train_test_split(
                    X,
                    y,
                    prot,
                    adm,
                    test_size=0.3,
                    random_state=rng.randint(0, 1_000_000),
                )

                ds_train, in_dim = _to_torch(X_train, y_train)
                loader = DataLoader(
                    ds_train,
                    batch_size=min(30000, len(ds_train)),
                    shuffle=True,
                )
                lin = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
                optimizer = torch.optim.SGD(lin.parameters(), lr=1e-2)
                loss_fn = nn.L1Loss()
                lin.train()
                for _epoch in range(10):  # fewer epochs to keep fast
                    for xb, yb in loader:
                        optimizer.zero_grad()
                        preds = lin(xb)
                        loss = loss_fn(preds, yb)  # both [batch,1]
                        loss.backward()
                        optimizer.step()

                # --- evaluate ---
                lin.eval()
                with torch.no_grad():
                    preds = lin(
                        torch.tensor(X_test, dtype=torch.float32)
                    ).numpy().reshape(-1)

                mae = float(np.mean(np.abs(preds - y_test)))
                csp = _conditional_statistical_parity(
                    preds, prot_test, adm_test
                )
                measure_vals["L1"].append(mae)
                measure_vals["CSP"].append(csp)

            # aggregate stats
            for key in results.keys():
                arr = np.array(measure_vals[key], dtype=float)
                arr = arr[~np.isnan(arr)]
                results[key]["mean"].append(arr.mean() if arr.size else np.nan)

        # ==== plotting (no shadows) ====
        for key, vals in results.items():
            means = np.array(vals["mean"])
            ax.plot(
                fractions, means, marker="o", linewidth=2, label=key
            )

        ax.set_xlabel("Permutation fraction of response")
        ax.set_title(f"{protected} , {response} | {admissible}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_yscale("log")

    axes[0].set_ylabel("Score / Unfairness (log scale)")
    fig.suptitle(
        "Sensitivity of Fairness Measures Scores and Model Unfairness "
        "to Increasing Label Permutation"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    import os
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_8_unconditional(
        epsilon: Optional[float] = None,
        num_tuples: int = 100000,   # kept for signature, not used
        repetitions: int = 1,
        outfile: str = "plots/experiment8_unconditional.xlsx",
):

    POS_THRESHOLD = 0.5
    TEST_SIZE = 0.15

    # ==== Fairness: Demographic Parity (unconditional) ====
    def _demographic_parity(y_hat: np.ndarray,
                            protected: pd.Series,
                            threshold: float = POS_THRESHOLD) -> float:
        """
        Demographic parity gap:
            max_{s,s'} | P(ŷ=1 | S=s) - P(ŷ=1 | S=s') |.
        """
        y_pos = (y_hat >= threshold).astype(int)
        rates = []
        for s in np.unique(protected):
            mask_s = (protected == s)
            if mask_s.sum() == 0:
                continue
            rates.append(y_pos[mask_s].mean())
        if len(rates) == 0:
            return 0.0
        return float(np.max(rates) - np.min(rates))

    # ===== Global rows for table =====
    all_rows = []
    all_index = []

    # === Adult dataset only ===
    ds_name = "Adult"
    path = datasets[ds_name]["path"]
    criteria = datasets[ds_name]["criteria"]

    # Read raw once and encode all columns (no manual permutation)
    df_raw = pd.read_csv(path)
    df = _encode_and_clean(path, df_raw.columns.values)

    for criterion in criteria:
        protected = criterion[0]
        response = criterion[1]

        # label uses full criterion so we don't accidentally merge 3 into 2
        crit_label = f"{protected} , {response}"
        crit_uncond = [protected, response]  # unconditional measures still use only these two

        # === Measures (computed once per criterion) ===
        df_for_measures = df[crit_uncond]

        tvd = float(
            ProxyMutualInformationTVD(data=df_for_measures).calculate([crit_uncond], epsilon=epsilon)
        )

        repair = float(
            ProxyRepairMaxSat(data=df_for_measures).calculate([crit_uncond], epsilon=epsilon)
        )
        print(repair)

        # TupleContribution typically uses full df
        tc = float(
            TupleContribution(data=df).calculate([crit_uncond], epsilon=epsilon)
        )

        # === Random Forest model with repetitions (inner loop) ===
        mae_sum = 0.0
        dp_sum = 0.0

        for rep in range(repetitions):
            # Features: all encoded columns except response, preserve order from df
            X_full = df.drop(columns=[response])
            y_full = df[response].to_numpy(dtype=float)
            prot_full = df[protected].to_numpy()

            # optional stratification
            stratify = None
            try:
                y_round = np.round(y_full)
                if len(np.unique(y_round)) <= 10:
                    stratify = y_round
            except Exception:
                pass

            X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
                X_full, y_full, prot_full,
                test_size=TEST_SIZE,
                stratify=stratify,
            )
            prot_test = pd.Series(prot_test).reset_index(drop=True)

            # MinMax scaling on all features (no column permutation)
            numeric_features = X_train.columns
            scaler = MinMaxScaler()
            X_train.loc[:, numeric_features] = scaler.fit_transform(X_train[numeric_features])
            X_test.loc[:, numeric_features] = scaler.transform(X_test[numeric_features])

            # RandomForest
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            yhat_proba = model.predict_proba(X_test)[:, 1]
            yhat_label = (yhat_proba >= POS_THRESHOLD).astype(int)

            mae_rf = float(np.mean(np.abs(yhat_label - y_test)))
            dp_rf = _demographic_parity(yhat_proba, prot_test, threshold=POS_THRESHOLD)

            mae_sum += mae_rf
            dp_sum += dp_rf

        mae_avg = mae_sum / repetitions
        dp_avg = dp_sum / repetitions

        all_rows.append([
            round(tvd, 4),
            round(repair, 4),
            round(tc, 4),
            round(mae_avg, 4),
            round(dp_avg, 4),
        ])
        all_index.append((ds_name, crit_label))

    # ===== Build one big table with row MultiIndex (Dataset, Criterion) =====
    row_index = pd.MultiIndex.from_tuples(all_index, names=["Dataset", "Criterion"])
    col_index = pd.MultiIndex.from_tuples([
        ("ProxyMutualInformationTVD", ""),
        ("RepairMaxSat", ""),
        ("TupleContribution", ""),
        ("RandomForest", "L1 Error"),
        ("RandomForest", "Unfairness (DP)"),
    ])

    table_all = pd.DataFrame(all_rows, index=row_index, columns=col_index)

    # ---------- Histogram plot (unconditional) ----------
    fairness_col = ("RandomForest", "Unfairness (DP)")
    l1_col = ("RandomForest", "L1 Error")

    fairness_vals = table_all[fairness_col].to_numpy(dtype=float)
    order = np.argsort(fairness_vals)  # least -> most unfair

    crit_labels = [f"{ds} | {crit}" for ds, crit in table_all.index]
    crit_labels = [crit_labels[i] for i in order]

    x = np.arange(len(table_all))
    width = 0.16

    measure_cols = [
        ("ProxyMutualInformationTVD", ""),
        ("RepairMaxSat", ""),
        ("TupleContribution", ""),
        l1_col,
        fairness_col,
    ]
    labels = ["TVD", "RepairMaxSat", "TupleContribution", "RF L1", "RF DP"]
    offsets = np.linspace(-2 * width, 2 * width, len(measure_cols))

    fig, ax = plt.subplots(figsize=(max(8, len(table_all) * 0.3), 4))

    for offset, label, col in zip(offsets, labels, measure_cols):
        vals = table_all[col].to_numpy(dtype=float)[order]
        ax.bar(x + offset, vals, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(crit_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Value")
    ax.set_title("Unconditional (Adult): Measures, RF L1 and RF DP per Criterion", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()

    import os
    png_outfile = os.path.splitext(outfile)[0] + ".png"
    os.makedirs(os.path.dirname(png_outfile), exist_ok=True)
    plt.savefig(png_outfile, dpi=256, bbox_inches="tight")
    plt.show()

    # print without truncation
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", None,
    ):
        print("\n=== Unconditional experiment (Adult, averaged, rounded to 4 decimals) ===")
        print(table_all)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    table_all.to_excel(outfile, merge_cells=True)


def run_experiment_8_conditional(
        epsilon: Optional[float] = None,
        num_tuples: int = 100000,
        num_tuples_repair: int = 1000,
        repetitions: int = 5,
        outfile="plots/experiment8_conditional.xlsx"
):
    import os
    import matplotlib.pyplot as plt

    # ==== DP-SGD / training params ====
    DP_NOISE_MULT = 1.0        # Gaussian noise multiplier
    DP_MAX_GRAD_NORM = 1.0     # Per-sample clipping norm
    BATCH_SIZE = 300000        # we'll batch smaller if sample is smaller
    EPOCHS = 5
    LR = 1e-2
    POS_THRESHOLD = 0.5        # for CSP thresholding of predictions
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TEST_SIZE = 0.3

    # ==== Models ====
    class DPLinear(nn.Module):
        """Regression model (DP-linear with sigmoid)."""
        def __init__(self, in_dim: int):
            super().__init__()
            self.linear = nn.Linear(in_dim, 1)
        def forward(self, x):
            return torch.sigmoid(self.linear(x))  # keep outputs in [0,1]

    class DPMLP(nn.Module):
        """Random Forest proxy model (DP-MLP with sigmoid)."""
        def __init__(self, in_dim: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        def forward(self, x):
            return torch.sigmoid(self.net(x))     # keep outputs in [0,1]

    # ==== Fairness: Conditional Statistical Parity (CSP) ====
    def _conditional_statistical_parity(y_hat: np.ndarray,
                                        protected: pd.Series,
                                        admissible: pd.Series,
                                        threshold: float = POS_THRESHOLD) -> float:
        """
        For each admissible value a:
           max_{s,s'} | P(ŷ=1 | S=s, A=a) - P(ŷ=1 | S=s', A=a) |
        Average over a, weighted by P(A=a).
        """
        y_pos = (y_hat >= threshold).astype(int)
        A_vals, A_counts = np.unique(admissible, return_counts=True)
        n = len(admissible)
        weighted_gaps = []
        for a, c in zip(A_vals, A_counts):
            mask_a = (admissible == a)
            if mask_a.sum() == 0:
                continue
            rates = []
            for s in np.unique(protected[mask_a]):
                mask_sa = mask_a & (protected == s)
                rates.append(y_pos[mask_sa].mean() if mask_sa.sum() > 0 else 0.0)
            if len(rates) == 0:
                continue
            gap = float(np.max(rates) - np.min(rates))
            weighted_gaps.append((c / n) * gap)
        return float(np.sum(weighted_gaps)) if weighted_gaps else 0.0

    # ==== Torch helpers ====
    def _to_torch(X: np.ndarray, y: np.ndarray) -> tuple[TensorDataset, int]:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        return TensorDataset(X_t, y_t), X_t.shape[1]

    def _train_dp_sgd(model: nn.Module,
                      train_loader: DataLoader,
                      epochs: int = EPOCHS,
                      lr: float = LR,
                      noise_multiplier: float = DP_NOISE_MULT,
                      max_grad_norm: float = DP_MAX_GRAD_NORM) -> None:
        model.to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
        loss_fn = nn.L1Loss()
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

    def _predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            yhat = model(X_t).cpu().numpy().reshape(-1)
        return np.clip(yhat, 0.0, 1.0)

    # ===== Global rows for ALL datasets =====
    all_rows = []
    all_index = []   # list of (dataset_name, criterion_label) tuples

    # === Core loop over datasets and criteria ===
    for ds_name, spec in datasets.items():
        print(f"\n=== DATASET: {ds_name} ===")
        path = spec["path"]
        criteria = spec["criteria"]

        # accumulators: criterion -> sums
        sum_linear    = defaultdict(lambda: np.array([0.0, 0.0], dtype=float))   # (MAE, CSP)
        sum_mlp       = defaultdict(lambda: np.array([0.0, 0.0], dtype=float))   # (MAE, CSP)
        sum_mi        = defaultdict(float)                                       # MutualInformation
        sum_tvd       = defaultdict(float)                                       # ProxyMutualInformationTVD
        sum_repair    = defaultdict(float)                                       # ProxyRepairMaxSat
        sum_tc        = defaultdict(float)                                       # TupleContribution

        for rep in range(repetitions):
            for criterion in criteria:
                protected, response, admissible = criterion[0], criterion[1], criterion[2]
                crit_label = f"{protected} , {response} | {admissible}".lower()

                # encode and clean only needed columns
                df = _encode_and_clean(path, criterion)
                df_repair = df.sample(n=min(num_tuples_repair, len(df)))

                # limit number of tuples
                n_total = len(df)
                n_use = int(min(n_total, num_tuples))
                if n_use < n_total:
                    df = df.sample(n=n_use)

                # ======= MEASURES =======
                mutual_information = MutualInformation(data=df)
                sum_mi[crit_label] += float(mutual_information.calculate([criterion], epsilon=epsilon))

                tvd_proxy = ProxyMutualInformationTVD(data=df)
                sum_tvd[crit_label] += float(tvd_proxy.calculate([criterion], epsilon=epsilon))

                if path == "data/census.csv":
                    repair_proxy = ProxyRepairMaxSat(data=df_repair)
                else:
                    repair_proxy = ProxyRepairMaxSat(data=df)
                sum_repair[crit_label] += float(repair_proxy.calculate([criterion], epsilon=epsilon))

                tc_proxy = TupleContribution(data=df)
                sum_tc[crit_label] += float(tc_proxy.calculate([criterion], epsilon=epsilon))

                # ======= MODELS (Regression / Random Forest) =======
                feat_cols = [protected] + ([admissible] if admissible else [])
                X_full = df[feat_cols].to_numpy(dtype=float)

                # ORIGINAL-SCALE target
                y_real_full = df[response].to_numpy(dtype=float)

                # scale y to [0, 1] for training
                y_min = y_real_full.min()
                y_max = y_real_full.max()
                if y_max > y_min:
                    y_full = (y_real_full - y_min) / (y_max - y_min)
                else:
                    y_full = np.zeros_like(y_real_full)

                # split (keep both scaled and real targets)
                stratify = None
                try:
                    y_round = np.round(y_full)
                    if len(np.unique(y_round)) <= 10:
                        stratify = y_round
                except Exception:
                    stratify = None

                prot_full = df[protected].to_numpy()
                if admissible:
                    adm_full = df[admissible].to_numpy()
                else:
                    adm_full = np.zeros(len(df), dtype=int)
                (X_train, X_test, y_train, y_test,
                 y_real_train, y_real_test,
                 prot_train, prot_test,
                 adm_train, adm_test) = train_test_split(
                    X_full, y_full, y_real_full, prot_full, adm_full,
                    test_size=TEST_SIZE,
                    stratify=stratify
                )
                # to torch
                train_ds, in_dim = _to_torch(X_train, y_train)
                effective_batch = min(BATCH_SIZE, len(train_ds))
                train_loader = DataLoader(train_ds, batch_size=effective_batch,
                                          shuffle=True, drop_last=False)

                # --- Regression (DPLinear) ---
                lin = DPLinear(in_dim)
                _train_dp_sgd(lin, train_loader)
                yhat_lin_scaled = _predict(lin, X_test)
                mae_lin = float(np.mean(np.abs(yhat_lin_scaled - y_test)))  # in [0,1]
                csp_lin = _conditional_statistical_parity(
                    yhat_lin_scaled,
                    prot_test,
                    adm_test,
                    threshold=POS_THRESHOLD
                )
                sum_linear[crit_label] += np.array([mae_lin, csp_lin], dtype=float)

                # --- Random Forest (DPMLP proxy) ---
                mlp = DPMLP(in_dim)
                _train_dp_sgd(mlp, train_loader)
                yhat_mlp = _predict(mlp, X_test)
                mae_mlp = float(np.mean(np.abs(yhat_mlp - y_real_test)))  # L1 vs REAL values
                csp_mlp = _conditional_statistical_parity(
                    yhat_mlp,
                    prot_test,
                    adm_test,
                    threshold=POS_THRESHOLD
                )
                sum_mlp[crit_label] += np.array([mae_mlp, csp_mlp], dtype=float)

        # ===== Averaging over repetitions; append to global table =====
        crits_sorted = sorted(sum_linear.keys())
        for crit in crits_sorted:
            tvd_avg = sum_tvd[crit] / repetitions
            repair_avg = sum_repair[crit] / repetitions
            tc_avg = sum_tc[crit] / repetitions

            mae_lin_avg = sum_linear[crit][0] / repetitions
            csp_lin_avg = sum_linear[crit][1] / repetitions
            mae_mlp_avg = sum_mlp[crit][0] / repetitions
            csp_mlp_avg = sum_mlp[crit][1] / repetitions

            all_rows.append([
                round(tvd_avg, 4),
                round(repair_avg, 4),
                round(tc_avg, 4),
                round(mae_lin_avg, 4),
                round(csp_lin_avg, 4),
                round(mae_mlp_avg, 4),
                round(csp_mlp_avg, 4),
            ])
            all_index.append((ds_name, crit))

    # ===== Build one big table with row MultiIndex (Dataset, Criterion) =====
    row_index = pd.MultiIndex.from_tuples(all_index, names=["Dataset", "Criterion"])
    col_index = pd.MultiIndex.from_tuples([
        ("ProxyMutualInformationTVD", ""),
        ("RepairMaxSat", ""),
        ("TupleContribution", ""),
        ("Regression", "L1 Error"),
        ("Regression", "Unfairness (CSP)"),
        ("Random Forest", "L1 Error"),
        ("Random Forest", "Unfairness (CSP)"),
    ])

    table_all = pd.DataFrame(all_rows, index=row_index, columns=col_index)

    # ---------- Histogram plot (conditional) ----------
    # Sort criteria by Regression Unfairness (CSP)
    fairness_col = ("Regression", "Unfairness (CSP)")
    fairness_vals = table_all[fairness_col].to_numpy(dtype=float)

    # order = np.argsort(fairness_vals)[::-1]  # descending (most unfair left)
    order = np.argsort(fairness_vals)         # ascending

    crit_labels = [f"{ds} | {crit}" for ds, crit in table_all.index]
    crit_labels = [crit_labels[i] for i in order]

    x = np.arange(len(table_all))
    width = 0.18

    measure_cols = [
        ("ProxyMutualInformationTVD", ""),
        ("RepairMaxSat", ""),
        ("TupleContribution", ""),
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(table_all) * 0.25), 4))

    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    labels = ["TVD", "RepairMaxSat", "TupleContribution", "Regression (CSP)"]

    for i, col in enumerate(measure_cols + [fairness_col]):
        vals = table_all[col].to_numpy(dtype=float)
        vals = vals[order]   # sort bars consistently by unfairness
        ax.bar(x + offsets[i], vals, width, label=labels[i])

    ax.set_xticks(x)
    ax.set_xticklabels(crit_labels, rotation=45, ha="right", fontsize=7)

    ax.set_ylabel("Unfairness / Measure value (log scale)", fontsize=9)
    ax.set_title("Conditional: Measures and Regression CSP per Criterion", fontsize=10)

    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=8)

    plt.tight_layout()

    png_outfile = os.path.splitext(outfile)[0] + ".png"
    os.makedirs(os.path.dirname(png_outfile), exist_ok=True)
    plt.savefig(png_outfile, dpi=256, bbox_inches="tight")
    plt.show()
    # ----------------------------------------

    # print without truncation
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", None,
    ):
        print("\n=== Combined table for all datasets (averaged, rounded to 4 decimals) ===")
        print(table_all)

    # save to Excel
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    table_all.to_excel(outfile, merge_cells=True)


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
    # run_experiment_7()
    # run_experiment_7_make_less_unfair()
    run_experiment_8_unconditional()
    # run_experiment_8_conditional()

