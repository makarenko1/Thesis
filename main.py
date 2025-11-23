import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sympy import Line2D
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mutual_information import MutualInformation
from proxy_mutual_information_tvd import ProxyMutualInformationTVD
from proxy_repair_maxsat import ProxyRepairMaxSat
from tuple_contribution import TupleContribution
from unused_measures.proxy_mutual_information_privbayes import ProxyMutualInformationPrivbayes


def create_plot_1():
    # --- config ---------------------------------------------------
    DATASETS = [
        {"name": "Adult", "path": "data/adult.csv", "attrs": [
            ("sex", "income>50K", "education"),
            ("race", "income>50K", "education"),
            ("education", "education-num", "sex"),
        ]},
        {"name": "StackOverflow", "path": "data/stackoverflow.csv", "attrs": [
            ("Country", "EdLevel", "Age"),
            ("Country", "DevType", "Age"),
            ("Country", "SurveyLength", "Age"),
            ("Country", "SOVisitFreq", "Age"),
        ]},
        {"name": "COMPAS", "path": "data/compas.csv", "attrs": [
            ("race", "c_charge_desc", "age"),
            ("race", "score_text", "age"),
            ("race", "sex", "age"),
        ]},
    ]

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
    TITLE_FS = 26  # column titles
    ROWLAB_FS = 26  # row (y-axis) labels
    TICK_FS = 24  # tick labels (bottom axis + y-ticks)
    ANNOT_FS = 18  # numbers above bars

    # --- compute values
    vals = {k: {} for _, k in PROXIES}
    for ds in DATASETS:
        ds_name, path, attrs = ds["name"], ds["path"], ds["attrs"]
        mi_scores, priv_scores_orig, priv_scores_offset, tvd_scores = [], [], [], []
        for s_col, o_col, a_col in attrs:
            mi_scores.append(MutualInformation(datapath=path).calculate([s_col, o_col, a_col]))
            priv_scores_orig.append(ProxyMutualInformationPrivbayes(datapath=path).calculate(s_col, o_col, a_col))
            priv_scores_offset.append(ProxyMutualInformationPrivbayes(datapath=path).calculate(s_col, o_col, a_col))
            tvd_scores.append(ProxyMutualInformationTVD(datapath=path).calculate([s_col, o_col, a_col]))
        vals["MI"][ds_name] = mi_scores
        vals["PRIV_ORIG"][ds_name] = priv_scores_orig
        vals["PRIV_OFFSET"][ds_name] = priv_scores_offset
        vals["TVD"][ds_name] = tvd_scores

    # labels per dataset
    ds_labels = {ds["name"]: [f"{s},{o} | {a}" for (s, o, a) in ds["attrs"]] for ds in DATASETS}

    # --- figure
    n_rows, n_cols = len(PROXIES), len(DATASETS)
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
        ax.annotate(f"{h:.3f}", xy=(x, h), xytext=(0, offset),
                    textcoords="offset points", ha="center", va=va, fontsize=ANNOT_FS)

    # column titles
    for c, ds in enumerate(DATASETS):
        axes[0, c].set_title(ds["name"], fontsize=TITLE_FS)

    # draw bars
    for r, (proxy_title, proxy_key) in enumerate(PROXIES):
        color = PROXY_COLOR[proxy_key]

        # common y-range per row (allow negatives for PRIV_ORIG if any)
        row_vals = []
        for ds in DATASETS:
            row_vals.extend(vals[proxy_key][ds["name"]])
        row_vals = np.asarray(row_vals, dtype=float)


        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            y = vals[proxy_key][ds["name"]]
            labels = ds_labels[ds["name"]]

            if row_vals.size == 0:
                row_min, row_max = -1.0, 1.0
            else:
                row_min, row_max = float(np.min(row_vals)), float(np.max(row_vals))

            # Always add some proportional padding
            pad = 0.05 * max(1.0, abs(row_max - row_min))  # 5% of the data range

            if proxy_key in ("MI", "PRIV_OFFSET", "TVD"):
                ymin, ymax = 0.0, row_max + pad
            else:
                ymin, ymax = row_min - pad, row_max + pad

            ax.set_ylim(ymin, ymax)

            # tighter spacing: compress positions + narrower bars
            x = np.arange(len(y)) * 0.65
            bars = ax.bar(x, y, color=color, width=0.5)
            for rect in bars:
                annotate_bar(ax, rect)

            ax.set_xticks(x)

            # bottom row: bigger xticklabels
            if r == n_rows - 1:
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=TICK_FS)
            else:
                ax.set_xticklabels([])

            if c == 0:
                ax.set_ylabel(proxy_title, fontsize=ROWLAB_FS, labelpad=20)

            ax.yaxis.grid(True, linestyle=":", linewidth=0.9, alpha=0.65)
            ax.tick_params(axis='y', labelsize=TICK_FS)
            ax.set_ylim(row_min, row_max)

            # NO zero line anywhere (PrivBayes orig can be negative; limits already handle it)
            if c == 0:
                ax.set_ylabel(proxy_title, fontsize=ROWLAB_FS)

    # No legend (colors are per-row and labeled by y-axis)
    plt.savefig("plots/plot1.png", dpi=220)
    plt.show()

def create_plot_2():
    # --- config ---------------------------------------------------
    DATASETS = [
        {"name": "Adult", "path": "data/adult.csv", "attrs": [
            ("sex", "income>50K", "education"),
            ("race", "income>50K", "education"),
            ("education", "education-num", "sex"),
        ]},
        {"name": "StackOverflow", "path": "data/stackoverflow.csv", "attrs": [
            ("Country", "EdLevel", "Age"),
            ("Country", "DevType", "Age"),
            ("Country", "SurveyLength", "Age"),
            ("Country", "SOVisitFreq", "Age"),
        ]},
        {"name": "COMPAS", "path": "data/compas.csv", "attrs": [
            ("race", "c_charge_desc", "age"),
            ("race", "score_text", "age"),
            ("race", "sex", "age"),
        ]},
    ]

    MEASURES = [
        ("TVD Proxy", "TVD"),
        ("Tuple Contribution", "AUC"),
    ]

    # blue and orange colors
    MEASURE_COLOR = {
        "TVD": "#1f77b4",   # blue
        "AUC": "#ff7f0e",   # orange
    }

    TITLE_FS = 26
    ROWLAB_FS = 26
    TICK_FS = 24
    ANNOT_FS = 18

    # --- compute values ------------------------------------------
    vals = {k: {} for _, k in MEASURES}
    for ds in DATASETS:
        ds_name, path, attrs = ds["name"], ds["path"], ds["attrs"]
        tvd_scores, auc_scores = [], []
        for s_col, o_col, a_col in attrs:
            tvd_scores.append(ProxyMutualInformationTVD(datapath=path).calculate([s_col, o_col, a_col]))
            auc_scores.append(TupleContribution(datapath=path).calculate([s_col, o_col, a_col]))
        vals["TVD"][ds_name] = tvd_scores
        vals["AUC"][ds_name] = auc_scores

    ds_labels = {ds["name"]: [f"{s},{o} | {a}" for (s, o, a) in ds["attrs"]] for ds in DATASETS}

    # --- figure ---------------------------------------------------
    n_rows, n_cols = len(MEASURES), len(DATASETS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), constrained_layout=True)

    def annotate_bar(ax, rect):
        h = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2.0
        va = 'bottom' if h >= 0 else 'top'
        offset = 6 if h >= 0 else -8
        ax.annotate(f"{h:.3f}", xy=(x, h), xytext=(0, offset),
                    textcoords="offset points", ha="center", va=va, fontsize=ANNOT_FS)

    # Column titles
    for c, ds in enumerate(DATASETS):
        axes[0, c].set_title(ds["name"], fontsize=TITLE_FS)

    # Draw bars ----------------------------------------------------
    for r, (measure_title, measure_key) in enumerate(MEASURES):
        color = MEASURE_COLOR[measure_key]
        row_vals = []
        for ds in DATASETS:
            row_vals.extend(vals[measure_key][ds["name"]])
        row_vals = np.asarray(row_vals, dtype=float)
        row_min, row_max = float(np.min(row_vals)), float(np.max(row_vals))
        pad = 0.05 * max(1.0, abs(row_max - row_min))
        ymin, ymax = 0.0, row_max + pad

        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            y = vals[measure_key][ds["name"]]
            labels = ds_labels[ds["name"]]

            ax.set_ylim(ymin, ymax)
            x = np.arange(len(y)) * 0.65
            bars = ax.bar(x, y, color=color, width=0.5)
            for rect in bars:
                annotate_bar(ax, rect)

            ax.set_xticks(x)
            if r == n_rows - 1:
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=TICK_FS)
            else:
                ax.set_xticklabels([])

            if c == 0:
                ax.set_ylabel(measure_title, fontsize=ROWLAB_FS, labelpad=20)

            ax.yaxis.grid(True, linestyle=":", linewidth=0.9, alpha=0.65)
            ax.tick_params(axis='y', labelsize=TICK_FS)

    plt.savefig("plots/plot2.png", dpi=220)
    plt.show()


######################################### Experiments ##########################################

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

measures = {
    "Proxy Mutual Information TVD": ProxyMutualInformationTVD,
    "Proxy RepairMaxSat": ProxyRepairMaxSat,
    "Tuple Contribution": TupleContribution,
}

timeout_seconds = 1 * 60 * 60

from sklearn.preprocessing import LabelEncoder
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
    repetitions=5,
    outfile="plots/experiment1.png"
):
    """Plotting average runtimes over 'repetitions' repetitions per measure and dataset while keeping criteria constant
    for every dataset and increasing the number of tuples."""
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
        num_tupless = num_tupless_per_dataset[ds_name]

        # store mean / min / max per measure
        results = {
            measure_name: {"mean": [], "min": [], "max": []}
            for measure_name in measures.keys()
        }

        for measure_name, measure_cls in measures.items():
            if measure_name == "Proxy RepairMaxSat" and path == "data/census.csv":  # this dataset timeouted whole
                # no data -> all NaNs
                for _ in num_tupless:
                    results[measure_name]["mean"].append(np.nan)
                    results[measure_name]["min"].append(np.nan)
                    results[measure_name]["max"].append(np.nan)
                continue

            flag_timeout = False
            for num_tuples in num_tupless:
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
        xs = np.arange(len(num_tupless))
        tick_labels = [
            f"{num_tuples // 1000}K" if num_tuples % 1000 == 0 else str(num_tuples)
            for num_tuples in num_tupless
        ]

        # ↓↓↓ reduce xtick labels specifically for IPUMS-CPS ↓↓↓
        if ds_name == "IPUMS-CPS":
            # e.g. show only indices 0, 2, 4, 6 (for 7 points)
            if len(num_tupless) >= 7:
                show_idx = [0, 2, 4, 6]
            else:
                show_idx = list(range(len(num_tupless)))
            ax.set_xticks(np.array(show_idx))
            ax.set_xticklabels([tick_labels[i] for i in show_idx])
        else:
            ax.set_xticks(xs)
            ax.set_xticklabels(tick_labels)
        # ↑↑↑ end IPUMS-CPS special-casing ↑↑↑

        for measure_name, stats in results.items():
            means = np.array(stats["mean"])
            lows  = np.array(stats["min"])
            highs = np.array(stats["max"])

            # plot main line
            line, = ax.plot(xs, means, marker="o", linewidth=2, label=measure_name)

            # shadow band = min/max across repetitions
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
    repetitions=5,
    outfile="plots/experiment2.png"
):
    """Plot average runtimes over `repetitions` per measure and dataset
    while keeping #tuples constant and increasing the number of criteria."""
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
    repetitions=5,
    outfile="plots/experiment3.png"
):
    """Plot relative L1 error vs epsilon, averaging over `repetitions`
    and showing min/max bands as shadows around each line."""

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
            if measure_name == "Proxy RepairMaxSat" and path == "data/census.csv":  # this dataset timeouted
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
        epsilon: float = 1.0,
        num_tuples: int = 100000,
        repetitions: int = 5,
        outfile: str = "plots/experiment4.png",
):
    """
    For each dataset: histogram with X = fairness criteria, Y = value.
    For each criterion, show two bars: MutualInformation and its proxy
    ProxyMutualInformationTVD, averaged over `repetitions`.
    """

    plt.rcParams.update({
        "axes.titlesize": 32,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 22,
        "figure.titlesize": 32,
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
            # Normalize criterion label for plotting
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
            # Use the same df but re-instantiate measures each repetition
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
                        print(f"Skipping the iteration due to timeout.")
                        # skip this repetition for that criterion

                # ProxyMutualInformationTVD
                with ThreadPoolExecutor() as executor:
                    try:
                        tvd_val = executor.submit(
                            tvd_measure.calculate, [criterion], epsilon=epsilon
                        ).result(timeout=timeout_seconds)
                        tvd_sums[crit_label] += float(tvd_val)
                        tvd_counts[crit_label] += 1
                    except TimeoutError:
                        print(f"Skipping the iteration due to timeout.")
                        # skip this repetition for that criterion

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
        ax.bar(x - width / 2, mi_vals, width, label="MutualInformation")
        ax.bar(x + width / 2, tvd_vals, width, label="ProxyMutualInformationTVD")

        ax.set_xticks(x)
        ax.set_xticklabels(crit_labels, rotation=45, ha="right")
        ax.set_title(ds_name)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("measure value")

    fig.suptitle(f"Comparison of MutualInformation and ProxyMutualInformationTVD, at most "
                 f"{round(num_tuples / 1000)}K tuples, ε = {epsilon}",
                 y=1.03)
    fig.tight_layout()

    import os
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_5(
    num_tuples=100000,
    repetitions=5,
    epsilon=None,
    outfile="plots/experiment5.png",
):
    """Plot TupleContribution runtime over `repetitions` per dataset while keeping #tuples constant
    and increasing k (using ks_per_dataset for each dataset)."""

    ks_per_dataset = {
        "Adult": [1000, 5000, 10000, 15000, 30000],
        "IPUMS-CPS": [5000, 10000, 50000, 100000, 300000, 600000, 1000000],
        "Stackoverflow": [5000, 10000, 20000, 40000, 60000],
        "Compas": [1000, 1500, 3000, 7000, 10000],
        "Healthcare": [100, 200, 400, 700, 1000],
    }

    import os
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator  # <-- added

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

        # Stats for TupleContribution runtimes per k
        stats = {"mean": [], "min": [], "max": []}

        # One TupleContribution instance on the sampled data
        m = TupleContribution(data=sample)

        flag_timeout = False
        for k in ks:
            if flag_timeout:
                # If we already timed out for a smaller k, fill with NaNs
                print("Skipping the iteration due to timeout.")
                stats["mean"].append(np.nan)
                stats["min"].append(np.nan)
                stats["max"].append(np.nan)
                continue

            runtimes_rep = []
            for _ in range(repetitions):
                start_time = time.time()
                with ThreadPoolExecutor() as executor:
                    try:
                        _ = executor.submit(
                            m.calculate,
                            criteria,   # fixed set of criteria
                            k=k,
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

            stats["mean"].append(mean_v)
            stats["min"].append(min_v)
            stats["max"].append(max_v)

        # ---- Plotting for this dataset ----
        xs = np.arange(len(ks))
        means = np.array(stats["mean"])
        lows  = np.array(stats["min"])
        highs = np.array(stats["max"])

        # main line
        line, = ax.plot(xs, means, marker="o", linewidth=2,
                        label="TupleContribution runtime")

        # shadow band = min/max across repetitions
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

        # tick labels (fewer for IPUMS-CPS)
        full_tick_labels = [str(k) if k % 1000 != 0 else f"{k // 1000}K" for k in ks]
        if ds_name == "IPUMS-CPS":
            show_idx = [0, 2, 4, 6] if len(ks) >= 7 else list(range(len(ks)))
            ax.set_xticks(np.array(show_idx))
            ax.set_xticklabels([full_tick_labels[i] for i in show_idx])
        else:
            ax.set_xticks(xs)
            ax.set_xticklabels(full_tick_labels)

        ax.set_xlabel("k (top-k tuples)")
        ax.set_yscale('log')

        # ↓↓↓ FEWER **Y** TICKS FOR HEALTHCARE ↓↓↓
        if ds_name == "Healthcare":
            # e.g. at most 3 major ticks on log scale
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
        # ↑↑↑

        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("runtime (s), log scale")
    fig.suptitle(
        f"Runtime of TupleContribution as Function of k, at most {round(num_tuples / 1000)}K tuples",
        y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=256, bbox_inches="tight")
    plt.show()


def run_experiment_6(
    num_tuples=100000,
    repetitions=5,
    epsilon=1.0,
    outfile="plots/experiment6.png",
):
    """Plot average relative L1 error of TupleContribution over `repetitions` per dataset,
    while keeping #tuples constant and increasing k (using ks_per_dataset for each dataset)."""

    ks_per_dataset = {
        "Adult": [1000, 5000, 10000, 15000, 30000],
        "IPUMS-CPS": [5000, 10000, 50000, 100000, 300000, 600000, 1000000],
        "Stackoverflow": [5000, 10000, 20000, 40000, 60000],
        "Compas": [1000, 1500, 3000, 7000, 10000],
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


def run_experiment_7_unconditional(
        epsilon: Optional[float] = None,
        num_tuples: int = 100000,
        num_tuples_repair: int = 1000,
        repetitions: int = 5,
        outfile: str = "plots/experiment7_unconditional.xlsx",
):
    import os
    import matplotlib.pyplot as plt

    # ==== DP-SGD / training params ====
    DP_NOISE_MULT = 1.0        # Gaussian noise multiplier
    DP_MAX_GRAD_NORM = 1.0     # Per-sample clipping norm
    BATCH_SIZE = 300000        # we'll batch smaller if sample is smaller
    EPOCHS = 5
    LR = 1e-2
    POS_THRESHOLD = 0.5        # for thresholding predictions
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
        # privacy_engine = PrivacyEngine()
        # model, optimizer, train_loader = privacy_engine.make_private(
        #     module=model,
        #     optimizer=optimizer,
        #     data_loader=train_loader,
        #     noise_multiplier=noise_multiplier,
        #     max_grad_norm=max_grad_norm,
        # )

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
        print(f"\n=== DATASET (unconditional): {ds_name} ===")
        path = spec["path"]
        criteria = spec["criteria"]

        # accumulators: criterion -> sums
        sum_linear = defaultdict(lambda: np.array([0.0, 0.0], dtype=float))   # (MAE, DP)
        sum_mlp    = defaultdict(lambda: np.array([0.0, 0.0], dtype=float))   # (MAE, DP)
        sum_mi     = defaultdict(float)                                       # MutualInformation
        sum_tvd    = defaultdict(float)                                       # ProxyMutualInformationTVD
        sum_repair = defaultdict(float)                                       # ProxyRepairMaxSat
        sum_tc     = defaultdict(float)                                       # TupleContribution

        for rep in range(repetitions):
            for criterion in criteria:
                # Use only protected + response; ignore admissible for the unconditional version
                protected = criterion[0]
                response = criterion[1]
                admissible = criterion[2] if len(criterion) > 2 else None

                crit_label = f"{protected}->{response}".lower()
                crit_uncond = [protected, response]  # 2-element criterion for unconditional measures

                # encode and clean only needed columns (protected + response)
                df = _encode_and_clean(path, crit_uncond)
                df_repair = df.sample(n=min(num_tuples_repair, len(df)))

                # limit number of tuples
                n_total = len(df)
                n_use = int(min(n_total, num_tuples))
                if n_use < n_total:
                    df = df.sample(n=n_use)

                # ======= MEASURES (unconditional; pass 2-element criterion) =======
                mutual_information = MutualInformation(data=df)
                sum_mi[crit_label] += float(mutual_information.calculate([crit_uncond], epsilon=epsilon))

                tvd_proxy = ProxyMutualInformationTVD(data=df)
                sum_tvd[crit_label] += float(tvd_proxy.calculate([crit_uncond], epsilon=epsilon))

                if path == "data/census.csv":
                    repair_proxy = ProxyRepairMaxSat(data=df_repair)
                else:
                    repair_proxy = ProxyRepairMaxSat(data=df)
                sum_repair[crit_label] += float(repair_proxy.calculate([crit_uncond], epsilon=epsilon))

                tc_proxy = TupleContribution(data=df)
                sum_tc[crit_label] += float(tc_proxy.calculate([crit_uncond], epsilon=epsilon))

                # ======= MODELS (Regression / “Random Forest” proxy MLP) =======
                feat_cols = [protected]
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
                X_train, X_test, y_train, y_test, y_real_train, y_real_test, prot_train, prot_test = train_test_split(
                    X_full, y_full, y_real_full, prot_full,
                    test_size=TEST_SIZE,
                    stratify=stratify
                )
                prot_test = pd.Series(prot_test).reset_index(drop=True)

                # to torch
                train_ds, in_dim = _to_torch(X_train, y_train)
                effective_batch = min(BATCH_SIZE, len(train_ds))
                train_loader = DataLoader(
                    train_ds, batch_size=effective_batch,
                    shuffle=True, drop_last=False
                )

                # helper to unscale predictions back to real units
                def _unscale(pred_scaled: np.ndarray) -> np.ndarray:
                    if y_max > y_min:
                        return pred_scaled * (y_max - y_min) + y_min
                    else:
                        return np.full_like(pred_scaled, y_min)

                # --- Regression (DPLinear) ---
                lin = DPLinear(in_dim)
                _train_dp_sgd(lin, train_loader)
                yhat_lin_scaled = _predict(lin, X_test)
                mae_lin = float(np.mean(np.abs(yhat_lin_scaled - y_test)))  # in [0,1]

                dp_lin = _demographic_parity(
                    yhat_lin_scaled,
                    prot_test,
                    threshold=POS_THRESHOLD
                )
                sum_linear[crit_label] += np.array([mae_lin, dp_lin], dtype=float)

                # --- Random Forest (DPMLP proxy) ---
                mlp = DPMLP(in_dim)
                _train_dp_sgd(mlp, train_loader)
                yhat_mlp_scaled = _predict(mlp, X_test)
                yhat_mlp_real = _unscale(yhat_mlp_scaled)
                mae_mlp = float(np.mean(np.abs(yhat_mlp_real - y_real_test)))  # L1 vs REAL values

                dp_mlp = _demographic_parity(
                    yhat_mlp_scaled,
                    prot_test,
                    threshold=POS_THRESHOLD
                )
                sum_mlp[crit_label] += np.array([mae_mlp, dp_mlp], dtype=float)

        # ===== Averaging over repetitions; append to global table =====
        crits_sorted = sorted(sum_linear.keys())
        for crit in crits_sorted:
            tvd_avg = sum_tvd[crit] / repetitions
            repair_avg = sum_repair[crit] / repetitions
            tc_avg = sum_tc[crit] / repetitions

            mae_lin_avg = sum_linear[crit][0] / repetitions
            dp_lin_avg = sum_linear[crit][1] / repetitions
            mae_mlp_avg = sum_mlp[crit][0] / repetitions
            dp_mlp_avg = sum_mlp[crit][1] / repetitions

            all_rows.append([
                round(tvd_avg, 4),
                round(repair_avg, 4),
                round(tc_avg, 4),
                round(mae_lin_avg, 4),
                round(dp_lin_avg, 4),
                round(mae_mlp_avg, 4),
                round(dp_mlp_avg, 4),
            ])
            all_index.append((ds_name, crit))

    # ===== Build one big table with row MultiIndex (Dataset, Criterion) =====
    row_index = pd.MultiIndex.from_tuples(all_index, names=["Dataset", "Criterion"])
    col_index = pd.MultiIndex.from_tuples([
        ("ProxyMutualInformationTVD", ""),
        ("RepairMaxSat", ""),
        ("TupleContribution", ""),
        ("Regression", "L1 Error"),
        ("Regression", "Unfairness (DP)"),
        ("Random Forest", "L1 Error"),
        ("Random Forest", "Unfairness (DP)"),
    ])

    table_all = pd.DataFrame(all_rows, index=row_index, columns=col_index)

    # ---------- Histogram plot (unconditional) ----------
    # Sort criteria by Regression Unfairness (DP)
    fairness_col = ("Regression", "Unfairness (DP)")
    fairness_vals = table_all[fairness_col].to_numpy(dtype=float)

    # order = np.argsort(fairness_vals)[::-1]  # for descending (most unfair left)
    order = np.argsort(fairness_vals)         # ascending (least -> most unfair)

    # Reorder labels and x positions
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
    labels = ["TVD", "RepairMaxSat", "TupleContribution", "Regression (DP)"]

    for i, col in enumerate(measure_cols + [fairness_col]):
        vals = table_all[col].to_numpy(dtype=float)
        vals = vals[order]   # sort bars by unfairness order
        ax.bar(x + offsets[i], vals, width, label=labels[i])

    ax.set_xticks(x)
    ax.set_xticklabels(crit_labels, rotation=45, ha="right", fontsize=7)

    ax.set_yscale('log')
    ax.set_ylabel("Unfairness / Measure value (log scale)", fontsize=9)
    ax.set_title("Unconditional: Measures and Regression Demographic Parity per Criterion", fontsize=10)

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
        print("\n=== Unconditional experiment (averaged, rounded to 4 decimals) ===")
        print(table_all)

    # save to Excel
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    table_all.to_excel(outfile, merge_cells=True)


def run_experiment_7_conditional(
        epsilon: Optional[float] = None,
        num_tuples: int = 100000,
        num_tuples_repair: int = 1000,
        repetitions: int = 5,
        outfile="plots/experiment7_conditional.xlsx"
):
    import os
    import matplotlib.pyplot as plt

    # ==== DP-SGD / training params ====
    DP_NOISE_MULT = 1.0        # Gaussian noise multiplier
    DP_MAX_GRAD_NORM = 1.0     # Per-sample clipping norm
    BATCH_SIZE = 300000        # we'll batch smaller if sample is smaller
    EPOCHS = 20
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

                # helper to unscale predictions back to real units
                def _unscale(pred_scaled: np.ndarray) -> np.ndarray:
                    if y_max > y_min:
                        return pred_scaled * (y_max - y_min) + y_min
                    else:
                        return np.full_like(pred_scaled, y_min)

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
                yhat_mlp_scaled = _predict(mlp, X_test)
                yhat_mlp_real = _unscale(yhat_mlp_scaled)
                mae_mlp = float(np.mean(np.abs(yhat_mlp_real - y_real_test)))  # L1 vs REAL values
                csp_mlp = _conditional_statistical_parity(
                    yhat_mlp_scaled,
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
    run_experiment_5()
    run_experiment_6()
    run_experiment_7_unconditional()
    # run_experiment_7_conditional()

