import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict

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
                          ["Country", "DevType", "YearsCodePro"], ["Age", "BuyNewTool", "BuildvsBuy"]]

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
    "Mutual Information": MutualInformation,
    "Proxy Mutual Information TVD": ProxyMutualInformationTVD,
    "Proxy RepairMaxSat": ProxyRepairMaxSat,
    "Tuple Contribution": TupleContribution,
}

timeout_seconds = 1 * 60 * 60

def _encode_and_clean(data_path, cols):
    df = pd.read_csv(data_path)
    df = df.replace(["NA", "N/A", ""], pd.NA).dropna(subset=cols).copy()
    for c in cols:
        df[c] = LabelEncoder().fit_transform(df[c])
    return df


def plot_legend(save=True, outfile="plots/legend.png"):
    """Creating a standalone legend figure for Experiment 1 with the four measures arranged in a single horizontal row,
    and save it to `outfile`.
    """
    labels = [
        "Mutual Information",
        "Proxy Mutual Information TVD",
        "Proxy RepairMaxSat",
        "Tuple Contribution",
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(8, 1.6))

    handles = []
    for i, label in enumerate(labels):
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
        labels,
        loc="center",
        ncol=len(labels),
        frameon=True,
        fontsize=10,
    )

    if save:
        import os
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=180, bbox_inches="tight")
        print(f"Saved {outfile}")
    plt.show()


def run_experiment_1(
    epsilon=None,
    repetitions=5,
    save=True,
    outfile="plots/experiment1.png",
    seed=123
):
    """Plotting average runtimes over 'repetitions' repetitions per measure and dataset while keeping criteria constant
    for every dataset and increasing the number of tuples."""
    num_tupless_per_dataset = {
        "Adult": [1000, 5000, 10000, 15000, 30000],
        "IPUMS-CPS": [5000, 10000, 50000, 100000, 300000],
        "Stackoverflow": [5000, 10000, 20000, 40000, 60000],
        "Compas": [1000, 1500, 3000, 7000, 10000],
        "Healthcare": [100, 200, 400, 700, 1000],
    }

    plt.rcParams.update({
        "axes.titlesize": 28,
        "axes.labelsize": 26,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 28,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    rng = np.random.RandomState(seed)
    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        num_tupless = num_tupless_per_dataset[ds_name]

        results = {measure_name: [] for measure_name in measures.keys()}
        for measure_name, measure_cls in measures.items():
            if measure_name == "Proxy RepairMaxSat" and path in ["data/census.csv", "data/stackoverflow.csv"]:
                results[measure_name] += [np.nan] * len(num_tupless)
                continue
            flag_timeout = False
            for num_tuples in num_tupless:
                if flag_timeout:
                    print("Skipping next iterations because got timeout for smaller number of tuples.")
                    results[measure_name].append(np.nan)
                    continue
                results_for_num_tuples = []
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
                            results_for_num_tuples.append(elapsed_time)
                        except TimeoutError:
                            print("Skipping the iteration due to timeout.")
                            results_for_num_tuples.append(np.nan)
                            flag_timeout = True
                            break
                results[measure_name].append(np.mean(results_for_num_tuples))

        for measure_name, runtimes in results.items():
            ax.plot([str(num_tuples // 1000) + "K" if num_tuples % 1000 == 0 else str(num_tuples) for num_tuples in
                     num_tupless], runtimes, marker="o", linewidth=2, label=measure_name)
        ax.set_xlabel("number of tuples")
        ax.set_yscale('log')
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("runtime (s), log scale")
    fig.suptitle("Runtime as Function of Number of Tuples", y=1.02)
    fig.tight_layout()

    if save:
        import os
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=180, bbox_inches="tight")
        print(f"Saved {outfile}")
    plt.show()


def run_experiment_2(
    epsilon=None,
    num_tuples=100000,
    repetitions=5,
    save=True,
    outfile="plots/experiment2.png",
    seed=123
):
    """Plotting average runtimes over 'repetitions' repetitions per measure and dataset while keeping number of tuples
        constant and increasing the number of criteria in the set."""
    plt.rcParams.update({
        "axes.titlesize": 28,
        "axes.labelsize": 26,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 28,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    rng = np.random.RandomState(seed)
    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)

        results = {measure_name: [] for measure_name in measures.keys()}
        for measure_name, measure_cls in measures.items():
            flag_timeout = False
            for num_criteria in range(1, len(criteria) + 1):
                if measure_name == "Proxy RepairMaxSat" and (path == "data/census.csv" or
                                                             (path == "data/stackoverflow.csv" and num_criteria > 2)):
                    results[measure_name].append(np.nan)
                    continue
                results_for_num_criteria = []
                if flag_timeout:
                    print("Skipping next iterations because got timeout for smaller number of tuples.")
                    results[measure_name].append(np.nan)
                    continue
                for _ in range(repetitions):
                    n = min(num_tuples, len(data))
                    sample = data.sample(n=n, replace=False, random_state=rng.randint(0, 1_000_000))
                    m = measure_cls(data=sample)
                    start_time = time.time()
                    with ThreadPoolExecutor() as executor:
                        try:
                            _ = executor.submit(m.calculate, criteria[:num_criteria], epsilon=epsilon).result(timeout=timeout_seconds)
                            elapsed_time = time.time() - start_time
                            results_for_num_criteria.append(elapsed_time)
                        except TimeoutError:
                            print("Skipping the iteration due to timeout.")
                            results_for_num_criteria.append(np.nan)
                            flag_timeout = True
                            break
                results[measure_name].append(np.nan if np.any(np.isnan(results_for_num_criteria)) else
                                             np.mean(results_for_num_criteria))

        for measure_name, runtimes in results.items():
            ax.plot([str(int(num_criteria)) for num_criteria in range(1, len(criteria) + 1)], runtimes, marker="o",
                    linewidth=2, label=measure_name)
        ax.set_xlabel("number of criteria")
        ax.set_yscale('log')
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("runtime (s), log scale")
    fig.suptitle(f"Runtime as Function of Number of Criteria, number of tuples at most {round(num_tuples / 1000)}K", y=1.02)
    fig.tight_layout()

    if save:
        import os
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=180, bbox_inches="tight")
        print(f"Saved {outfile}")
    plt.show()


def run_experiment_3(
    epsilons=(0.1, 1, 5, 10),
    num_tuples=100000,
    repetitions=5,
    save=True,
    outfile="plots/experiment3.png",
    seed=123
):
    """Plotting average runtimes over 'repetitions' repetitions per measure and dataset while keeping criteria and the
        number of tuples constant for every dataset and increasing the privacy budget epsilon."""

    def _rel_error(x, y, tiny=1e-20):
        denom = max(abs(y), tiny)  # ensure we do not divide by 0
        return abs(x - y) / denom

    plt.rcParams.update({
        "axes.titlesize": 28,
        "axes.labelsize": 26,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 28,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    rng = np.random.RandomState(seed)
    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        cols_list = []
        for criterion in criteria:
            cols_list += criterion
        data = _encode_and_clean(path, cols_list)
        n = min(num_tuples, len(data))
        sample = data.sample(n=n, replace=False, random_state=rng.randint(0, 1_000_000))

        results = {measure_name: [] for measure_name in measures.keys()}
        for measure_name, measure_cls in measures.items():
            m = measure_cls(data=sample)
            if measure_name == "Proxy RepairMaxSat" and path in ["data/census.csv", "data/stackoverflow.csv"]:
                results[measure_name] += [np.nan] * len(epsilons)
                continue
            with ThreadPoolExecutor() as executor:
                try:
                    non_private_result = executor.submit(m.calculate, criteria, epsilon=None).result(
                        timeout=timeout_seconds)
                except TimeoutError:
                    print("Skipping the iteration due to timeout.")
                    results[measure_name] += [np.nan] * len(epsilons)
                    continue
            for epsilon in epsilons:
                results_for_epsilon = []
                for _ in range(repetitions):
                    with ThreadPoolExecutor() as executor:
                        try:
                            private_result = executor.submit(m.calculate, criteria, epsilon=epsilon).result(
                                timeout=timeout_seconds)
                            results_for_epsilon.append(_rel_error(private_result, non_private_result))
                        except TimeoutError:
                            print("Skipping the iteration due to timeout.")
                            results_for_epsilon.append(np.nan)
                            break
                results[measure_name].append(np.nan if np.any(np.isnan(results_for_epsilon)) else
                                             np.mean(results_for_epsilon))

        for measure_name, losses in results.items():
            ax.plot(epsilons, losses, marker="o", linewidth=2, label=measure_name)
        ax.set_xlabel("privacy budget ε")
        ax.set_yscale('log')
        ax.set_title(ds_name)
        ax.grid(True, linestyle="--", alpha=0.4)
        
    axes[0].set_ylabel("relative L1 error, log scale")
    fig.suptitle(f"Relative L1 Error as Function of Privacy Budget, number of tuples at most {round(num_tuples / 1000)}K",
                 y=1.02)
    fig.tight_layout()

    if save:
        import os
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=180, bbox_inches="tight")
        print(f"Saved {outfile}")
    plt.show()


def run_experiment_4(
        epsilon: float = 1.0,
        num_tuples: int = 30000,
        repetitions: int = 5,
        outfile="plots/experiment4.xlsx",
        seed=123
):
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
        sum_linear = defaultdict(lambda: np.array([0.0, 0.0], dtype=float))   # (MAE, CSP)
        sum_mlp    = defaultdict(lambda: np.array([0.0, 0.0], dtype=float))   # (MAE, CSP)
        sum_tvd    = defaultdict(float)                                       # ProxyMutualInformationTVD
        sum_repair = defaultdict(float)                                       # ProxyRepairMaxSat
        sum_tc     = defaultdict(float)                                       # TupleContribution

        for rep in range(repetitions):
            rep_seed = seed + rep

            for criterion in criteria:
                protected, response, admissible = criterion[0], criterion[1], criterion[2]
                crit_label = f"{protected}->{response} | {admissible}".lower()

                # encode and clean only needed columns
                df = _encode_and_clean(path, criterion)

                # limit number of tuples
                n_total = len(df)
                n_use = int(min(n_total, num_tuples))
                if n_use < n_total:
                    df = df.sample(n=n_use, random_state=rep_seed)

                # ======= PROXY MEASURES =======
                tvd_proxy = ProxyMutualInformationTVD(data=df)
                sum_tvd[crit_label] += float(tvd_proxy.calculate([criterion], epsilon=epsilon))

                repair_proxy = ProxyRepairMaxSat(data=df)
                sum_repair[crit_label] += float(repair_proxy.calculate([criterion], epsilon=epsilon))

                tc_proxy = TupleContribution(data=df)
                sum_tc[crit_label] += float(tc_proxy.calculate([criterion], epsilon=epsilon))

                # ======= MODELS (Regression / Random Forest) =======
                feat_cols = [protected] + ([admissible] if admissible else [])
                X_full = df[feat_cols].to_numpy(dtype=float)
                y_full = df[response].to_numpy(dtype=float)

                # normalize y to [0,1]
                if np.ptp(y_full) > 0:
                    y_full = (y_full - y_full.min()) / (np.ptp(y_full))

                # split
                stratify = None
                try:
                    y_round = np.round(y_full)
                    if len(np.unique(y_round)) <= 10:
                        stratify = y_round
                except Exception:
                    stratify = None

                X_train, X_test, y_train, y_test = train_test_split(
                    X_full, y_full, test_size=TEST_SIZE,
                    random_state=rep_seed, stratify=stratify
                )

                # indices for fairness computation
                idx_all = np.arange(len(X_full))
                _, idx_test = train_test_split(
                    idx_all, test_size=TEST_SIZE,
                    random_state=rep_seed, stratify=stratify
                )
                prot_test = df[protected].iloc[idx_test].reset_index(drop=True)
                if admissible:
                    adm_test = df[admissible].iloc[idx_test].reset_index(drop=True)
                else:
                    adm_test = pd.Series(np.zeros(len(idx_test), dtype=int))

                # to torch
                train_ds, in_dim = _to_torch(X_train, y_train)
                effective_batch = min(BATCH_SIZE, len(train_ds))
                train_loader = DataLoader(train_ds, batch_size=effective_batch,
                                          shuffle=True, drop_last=False)

                # --- Regression (DPLinear) ---
                lin = DPLinear(in_dim)
                _train_dp_sgd(lin, train_loader)
                yhat_lin = _predict(lin, X_test)
                mae_lin = float(np.mean(np.abs(yhat_lin - y_test)))
                csp_lin = _conditional_statistical_parity(yhat_lin, prot_test, adm_test,
                                                          threshold=POS_THRESHOLD)
                sum_linear[crit_label] += np.array([mae_lin, csp_lin], dtype=float)

                # --- Random Forest (DPMLP proxy) ---
                mlp = DPMLP(in_dim)
                _train_dp_sgd(mlp, train_loader)
                yhat_mlp = _predict(mlp, X_test)
                mae_mlp = float(np.mean(np.abs(yhat_mlp - y_test)))
                csp_mlp = _conditional_statistical_parity(yhat_mlp, prot_test, adm_test,
                                                          threshold=POS_THRESHOLD)
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
        (r"$\mathcal{U}^{TVD}_{MI}$", ""),
        (r"$\mathcal{U}^{SAT}_{R}$", ""),
        (r"$\mathcal{U}_{TC}$", ""),
        ("Regression", "Accuracy"),
        ("Regression", "Fairness"),
        ("Random Forest", "Accuracy"),
        ("Random Forest", "Fairness"),
    ])

    table_all = pd.DataFrame(all_rows, index=row_index, columns=col_index)

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
    table_all.to_excel(outfile, merge_cells=True)
    print(f"\nSaved combined table to {outfile}")




if __name__ == "__main__":
    # create_plot_1()
    # create_plot_2()
    # plot_legend()
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
