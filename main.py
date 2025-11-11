import time
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from proxy_repair_maxsat import ProxyRepairMaxSat
from tuple_contribution import TupleContribution
from mutual_information import MutualInformation
from proxy_mutual_information_tvd import ProxyMutualInformationTVD
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

TIME_LIMIT = 7200  # 2 hours in seconds

def _encode_and_clean(data_path, cols):
    df = pd.read_csv(data_path)
    df = df.replace(["NA", "N/A", ""], pd.NA).dropna(subset=cols).copy()
    for c in cols:
        df[c] = LabelEncoder().fit_transform(df[c])
    return df


def run_experiment_1(
    epsilon=None,
    repeats=5,
    save=True,
    outfile="plots/experiment1.png",
    seed=123
):
    sample_sizes_per_dataset = {
        "Adult": [1000, 5000, 10000, 15000, 30000],
        "IPUMS-CPS": [5000, 10000, 100000, 300000, 500000],
        "Stackoverflow": [5000, 10000, 20000, 40000, 60000],
        "Compas": [1000, 1500, 3000, 7000, 10000],
        "Healthcare": [100, 200, 400, 700, 1000],
    }

    def _runtime_avg_over_repeats_and_criteria(measure_name, measure_cls, data, criteria, sample_size,
                                               repeats=repeats, epsilon=epsilon, seed=seed):
        rng = np.random.RandomState(seed)
        crit_times = []

        for crit in criteria:
            rep_times = []
            for r in range(repeats):
                n = min(sample_size, len(data))
                sample = data.sample(n=min(len(data), sample_size), replace=False,
                                     random_state=rng.randint(0, 1_000_000))
                m = measure_cls(data=sample)

                start_time = time.time()
                try:
                    _ = m.calculate([crit], epsilon=epsilon)
                except Exception:
                    # In case the measure fails, skip
                    rep_times.append(np.nan)
                    continue

                elapsed = time.time() - start_time

                # Skip if runtime > 2 hours
                if elapsed > TIME_LIMIT:
                    print(f"Skipping {measure_name} for {crit} (runtime exceeded 2h)")
                    rep_times.append(np.nan)
                    continue

                rep_times.append(elapsed)
            crit_times.append(np.nanmean(rep_times))
        return float(np.nanmean(crit_times))

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        path = spec["path"]
        criteria = spec["criteria"]
        sample_sizes = sample_sizes_per_dataset[ds_name]
        data = _encode_and_clean(path, criteria[0])

        for measure_name, measure_cls in measures.items():
            runtimes = []
            for n in sample_sizes:
                t = _runtime_avg_over_repeats_and_criteria(
                    measure_name, measure_cls, data, criteria, n, repeats=repeats, epsilon=epsilon)
                runtimes.append(t)

            # Connect through skipped points (matplotlib ignores NaNs automatically)
            ax.plot(sample_sizes, runtimes, marker="o", linewidth=2, label=measure_name)

        ax.set_title(ds_name)
        ax.set_xlabel("number of tuples (sample size)")
        ax.set_ylabel("runtime (s)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.suptitle("Runtime as function of Sample Size", y=1.02)
    fig.tight_layout()

    if save:
        import os
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=180, bbox_inches="tight")
        print(f"Saved {outfile}")

    plt.show()


def run_experiment_2(
    epsilons=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
    sample_size=300000,
    repeats=5,
    save=True,
    outfile="plots/experiment2.png",
    seed=123
):
    def _rel_error(x, y, tiny=1e-12):
        denom = max(abs(y), tiny)
        return abs(x - y) / denom

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    })

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (ds_name, spec) in zip(axes, datasets.items()):
        rng = np.random.RandomState(seed)
        path = spec["path"]
        criteria = spec["criteria"]
        data = _encode_and_clean(path, criteria[0])
        sample = data.sample(n=min(len(data), sample_size), replace=False,
                             random_state=rng.randint(0, 1_000_000))

        baselines = []
        for measure_name, measure_cls in measures.items():
            Y = []
            for crit in criteria:
                m = measure_cls(data=sample)
                y_val = m.calculate([crit], epsilon=None)
                Y.append(float(y_val))
            baselines.append((measure_name, np.array(Y, dtype=float)))

        for (measure_name, measure_cls), (_, Y_vec) in zip(measures.items(), baselines):
            rel_errors_for_eps = []
            for eps in epsilons:
                crit_means = []
                for crit_idx, crit in enumerate(criteria):
                    y = Y_vec[crit_idx]
                    reps = []
                    for r in range(repeats):
                        m = measure_cls(data=sample)
                        start_time = time.time()
                        try:
                            x = float(m.calculate([crit], epsilon=eps))
                        except Exception:
                            reps.append(np.nan)
                            continue
                        elapsed = time.time() - start_time

                        # Skip if computation exceeds 2 hours (for RepairMaxSat only)
                        if elapsed > TIME_LIMIT:
                            print(f"Skipping {measure_name} for {crit} at ε={eps} (runtime > 2h)")
                            reps.append(np.nan)
                            continue

                        reps.append(_rel_error(x, y))
                    crit_means.append(np.nanmean(reps))
                rel_errors_for_eps.append(np.nanmean(crit_means))

            ax.plot(epsilons, rel_errors_for_eps, marker="o", linewidth=2, label=measure_name)

        ax.set_title(ds_name)
        ax.set_xlabel("privacy budget ε")
        ax.set_ylabel("relative L1 error")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.suptitle("Relative L1 Error as function of Privacy Budget", y=1.02)
    fig.tight_layout()

    if save:
        import os
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        plt.savefig(outfile, dpi=180, bbox_inches="tight")
        print(f"Saved {outfile}")

    plt.show()

def run_experiment_3():
    # ==== DP-SGD / training params ====
    DP_NOISE_MULT = 1.0        # Gaussian noise multiplier
    DP_MAX_GRAD_NORM = 1.0     # Per-sample clipping norm
    BATCH_SIZE = 300000        # you asked for this value; we'll batch smaller if sample is smaller
    EPOCHS = 5
    LR = 1e-2
    POS_THRESHOLD = 0.5        # for CSP thresholding of predictions
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TEST_SIZE = 0.3
    RANDOM_STATE = 42

    # ==== Models ====
    class DPLinear(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.linear = nn.Linear(in_dim, 1)
        def forward(self, x):
            return torch.sigmoid(self.linear(x))  # keep outputs in [0,1]

    class DPMLP(nn.Module):
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
    def _to_torch(X: np.ndarray, y: np.ndarray) -> Tuple[TensorDataset, int]:
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
        try:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
        except ImportError:
            raise RuntimeError("Opacus not installed. Install via: pip install opacus torch")

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

    # ==== safety: check columns exist before calling _encode_and_clean ====
    def _has_all_columns(csv_path: str, cols: list) -> bool:
        try:
            header = pd.read_csv(csv_path, nrows=0)
            return set(cols).issubset(set(header.columns))
        except Exception:
            return False

    # === Core loop over datasets and criteria (expects `datasets` to be defined in scope) ===
    for ds_name, spec in datasets.items():
        print(f"\n=== DATASET: {ds_name} ===")
        path = spec["path"]
        crits = spec["criteria"]

        # results dicts: criterion (lowercase) -> (mae, csp)
        results_linear: Dict[str, Tuple[float, float]] = {}
        results_mlp:    Dict[str, Tuple[float, float]] = {}

        for crit in crits:
            protected, response = crit[0], crit[1]
            admissible = crit[2] if len(crit) == 3 else None
            cols_needed = [protected, response] + ([admissible] if admissible else [])
            # lowercase label
            crit_label = (f"{protected}->{response}|{admissible}" if admissible else f"{protected}->{response}").lower()

            if not _has_all_columns(path, cols_needed):
                print(f"Skipping {crit_label}: missing columns in {path}")
                continue

            # preprocess ONLY the needed columns (as you requested)
            df_enc = _encode_and_clean(path, cols_needed)

            # choose features: use encoded protected + admissible; target is encoded response
            feat_cols = [protected] + ([admissible] if admissible else [])
            X_full = df_enc[feat_cols].to_numpy(dtype=float)
            y_full = df_enc[response].to_numpy(dtype=float)

            # normalize y into [0,1] so MAE is well-behaved (in case label encoder created >2 classes)
            if np.ptp(y_full) > 0:
                y_full = (y_full - y_full.min()) / (np.ptp(y_full))

            # limit training size
            n_total = len(df_enc)
            n_use = int(min(n_total, 300000))
            if n_use < n_total:
                df_enc = df_enc.sample(n=n_use, random_state=RANDOM_STATE)
                X_full = df_enc[feat_cols].to_numpy(dtype=float)
                y_full = df_enc[response].to_numpy(dtype=float)
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
                X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
            )

            # protected/admissible series for CSP on TEST split
            prot_test = df_enc[protected].iloc[X_test.shape[0]*0:len(X_test)].reset_index(drop=True)  # placeholder
            # Better: recompute masks from the same split indices
            # build indices by a second split call with return of indices
            idx_all = np.arange(len(X_full))
            _, idx_test = train_test_split(
                idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
            )
            prot_test = df_enc[protected].iloc[idx_test].reset_index(drop=True)
            if admissible:
                adm_test = df_enc[admissible].iloc[idx_test].reset_index(drop=True)
            else:
                # if no admissible, make a single-group placeholder (all zeros)
                adm_test = pd.Series(np.zeros(len(idx_test), dtype=int))

            # to torch
            train_ds, in_dim = _to_torch(X_train, y_train)
            effective_batch = min(BATCH_SIZE, len(train_ds))
            train_loader = DataLoader(train_ds, batch_size=effective_batch, shuffle=True, drop_last=False)

            # --- DPLinear ---
            lin = DPLinear(in_dim)
            _train_dp_sgd(lin, train_loader)
            yhat_lin = _predict(lin, X_test)
            mae_lin = float(np.mean(np.abs(yhat_lin - y_test)))
            csp_lin = _conditional_statistical_parity(yhat_lin, prot_test, adm_test, threshold=POS_THRESHOLD)
            results_linear[crit_label] = (mae_lin, csp_lin)

            # --- DPMLP ---
            mlp = DPMLP(in_dim)
            _train_dp_sgd(mlp, train_loader)
            yhat_mlp = _predict(mlp, X_test)
            mae_mlp = float(np.mean(np.abs(yhat_mlp - y_test)))
            csp_mlp = _conditional_statistical_parity(yhat_mlp, prot_test, adm_test, threshold=POS_THRESHOLD)
            results_mlp[crit_label] = (mae_mlp, csp_mlp)

        # Print dicts
        print("DPLinear results (criterion -> (mae, csp)):")
        print(results_linear)
        print("DPMLP results (criterion -> (mae, csp)):")
        print(results_mlp)



if __name__ == "__main__":
    # create_plot_1()
    # create_plot_2()
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
