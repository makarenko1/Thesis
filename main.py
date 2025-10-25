import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt

from anomalous_treatment_count import AnomalousTreatmentCount
from mutual_information import MutualInformation
from pmi_threshold_detector import PMIThresholdDetector
from proxy_mutual_information_lipshitz import ProxyMutualInformationLipschitz
from proxy_mutual_information_nist_contest import ProxyMutualInformationNistContest
from proxy_mutual_information_privbayes import ProxyMutualInformationPrivbayes
from proxy_mutual_information_tvd import ProxyMutualInformationTVD
from repair_maxsat import ProxyRepairMaxSat
from residual_auc_measure import ResidualAUCMeasure
from shapley_values import ShapleyValues, LayeredShapleyValues


def calculate_mi_proxies(dataset, attributes, domain_paths, label):
    print(f"\n{label} dataset results:")
    for attribute_combination, domain_path in zip(attributes, domain_paths):
        col1, col2 = attribute_combination[0], attribute_combination[1]
        col3 = None if len(attribute_combination) == 2 else attribute_combination[2]
        mi_results.append(MutualInformation(datapath=dataset).calculate(col1, col2, col3))
        privbayes_results.append(
            ProxyMutualInformationPrivbayes(datapath=dataset).calculate(col1, col2, col3)
        )
        mst_results.append(
            ProxyMutualInformationNistContest(datapath=dataset).calculate(col1, col2, domain_path, col3)
        )
        tvd_results.append(
            ProxyMutualInformationTVD(datapath=dataset).calculate(col1, col2, col3)
        )
        lipschitz_results.append(
            ProxyMutualInformationLipschitz(datapath=dataset).calculate(col1, col2, col3)
        )
        print()


def plot_mi_proxies(plot_name):
    """
    Plots bar charts for Mutual Information and its four proxies:
    PrivBayes, MST, TVD, and Lipschitz-smoothed MI, each with a color legend.

    Parameters:
        plot_name (str): Filename to save the plot.
    """

    grouped_colors = {
        'mi': ['#b3cde3'] * 3 + ['#6497b1'] * 4 + ['#005b96'] * 3,
        'privbayes': ['#ccebc5'] * 3 + ['#5ab4ac'] * 4 + ['#01665e'] * 3,
        'mst': ['#fbb4ae'] * 3 + ['#f768a1'] * 4 + ['#ae017e'] * 3,
        'tvd': ['#decbe4'] * 3 + ['#b3a2c7'] * 4 + ['#6a51a3'] * 3,
        'lipschitz': ['#fde0dd'] * 3 + ['#fa9fb5'] * 4 + ['#c51b8a'] * 3
    }

    result_sets = [
        ("Regular Mutual Information", mi_results, grouped_colors['mi']),
        ("PrivBayes Proxy", privbayes_results, grouped_colors['privbayes']),
        ("MST Proxy", mst_results, grouped_colors['mst']),
        ("TVD Proxy", tvd_results, grouped_colors['tvd']),
        ("Lipschitz Proxy", lipschitz_results, grouped_colors['lipschitz'])
    ]

    legend_labels = ["Adult", "Stackoverflow 2024", "Compas"]
    legend_indices = [0, 3, 7]  # Index ranges for groups: 0–2, 3–6, 7–9

    x = np.arange(len(labels))
    fig, axes = plt.subplots(len(result_sets), 1, figsize=(14, 20), sharex=True)

    for ax, (title, results, colors) in zip(axes, result_sets):
        ax.bar(x, results, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(False)

        # Create legend from group colors
        legend_patches = [
            mpatches.Patch(color=colors[i], label=legend_labels[j])
            for j, i in enumerate(legend_indices)
        ]
        ax.legend(handles=legend_patches, loc="upper right")

    plt.xticks(ticks=x, labels=labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.show()


def unconditional_mi_proxies():
    global mi_results, privbayes_results, mst_results, tvd_results, lipschitz_results, labels, adult_attributes, \
        adult_domains, stackoverflow_attributes, stackoverflow_domains, compas_attributes, compas_domains
    mi_results = []
    privbayes_results = []
    mst_results = []
    tvd_results = []
    lipschitz_results = []
    labels = [
        "sex/income", "race/income", "education/education-num",
        "Country/EdLevel", "Country/DevType", "Country/SurveyLength", "Country/SOVisitFreq",
        "race/charge_desc", "race/score_text", "race/sex"
    ]
    # Adult dataset
    adult_attributes = [
        ("sex", "income>50K"),
        ("race", "income>50K"),
        ("education", "education-num")
    ]
    adult_domains = [
        "data/domains/adult/adult-domain-sex-income.json",
        "data/domains/adult/adult-domain-race-income.json",
        "data/domains/adult/adult-domain-education-education-num.json"
    ]
    calculate_mi_proxies("data/adult.csv", adult_attributes, adult_domains, "Adult")
    # Stackoverflow dataset
    stackoverflow_attributes = [
        ("Country", "EdLevel"),
        ("Country", "DevType"),
        ("Country", "SurveyLength"),
        ("Country", "SOVisitFreq")
    ]
    stackoverflow_domains = [
        "data/domains/stackoverflow/stackoverflow-domain-country-edlevel.json",
        "data/domains/stackoverflow/stackoverflow-domain-country-devtype.json",
        "data/domains/stackoverflow/stackoverflow-domain-country-surveylength.json",
        "data/domains/stackoverflow/stackoverflow-domain-country-sovisitfreq.json"
    ]
    calculate_mi_proxies("data/stackoverflow.csv", stackoverflow_attributes, stackoverflow_domains, "Stackoverflow")
    # Compas dataset
    compas_attributes = [
        ("race", "c_charge_desc"),
        ("race", "score_text"),
        ("race", "sex")
    ]
    compas_domains = [
        "data/domains/compas/compas-domain-race-c_charge_desc.json",
        "data/domains/compas/compas-domain-race-score_text.json",
        "data/domains/compas/compas-domain-race-sex.json"
    ]
    calculate_mi_proxies("data/compas.csv", compas_attributes, compas_domains, "Compas")
    plot_mi_proxies("plots/plot_mutual_information_colored_by_dataset.png")


def conditional_mi_proxies():
    global mi_results, privbayes_results, mst_results, tvd_results, lipschitz_results, labels, adult_attributes, \
        adult_domains, stackoverflow_attributes, stackoverflow_domains, compas_attributes, compas_domains
    mi_results = []
    privbayes_results = []
    mst_results = []
    tvd_results = []
    lipschitz_results = []
    labels = [
        "sex/income | education", "race/income | education", "education/education-num | sex",
        "Country/EdLevel | Age", "Country/DevType | Age", "Country/SurveyLength | Age", "Country/SOVisitFreq | Age",
        "race/charge_desc | age", "race/score_text | age", "race/sex | age"
    ]
    # Adult dataset
    adult_attributes = [
        ("sex", "income>50K", "education"),
        ("race", "income>50K", "education"),
        ("education", "education-num", "sex")
    ]
    adult_domains = [
        "data/domains/adult/adult-domain-sex-income-education.json",
        "data/domains/adult/adult-domain-race-income-education.json",
        "data/domains/adult/adult-domain-education-education-num-sex.json"
    ]
    calculate_mi_proxies("data/adult.csv", adult_attributes, adult_domains, "Adult")
    # Stackoverflow dataset
    stackoverflow_attributes = [
        ("Country", "EdLevel", "Age"),
        ("Country", "DevType", "Age"),
        ("Country", "SurveyLength", "Age"),
        ("Country", "SOVisitFreq", "Age")
    ]
    stackoverflow_domains = [
        "data/domains/stackoverflow/stackoverflow-domain-country-edlevel-age.json",
        "data/domains/stackoverflow/stackoverflow-domain-country-devtype-age.json",
        "data/domains/stackoverflow/stackoverflow-domain-country-surveylength-age.json",
        "data/domains/stackoverflow/stackoverflow-domain-country-sovisitfreq-age.json"
    ]
    calculate_mi_proxies("data/stackoverflow.csv", stackoverflow_attributes, stackoverflow_domains, "Stackoverflow")
    # Compas dataset
    compas_attributes = [
        ("race", "c_charge_desc", "age"),
        ("race", "score_text", "age"),
        ("race", "sex", "age")
    ]
    compas_domains = [
        "data/domains/compas/compas-domain-race-c_charge_desc-age.json",
        "data/domains/compas/compas-domain-race-score_text-age.json",
        "data/domains/compas/compas-domain-race-sex-age.json"
    ]
    calculate_mi_proxies("data/compas.csv", compas_attributes, compas_domains, "Compas")
    plot_mi_proxies("plots/plot_conditional_mutual_information_colored_by_dataset.png")


def tvd_with_laplace():
    epsilons = [1, 10, 100]
    num_runs = 9

    # Attribute combinations
    adult_attributes = [
        ("sex", "income>50K", "education"),
        ("race", "income>50K", "education"),
        ("education", "education-num", "sex")
    ]
    labels = [f"{a} ⊥ {b} | {c}" for a, b, c in adult_attributes]

    # Store results per epsilon
    results_by_epsilon = {eps: [] for eps in epsilons}

    for eps in epsilons:
        for _ in range(num_runs):
            run_results = []
            for a, b, c in adult_attributes:
                score = ProxyMutualInformationTVD(datapath="data/adult.csv").calculate(a, b, c, epsilon=eps)
                run_results.append(score)
            results_by_epsilon[eps].append(run_results)

    # Prepare plot
    fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharey=True)
    axes = axes.flatten()
    colors = {1: "#1f77b4", 10: "#2ca02c", 100: "#d62728"}  # blue, green, red

    for i in range(num_runs):
        ax = axes[i]
        for j, eps in enumerate(epsilons):
            values = np.array(results_by_epsilon[eps])[i]
            positions = np.arange(len(labels)) + (j - 1) * 0.2  # Offset for grouped bars
            ax.bar(positions, values, width=0.2, label=f"ε={eps}", color=colors[eps])
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Run {i + 1}")

    # Final subplot: mean ± std
    ax_summary = axes[-1]
    x = np.arange(len(labels))
    for j, eps in enumerate(epsilons):
        values = np.array(results_by_epsilon[eps])
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        positions = x + (j - 1) * 0.2
        ax_summary.bar(positions, mean, yerr=std, capsize=8, width=0.2,
                       label=f"ε={eps}", color=colors[eps])
    ax_summary.set_title("Mean ± Std Dev")
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(labels, rotation=30)
    ax_summary.set_ylim(0, 1.1)

    # Final touches
    handles, labels_legend = ax_summary.get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper right", fontsize=12)
    plt.suptitle("TVD Proxy Comparison Across Epsilons (9 Runs + Summary)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    plt.savefig("plots/plot_tvd_proxy_comparison_with_laplace.png")
    plt.show()


def plot_anomalous_treatment_count_pmi_repair():
    labels = [
        "sex/income | education", "race/income | education", "education/education-num | sex",
        "Country/EdLevel | Age", "Country/DevType | Age", "Country/SurveyLength | Age", "Country/SOVisitFreq | Age",
        "race/charge_desc | age", "race/score_text | age", "race/sex | age"
    ]

    # Define attribute triplets
    adult_attributes = [
        ("sex", "income>50K", "education"),
        ("race", "income>50K", "education"),
        ("education", "education-num", "sex")
    ]
    stackoverflow_attributes = [
        ("Country", "EdLevel", "Age"),
        ("Country", "DevType", "Age"),
        ("Country", "SurveyLength", "Age"),
        ("Country", "SOVisitFreq", "Age")
    ]
    compas_attributes = [
        ("race", "c_charge_desc", "age"),
        ("race", "score_text", "age"),
        ("race", "sex", "age")
    ]

    all_attributes = adult_attributes + stackoverflow_attributes + compas_attributes
    all_paths = ["data/adult.csv"] * len(adult_attributes) + \
                ["data/stackoverflow.csv"] * len(stackoverflow_attributes) + \
                ["data/compas.csv"] * len(compas_attributes)

    # Prepare result containers
    mutual_information_scores = []
    anomalous_counts = []
    pmi_scores = []
    repair_scores = []

    # Compute metric values
    for (s_col, o_col, a_col), path in zip(all_attributes, all_paths):
        mutual_information = MutualInformation(datapath=path).calculate(s_col, o_col, a_col)
        anomalous_count = AnomalousTreatmentCount(datapath=path).calculate(s_col, o_col, a_col)
        pmi_score = PMIThresholdDetector(datapath=path).calculate(s_col, o_col, a_col)
        repair_score = ProxyRepairMaxSat(datapath=path).calculate(s_col, o_col, a_col)

        mutual_information_scores.append(mutual_information)
        anomalous_counts.append(anomalous_count)
        pmi_scores.append(pmi_score)
        repair_scores.append(repair_score)

    # Grouped color coding for datasets
    grouped_colors = {
        'adult': ['#b3cde3'] * len(adult_attributes),
        'stackoverflow': ['#6497b1'] * len(stackoverflow_attributes),
        'compas': ['#005b96'] * len(compas_attributes)
    }
    colors = grouped_colors['adult'] + grouped_colors['stackoverflow'] + grouped_colors['compas']

    datasets = ["Adult", "Stackoverflow 2024", "Compas"]
    legend_indices = [0, len(adult_attributes), len(adult_attributes) + len(stackoverflow_attributes)]

    x = np.arange(len(labels))
    metrics = [
        ("Mutual Information", mutual_information_scores),
        ("Anomalous Treatment Count", anomalous_counts),
        ("PMI Threshold Count", pmi_scores),
        ("Proxy Repair MaxSAT", repair_scores)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    for ax, (title, values) in zip(axes, metrics):
        bars = ax.bar(x, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(False)

        # Annotate values on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        legend_patches = [
            mpatches.Patch(color=colors[i], label=datasets[j])
            for j, i in enumerate(legend_indices)
        ]
        ax.legend(handles=legend_patches, loc="upper right")

    plt.xticks(ticks=x, labels=labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/plot_anomalous_treatment_count_pmi_repair.png")
    plt.show()


def plot_proxies_comparison_1():
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

    import textwrap

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
            mi_scores.append(MutualInformation(datapath=path).calculate(s_col, o_col, a_col))
            priv_scores_orig.append(ProxyMutualInformationPrivbayes(datapath=path).calculate(
                s_col, o_col, a_col, add_offset=False))
            priv_scores_offset.append(ProxyMutualInformationPrivbayes(datapath=path).calculate(
                s_col, o_col, a_col, add_offset=True))
            tvd_scores.append(ProxyMutualInformationTVD(datapath=path).calculate(s_col, o_col, a_col))
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
    plt.savefig("plots/plot_proxies_4x3.png", dpi=220)
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_proxies_comparison_2():
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
            tvd_scores.append(ProxyMutualInformationTVD(datapath=path).calculate(s_col, o_col, a_col))
            auc_scores.append(ResidualAUCMeasure(datapath=path).calculate(s_col, o_col, a_col))
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

    plt.savefig("plots/plot_proxies_2x3.png", dpi=220)
    plt.show()




if __name__ == "__main__":
    # ----------------Unconditional MI Proxies----------------
    # unconditional_mi_proxies()

    # -----------------Conditional MI Proxies-----------------
    # conditional_mi_proxies()

    # ----------------------Private TVD-----------------------
    # tvd_with_laplace()

    # --------------------MaxSAT Repair-----------------------
    # maxsat_results = []
    # maxsat_results.append(ProxyRepairMaxSat(datapath='data/adult.csv').calculate("sex", "income>50K", "education"))
    # maxsat_results.append(ProxyRepairMaxSat(datapath='data/adult.csv').calculate("race", "income>50K", "education"))
    # maxsat_results.append(ProxyRepairMaxSat(datapath='data/adult.csv').calculate("education", "education-num", "sex"))
    # print()
    #
    # maxsat_results.append(ProxyRepairMaxSat(datapath="data/stackoverflow.csv").calculate("Country", "EdLevel", "Age"))
    # maxsat_results.append(ProxyRepairMaxSat(datapath="data/stackoverflow.csv").calculate("Country", "DevType", "Age"))
    # maxsat_results.append(ProxyRepairMaxSat(datapath="data/stackoverflow.csv").calculate("Country",
    # "SurveyLength", "Age"))
    # maxsat_results.append(ProxyRepairMaxSat(datapath="data/stackoverflow.csv").calculate("Country",
    # "SOVisitFreq", "Age"))
    # print()
    #
    # maxsat_results.append(ProxyRepairMaxSat(datapath="data/compas.csv").calculate("race", "c_charge_desc",
    # "age"))
    # maxsat_results.append(ProxyRepairMaxSat(datapath="data/compas.csv").calculate("race", "score_text", "age"))
    # maxsat_results.append(ProxyRepairMaxSat(datapath="data/compas.csv").calculate("race", "sex", "age"))
    #
    # maxsat_results.append(ProxyRepairMaxSat(datapath='data/toy_example.csv').calculate("A", "B", "C"))

    # ----------------Anomalous Treatment Count----------------
    # adult_attributes = [
    #     ("sex", "income>50K", "education"),
    #     ("race", "income>50K", "education"),
    #     ("education", "education-num", "sex")
    # ]
    # for s_col, o_col, a_col in adult_attributes:
    #     AnomalousTreatmentCount(datapath="data/adult.csv").calculate(s_col, o_col, a_col)

    # stackoverflow_attributes = [
    #     ("Country", "EdLevel", "Age"),
    #     ("Country", "DevType", "Age"),
    #     ("Country", "SurveyLength", "Age"),
    #     ("Country", "SOVisitFreq", "Age")
    # ]
    # for s_col, o_col, a_col in stackoverflow_attributes:
    #     AnomalousTreatmentCount(datapath="data/stackoverflow.csv").calculate(s_col, o_col, a_col)

    # -----------------PMI Threshold Detector------------------
    # adult_attributes = [
    #     ("sex", "income>50K", "education"),
    #     ("race", "income>50K", "education"),
    #     ("education", "education-num", "sex")
    # ]
    # for s_col, o_col, a_col in adult_attributes:
    #     PMIThresholdDetector(datapath="data/adult.csv").calculate(s_col, o_col, a_col)
    #
    # stackoverflow_attributes = [
    #     ("Country", "EdLevel", "Age"),
    #     ("Country", "DevType", "Age"),
    #     ("Country", "SurveyLength", "Age"),
    #     ("Country", "SOVisitFreq", "Age")
    # ]
    # for s_col, o_col, a_col in stackoverflow_attributes:
    #     PMIThresholdDetector(datapath="data/stackoverflow.csv").calculate(s_col, o_col, a_col)
    #
    # plot_anomalous_treatment_count_pmi_repair()

    # -----------------Shapley Values------------------
    plot_proxies_comparison_2()
