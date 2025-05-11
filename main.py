import numpy as np
from matplotlib import pyplot as plt

from proxy_mutual_information_nist_contest_conditional import ProxyMutualInformationNistContestConditional
from proxy_mutual_information_privbayes_conditional import ProxyMutualInformationPrivbayesConditional
from proxy_mutual_information_privbayes_unconditional import ProxyMutualInformationPrivbayesUnconditional
from proxy_mutual_information_nist_contest_unconditional import ProxyMutualInformationNistContestUnconditional
from mutual_information import MutualInformation
from proxy_repair_maxsat import ProxyRepairMaxSat


def calculate_mi_proxies(dataset, pairs, domain_paths, label):
    print(f"\n{label} dataset results:")
    for (col1, col2), domain_path in zip(pairs, domain_paths):
        mi_results.append(MutualInformation(dataset).calculate(col1, col2))
        privbayes_results.append(
            ProxyMutualInformationPrivbayesUnconditional(dataset).calculate(col1, col2)
        )
        mst_results.append(
            ProxyMutualInformationNistContestUnconditional(dataset).calculate(col1, col2, domain_path)
        )
        print()


def calculate_mi_proxies_conditional(dataset, pairs, domain_paths, label):
    print(f"\n{label} dataset results:")
    for (col1, col2, col3), domain_path in zip(pairs, domain_paths):
        mi_results.append(MutualInformation(dataset).calculate(col1, col2, col3))
        privbayes_results.append(
            ProxyMutualInformationPrivbayesConditional(dataset).calculate(col1, col2, col3)
        )
        mst_results.append(
            ProxyMutualInformationNistContestConditional(dataset).calculate(col1, col2, col3, domain_path)
        )
        print()


def plot_mi_proxies(mi_results, privbayes_results, mst_results, plot_name):
    """
    Plots bar charts for Mutual Information and its two proxies: PrivBayes and MST.

    Parameters:
        mi_results (list of float): Scores from mutual information.
        privbayes_results (list of float): Scores from PrivBayes proxy.
        mst_results (list of float): Scores from MST proxy.
        plot_name (str): Filename to save the plot.
    """

    labels = [
        "sex/income", "race/income", "education/education-num",
        "Country/EdLevel", "Country/DevType", "Country/SurveyLength", "Country/SOVisitFreq",
        "race/charge_desc", "race/score_text", "race/sex"
    ]

    grouped_colors = {
        'mi': ['#b3cde3'] * 3 + ['#6497b1'] * 4 + ['#005b96'] * 3,
        'privbayes': ['#ccebc5'] * 3 + ['#5ab4ac'] * 4 + ['#01665e'] * 3,
        'mst': ['#fbb4ae'] * 3 + ['#f768a1'] * 4 + ['#ae017e'] * 3
    }

    result_sets = [
        ("Regular Mutual Information", mi_results, grouped_colors['mi']),
        ("PrivBayes Proxy", privbayes_results, grouped_colors['privbayes']),
        ("MST Proxy", mst_results, grouped_colors['mst'])
    ]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(len(result_sets), 1, figsize=(14, 12), sharex=True)

    for ax, (title, results, colors) in zip(axes, result_sets):
        ax.bar(x, results, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(False)

    plt.xticks(ticks=x, labels=labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.show()


if __name__ == "__main__":
    mi_results = []
    privbayes_results = []
    mst_results = []

    # Adult dataset
    adult_attributes = [
        ("sex", "income>50K"),
        ("race", "income>50K"),
        ("education", "education-num")
    ]

    adult_domains = [
        "data/adult-domain-sex-income.json",
        "data/adult-domain-race-income.json",
        "data/adult-domain-education-education-num.json"
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
        "data/stackoverflow-domain-country-edlevel.json",
        "data/stackoverflow-domain-country-devtype.json",
        "data/stackoverflow-domain-country-surveylength.json",
        "data/stackoverflow-domain-country-sovisitfreq.json"
    ]

    calculate_mi_proxies("data/stackoverflow.csv", stackoverflow_attributes, stackoverflow_domains, "Stackoverflow")

    # Compas dataset
    compas_attributes = [
        ("race", "c_charge_desc"),
        ("race", "score_text"),
        ("race", "sex")
    ]

    compas_domains = [
        "data/compas-domain-race-c_charge_desc.json",
        "data/compas-domain-race-score_text.json",
        "data/compas-domain-race-sex.json"
    ]

    calculate_mi_proxies("data/compas-scores.csv", compas_attributes, compas_domains, "Compas")
    plot_mi_proxies(mi_results, privbayes_results, mst_results, "mutual_information_colored_by_dataset.png")

    mi_results = []
    privbayes_results = []
    mst_results = []

    # Adult dataset
    adult_attributes = [
        ("sex", "income>50K", "education"),
        ("race", "income>50K", "education"),
        ("education", "education-num", "sex")
    ]

    adult_domains = [
        "data/adult-domain-sex-income-education.json",
        "data/adult-domain-race-income-education.json",
        "data/adult-domain-education-education-num-sex.json"
    ]

    calculate_mi_proxies_conditional("data/adult.csv", adult_attributes, adult_domains, "Adult")
    plot_mi_proxies(mi_results, privbayes_results, mst_results, "conditional_mutual_information_colored_by_dataset.png")

    maxsat_results = []
    maxsat_results.append(ProxyRepairMaxSat('data/adult.csv').calculate("sex", "income>50K", "education"))
    maxsat_results.append(ProxyRepairMaxSat('data/adult.csv').calculate("race", "income>50K", "education"))
