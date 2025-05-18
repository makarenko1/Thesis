import numpy as np
from matplotlib import pyplot as plt

from mutual_information import MutualInformation
from proxy_mutual_information_lipshitz import ProxyMutualInformationLipschitz
from proxy_mutual_information_nist_contest import ProxyMutualInformationNistContest
from proxy_mutual_information_privbayes import ProxyMutualInformationPrivbayes
from proxy_mutual_information_tvd import ProxyMutualInformationTVD
from repair_maxsat import ProxyRepairMaxSat


def calculate_mi_proxies(dataset, attributes, domain_paths, label):
    print(f"\n{label} dataset results:")
    for attribute_combination, domain_path in zip(attributes, domain_paths):
        col1, col2 = attribute_combination[0], attribute_combination[1]
        col3 = None if len(attribute_combination) == 2 else attribute_combination[2]
        mi_results.append(MutualInformation(dataset).calculate(col1, col2, col3))
        privbayes_results.append(
            ProxyMutualInformationPrivbayes(dataset).calculate(col1, col2, col3)
        )
        mst_results.append(
            ProxyMutualInformationNistContest(dataset).calculate(col1, col2, domain_path, col3)
        )
        tvd_results.append(
            ProxyMutualInformationTVD(dataset).calculate(col1, col2, col3)
        )
        lipschitz_results.append(
            ProxyMutualInformationLipschitz(dataset).calculate(col1, col2, col3)
        )
        print()


def plot_mi_proxies(plot_name):
    """
    Plots bar charts for Mutual Information and its four proxies:
    PrivBayes, MST, TVD, and Lipschitz-smoothed MI.

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

    x = np.arange(len(labels))
    fig, axes = plt.subplots(len(result_sets), 1, figsize=(14, 20), sharex=True)

    for ax, (title, results, colors) in zip(axes, result_sets):
        ax.bar(x, results, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(False)

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
    calculate_mi_proxies("data/compas-scores.csv", compas_attributes, compas_domains, "Compas")
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
    calculate_mi_proxies("data/compas-scores.csv", compas_attributes, compas_domains, "Compas")
    plot_mi_proxies("plots/plot_conditional_mutual_information_colored_by_dataset.png")


if __name__ == "__main__":
    # ----------------Unconditional MI Proxies----------------
    # unconditional_mi_proxies()

    # ----------------Conditional MI Proxies----------------
    # conditional_mi_proxies()

    # ----------------MaxSAT Repair----------------
    maxsat_results = []
    maxsat_results.append(ProxyRepairMaxSat('data/adult.csv').calculate("sex", "income>50K", "education"))
    maxsat_results.append(ProxyRepairMaxSat('data/adult.csv').calculate("race", "income>50K", "education"))
    maxsat_results.append(ProxyRepairMaxSat('data/adult.csv').calculate("education", "education-num", "sex"))

    # maxsat_results.append(ProxyRepairMaxSat("data/stackoverflow.csv").calculate("Country", "EdLevel", "Age"))
    # maxsat_results.append(ProxyRepairMaxSat("data/stackoverflow.csv").calculate("Country", "DevType", "Age"))
    # maxsat_results.append(ProxyRepairMaxSat("data/stackoverflow.csv").calculate("Country", "SurveyLength", "Age"))
    # maxsat_results.append(ProxyRepairMaxSat("data/stackoverflow.csv").calculate("Country", "SOVisitFreq", "Age"))

    print()
    maxsat_results.append(ProxyRepairMaxSat("data/compas-scores.csv").calculate("race", "c_charge_desc", "age"))
    maxsat_results.append(ProxyRepairMaxSat("data/compas-scores.csv").calculate("race", "score_text", "age"))
    maxsat_results.append(ProxyRepairMaxSat("data/compas-scores.csv").calculate("race", "sex", "age"))

    # maxsat_results.append(ProxyRepairMaxSat('data/toy_example.csv').calculate("A", "B", "C"))
