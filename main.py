import numpy as np
from matplotlib import pyplot as plt

from proxy_mutual_information_privbayes import ProxyMutualInformationPrivbayes
from proxy_mutual_information_nist_contest import ProxyMutualInformationNistContest
from mutual_information import MutualInformation

if __name__ == "__main__":
    mi_results = []
    privbayes_results = []
    mst_results = []

    print("Adult dataset results:")
    mi_results.append(MutualInformation('data/adult.csv').calculate("sex", "income>50K"))
    privbayes_results.append(ProxyMutualInformationPrivbayes('data/adult.csv').calculate("sex", "income>50K"))
    mst_results.append(ProxyMutualInformationNistContest('data/adult.csv').calculate(
        "sex", "income>50K", 'data/adult-domain-sex-income.json'))

    print()

    mi_results.append(MutualInformation('data/adult.csv').calculate("race", "income>50K"))
    privbayes_results.append(ProxyMutualInformationPrivbayes('data/adult.csv').calculate("race", "income>50K"))
    mst_results.append(ProxyMutualInformationNistContest('data/adult.csv').calculate(
        "race", "income>50K", 'data/adult-domain-race-income.json'))

    print()

    mi_results.append(MutualInformation('data/adult.csv').calculate("education", "education-num"))
    privbayes_results.append(ProxyMutualInformationPrivbayes('data/adult.csv').calculate("education", "education-num"))
    mst_results.append(ProxyMutualInformationNistContest('data/adult.csv').calculate(
        "education", "education-num", 'data/adult-domain-race-income.json'))

    print()

    print("\nStackoverflow dataset results:")
    mi_results.append(MutualInformation('data/stackoverflow.csv').calculate("Country", "EdLevel"))
    privbayes_results.append(ProxyMutualInformationPrivbayes('data/stackoverflow.csv').calculate("Country", "EdLevel"))
    mst_results.append(ProxyMutualInformationNistContest('data/stackoverflow.csv').calculate(
        "Country", "EdLevel", 'data/stackoverflow-domain-country-edlevel.json'))

    print()

    mi_results.append(MutualInformation('data/stackoverflow.csv').calculate("Country", "DevType"))
    privbayes_results.append(ProxyMutualInformationPrivbayes('data/stackoverflow.csv').calculate("Country", "DevType"))
    mst_results.append(ProxyMutualInformationNistContest('data/stackoverflow.csv').calculate(
        "Country", "DevType", 'data/stackoverflow-domain-country-devtype.json'))

    print()

    mi_results.append(MutualInformation('data/stackoverflow.csv').calculate("Country", "SurveyLength"))
    privbayes_results.append(ProxyMutualInformationPrivbayes('data/stackoverflow.csv').calculate(
        "Country", "SurveyLength"))
    mst_results.append(ProxyMutualInformationNistContest('data/stackoverflow.csv').calculate(
        "Country", "SurveyLength", 'data/stackoverflow-domain-country-surveylength.json'))

    print()

    mi_results.append(MutualInformation('data/stackoverflow.csv').calculate("Country", "SOVisitFreq"))
    privbayes_results.append(ProxyMutualInformationPrivbayes('data/stackoverflow.csv').calculate(
        "Country", "SOVisitFreq"))
    mst_results.append(ProxyMutualInformationNistContest('data/stackoverflow.csv').calculate(
        "Country", "SOVisitFreq", 'data/stackoverflow-domain-country-sovisitfreq.json'))

    print()

    print("\nCompas dataset results:")
    mi_results.append(MutualInformation('data/compas-scores.csv').calculate("race", "c_charge_desc"))
    privbayes_results.append(ProxyMutualInformationPrivbayes('data/compas-scores.csv').calculate(
        "race", "c_charge_desc"))
    mst_results.append(ProxyMutualInformationNistContest('data/compas-scores.csv').calculate(
        "race", "c_charge_desc", 'data/compas-domain-race-c_charge_desc.json'))

    print()

    mi_results.append(MutualInformation('data/compas-scores.csv').calculate("race", "score_text"))
    privbayes_results.append(ProxyMutualInformationPrivbayes('data/compas-scores.csv').calculate("race", "score_text"))
    mst_results.append(ProxyMutualInformationNistContest('data/compas-scores.csv').calculate(
        "race", "score_text", 'data/compas-domain-race-score_text.json'))

    print()

    mi_results.append(MutualInformation('data/compas-scores.csv').calculate("race", "sex"))
    privbayes_results.append(ProxyMutualInformationPrivbayes('data/compas-scores.csv').calculate("race", "sex"))
    mst_results.append(ProxyMutualInformationNistContest('data/compas-scores.csv').calculate(
        "race", "sex", 'data/compas-domain-race-sex.json'))

    # Labels (x-axis)
    labels = [
        "sex/income", "race/income", "education/education-num",
        "Country/EdLevel", "Country/DevType", "Country/SurveyLength", "Country/SOVisitFreq",
        "race/charge_desc", "race/score_text", "race/sex"
    ]

    # Define colors by dataset
    mi_colors = ['#b3cde3'] * 3 + ['#6497b1'] * 4 + ['#005b96'] * 3
    priv_colors = ['#ccebc5'] * 3 + ['#5ab4ac'] * 4 + ['#01665e'] * 3
    mst_colors = ['#fbb4ae'] * 3 + ['#f768a1'] * 4 + ['#ae017e'] * 3

    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot MI
    axes[0].bar(x, mi_results, color=mi_colors)
    axes[0].set_title("Regular Mutual Information")
    axes[0].set_ylabel("Score")
    axes[0].grid(False)

    # Plot PrivBayes
    axes[1].bar(x, privbayes_results, color=priv_colors)
    axes[1].set_title("PrivBayes Proxy")
    axes[1].set_ylabel("Score")
    axes[1].grid(False)

    # Plot MST Proxy
    axes[2].bar(x, mst_results, color=mst_colors)
    axes[2].set_title("MST Proxy")
    axes[2].set_ylabel("Score")
    axes[2].grid(False)

    # Shared x-axis
    plt.xticks(ticks=x, labels=labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("mutual_information_colored_by_dataset.png")
    plt.show()
