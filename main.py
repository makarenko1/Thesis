from proxy_mutual_information_privbayes import ProxyMutualInformationPrivbayes
from proxy_mutual_information_nist_contest import ProxyMutualInformationNistContest
from mutual_information import MutualInformation

if __name__ == "__main__":
    MutualInformation('data/adult.csv').calculate("sex", "income>50K")
    ProxyMutualInformationPrivbayes('data/adult.csv').calculate("sex", "income>50K")
    ProxyMutualInformationNistContest('data/adult.csv').calculate("sex", "income>50K",
                                                                  'data/adult-domain-sex-income.json')

    print()

    MutualInformation('data/adult.csv').calculate("race", "income>50K")
    ProxyMutualInformationPrivbayes('data/adult.csv').calculate("race", "income>50K")
    ProxyMutualInformationNistContest('data/adult.csv').calculate("race", "income>50K",
                                                                  'data/adult-domain-race-income.json')

    print()

    MutualInformation('data/stackoverflow.csv').calculate("Country", "EdLevel")
    ProxyMutualInformationPrivbayes('data/stackoverflow.csv').calculate("Country", "EdLevel")
    ProxyMutualInformationNistContest('data/stackoverflow.csv').calculate(
        "Country", "EdLevel", 'data/stackoverflow-domain-country-edlevel.json')

    print()

    MutualInformation('data/stackoverflow.csv').calculate("Country", "DevType")
    ProxyMutualInformationPrivbayes('data/stackoverflow.csv').calculate("Country", "DevType")
    ProxyMutualInformationNistContest('data/stackoverflow.csv').calculate(
        "Country", "DevType", 'data/stackoverflow-domain-country-devtype.json')

    print()

    MutualInformation('data/stackoverflow.csv').calculate("Country", "SurveyLength")
    ProxyMutualInformationPrivbayes('data/stackoverflow.csv').calculate("Country", "SurveyLength")
    ProxyMutualInformationNistContest('data/stackoverflow.csv').calculate(
        "Country", "SurveyLength", 'data/stackoverflow-domain-country-surveylength.json')

    print()

    MutualInformation('data/stackoverflow.csv').calculate("Country", "SOVisitFreq")
    ProxyMutualInformationPrivbayes('data/stackoverflow.csv').calculate("Country", "SOVisitFreq")
    ProxyMutualInformationNistContest('data/stackoverflow.csv').calculate(
        "Country", "SOVisitFreq", 'data/stackoverflow-domain-country-sovisitfreq.json')
