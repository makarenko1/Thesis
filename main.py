from proxy_mutual_information_privbayes import ProxyMutualInformationPrivbayes
from proxy_mutual_information_nist_contest import ProxyMutualInformationNistContest
from mutual_information import MutualInformation

if __name__ == "__main__":
    MutualInformation().calculate("sex", "income>50K")
    ProxyMutualInformationPrivbayes().calculate("sex", "income>50K")
    ProxyMutualInformationNistContest().calculate("sex", "income>50K", 'data/adult-domain-sex-income.json')

    print()

    MutualInformation().calculate("race", "income>50K")
    ProxyMutualInformationPrivbayes().calculate("race", "income>50K")
    ProxyMutualInformationNistContest().calculate("race", "income>50K", 'data/adult-domain-race-income.json')
