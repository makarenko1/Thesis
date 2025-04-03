from proxy_mutual_information_privbayes import ProxyMutualInformationPrivbayes
from mutual_information import MutualInformation

if __name__ == "__main__":
    MutualInformation().calculate("sex", "income")
    ProxyMutualInformationPrivbayes().calculate("sex", "income")

    MutualInformation().calculate("race", "income")
    ProxyMutualInformationPrivbayes().calculate("race", "income")
