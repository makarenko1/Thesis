from private_mutual_information import PrivateMutualInformation
from non_private_mutual_information import NonPrivateMutualInformation

if __name__ == "__main__":
    NonPrivateMutualInformation().calculate("sex", "income")
    PrivateMutualInformation().calculate("sex", "income")

    NonPrivateMutualInformation().calculate("race", "income")
    PrivateMutualInformation().calculate("race", "income")
