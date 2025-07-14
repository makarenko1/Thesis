import copy
import math
import random
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from proxy_mutual_information_tvd import ProxyMutualInformationTVD


class ShapleyValues:
    def __init__(self, datapath):
        """
        Initializes the ShapleyTVDUnfairness object.

        Parameters:
        -----------
        datapath : str
            Path to the CSV dataset file.
        tvd_calculator : object
            An instance of ProxyMutualInformationTVD with a 'calculate' method.
        """
        self.dataset = pd.read_csv(datapath)

    def calculate(self, s_col, o_col, a_col=None, threshold=0.01, sample_size=10, times=10):
        """
        Calculates Shapley-based unfairness score.

        Parameters:
        -----------
        s_col : str
            Sensitive attribute name (S).
        o_col : str
            Outcome attribute name (O).
        a_col : str
            Admissible attribute name (A).
        threshold : float
            Threshold for a tuple's average Shapley value.
        times : int
            Number of permutations for Shapley approximation.

        Returns:
        --------
        int
            Number of tuples whose average Shapley value exceeds the threshold.
        """
        self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)
        cols = [s_col, o_col]
        if a_col is not None:
            cols += [a_col]
        self.dataset.dropna(inplace=True, subset=cols)
        for col in cols:
            self.dataset[col] = LabelEncoder().fit_transform(self.dataset[col])

        D = self.dataset[cols].to_numpy().tolist()
        avg_shapley_values_per_tuple = defaultdict(lambda: 0)

        for _ in range(times):
            for t in D:
                D_tag = copy.deepcopy(D)
                D_tag.remove(t)

                S = random.sample(population=D_tag, k=sample_size)
                tvd_S_and_t = ProxyMutualInformationTVD(data=pd.DataFrame(S + [t], columns=[
                    s_col, o_col, a_col])).calculate(s_col, o_col, a_col)
                tvd_S = ProxyMutualInformationTVD(data=pd.DataFrame(S, columns=[
                    s_col, o_col, a_col])).calculate(s_col, o_col, a_col)

                shapley_value = ((math.factorial(len(S)) * math.factorial(len(D) - len(S) - 1)) /
                                 math.factorial(len(D))) * abs(tvd_S_and_t - tvd_S)
                avg_shapley_values_per_tuple[tuple(t)] += (shapley_value / times)

        num_tuples_with_shapley_above_threshold = 0
        for t in D:
            if avg_shapley_values_per_tuple[tuple(t)] >= threshold:
                num_tuples_with_shapley_above_threshold += 1

        print(f"Shapley: {num_tuples_with_shapley_above_threshold} tuples flagged above threshold {threshold} with "
              f"sample size {sample_size} and num iterations {times}")
        return num_tuples_with_shapley_above_threshold
