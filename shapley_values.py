import copy
import math
import random
import time
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from proxy_mutual_information_tvd import ProxyMutualInformationTVD


class LayeredShapleyValues:
    def __init__(self, datapath=None, data=None):
        """
        Initialize the estimator.

        Parameters
        ----------
        datapath : str, optional
            Path to a CSV dataset to load.
        data : pd.DataFrame, optional
            DataFrame already loaded in memory.

        Notes
        -----
        Exactly one of `datapath` or `data` must be provided.
        """
        if (datapath is None and data is None) or (datapath is not None and data is not None):
            raise Exception("Usage: Should pass either datapath or data itself")
        if datapath is not None:
            self.dataset = pd.read_csv(datapath)
        else:
            self.dataset = data

    @staticmethod
    def _remove_multiset(base_list, to_remove):
        """
            Remove a multiset of tuples from a base list while preserving order.

            Parameters
            ----------
            base_list : list[tuple]
                Full dataset represented as a list of tuples (S, O, A).
            to_remove : list[tuple]
                Tuples to remove (with multiplicities).

            Returns
            -------
            list[tuple]
                base_list with one occurrence of each element in `to_remove` removed.
            """
        need = Counter(to_remove)
        out = []
        for x in base_list:
            if need[x] > 0:
                need[x] -= 1
            else:
                out.append(x)
        return out

    @staticmethod
    def _get_relevant_tuples(X, k=1000):
        """
            Fast, TVD-free proxy to shortlist candidate tuples.

            Parameters
            ----------
            X : list[tuple[int,int,int]] or np.ndarray shape (N,3)
                Dataset encoded as integer triples (S, O, A).
            k : int
                Maximum number of candidates to return (top-k by proxy score).

            Returns
            -------
            list[tuple[int,int,int]]
                Up to `k` unique tuples ranked by a residual-based proxy score that
                upper-bounds the marginal impact of removing one record from a cell.

            Notes
            -----
            The proxy uses residuals R_{s,o,a} = |P(S,O|a) - P(S|a)P(O|a)| scaled by
            1/N_a to favor impactful cells in smaller groups.
            """
        X = np.array(X, dtype=np.int64)
        # X is an int ndarray [N, 3] with columns (S,O,A)
        S, O, A = X[:, 0], X[:, 1], X[:, 2]
        nS, nO, nA = S.max() + 1, O.max() + 1, A.max() + 1
        N = len(X)

        flat = A * (nS * nO) + S * nO + O
        C = np.bincount(flat, minlength=nA * nS * nO).reshape(nA, nS, nO).astype(float)
        Na = C.sum(axis=(1, 2))  # [nA]
        mask = Na > 0

        Pso = np.zeros_like(C)
        Pso[mask] = C[mask] / Na[mask, None, None]
        Ps = Pso.sum(axis=2, keepdims=True)  # [nA,nS,1]
        Po = Pso.sum(axis=1, keepdims=True)  # [nA,1,nO]
        R = np.abs(Pso - Ps * Po)  # residuals

        # proxy score per cell; avoid div0 for empty groups
        proxy = np.zeros_like(C)
        proxy[mask] = R[mask] / Na[mask, None, None]

        # map back to unique tuples and scores
        tuples, scores = [], []
        for a in range(nA):
            if Na[a] == 0: continue
            idx = np.argwhere(C[a] > 0)  # cells present in data
            for s, o in idx:
                tuples.append((int(s), int(o), int(a)))
                scores.append(float(proxy[a, s, o]))
        scores = np.array(scores)

        K = min(k, len(tuples))
        order = np.argsort(-scores)
        return [tuples[i] for i in order[:K]]

    def _calculate_for_one_tuple(self, D, t, full_tvd, s_col, o_col, a_col=None, alpha=10, beta=10, n=1000):
        """
            Estimate the Shapley value for one representative tuple via the
            Layered-Shapley Monte Carlo approximation.

            Parameters
            ----------
            D : list[tuple]
                Dataset as a list of integer tuples (S, O, A).
            t : tuple
                The representative tuple whose marginal contribution is estimated.
            full_tvd : float
                TVD proxy computed on the full dataset (used by the current formula).
            s_col, o_col, a_col : str
                Column names for sensitive, outcome, and admissible attributes.
            alpha : float
                Precision control parameter used in the per-level sample size `m_k`.
            beta : float
                Confidence control parameter used in the per-level sample size `m_k`.
            n : int
                Number of “levels” (coalition sizes) to sample.

            Returns
            -------
            float
                Estimated Shapley value for tuple `t`.
            """
        shapley_estimate_for_all_levels = 0
        for _ in range(1, n):
            k = random.randrange(1, len(D))  # randomize the level size
            shapley_estimate_for_kth_level = 0
            m_k = math.ceil(((2 ** 2) / (2 * (alpha ** 2) * (k ** 2))) *
                            math.log(2 * n / beta, math.e))

            for _ in range(m_k):
                S = random.sample(D, k)
                D_minus_S = self._remove_multiset(D, S)
                D_minus_S_minus_t = D_minus_S.copy()
                try:
                    D_minus_S_minus_t.remove(t)
                except ValueError:
                    pass
                tvd_S_and_t = ProxyMutualInformationTVD(data=pd.DataFrame(D_minus_S, columns=[
                    s_col, o_col, a_col])).calculate(s_col, o_col, a_col)
                tvd_S = ProxyMutualInformationTVD(data=pd.DataFrame(D_minus_S_minus_t, columns=[
                    s_col, o_col, a_col])).calculate(s_col, o_col, a_col)
                shapley_estimate_for_kth_level += ((1 / m_k) * abs(abs(full_tvd - tvd_S_and_t) - abs(full_tvd - tvd_S)))

            shapley_estimate_for_all_levels += (1 / n) * shapley_estimate_for_kth_level
        return shapley_estimate_for_all_levels

    @staticmethod
    def _get_flagged_tuples(D, avg_shapley_value_per_tuple):
        """
        Select tuples that comprise the upper half of the total Shapley “mass”.

        Parameters
        ----------
        D : list[tuple]
            Dataset as a list of tuples (S, O, A).
        avg_shapley_value_per_tuple : dict[tuple, float]
            Mapping from tuple -> estimated Shapley value.

        Returns
        -------
        int
            The number of tuples needed (from the top, after sorting by value
            descending) to accumulate at least 50% of the total Shapley sum.

        Notes
        -----
        This is a robust cutoff for hyperbola-like tails.
        """
        items = list(avg_shapley_value_per_tuple.items())
        # sort by value desc
        items.sort(key=lambda kv: kv[1], reverse=True)
        vals = np.array([v for _, v in items], dtype=float)

        total = vals.sum()
        if total <= 0:
            return 0  # nothing to select

        cum = np.cumsum(vals)
        k = int(np.searchsorted(cum, 0.5 * total))  # index of half-mass cutoff

        num_flagged_tuples = 0
        for i in range(k + 1):
            num_flagged_tuples += D.count(items[i][0])
        return num_flagged_tuples

    def calculate(self, s_col, o_col, a_col=None, alpha=1, beta=2.5 , n=10, data=None):
        """
        Calculates Shapley-based unfairness score according to the Layered Shapley Algorithm.

        Parameters:
        -----------
        s_col : str
            Sensitive attribute name (S).
        o_col : str
            Outcome attribute name (O).
        a_col : str
            Admissible attribute name (A).
        alpha : float
            Controls the precision of the Shapley value estimate.
        beta : float
            Controls the confidence level.

        Returns:
        --------
        int
            Number of tuples whose average Shapley value exceeds the threshold.
        """
        cols = [s_col, o_col]
        if a_col is not None:
            cols += [a_col]

        if data is None:
            self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)
            self.dataset.dropna(inplace=True, subset=cols)
            for col in cols:
                self.dataset[col] = LabelEncoder().fit_transform(self.dataset[col])
        self.dataset = self.dataset[cols]

        D = self.dataset[cols].to_numpy().tolist() if data is None else data
        for i in range(len(D)):
            D[i] = tuple(D[i])

        start_time = time.time()  # Record start time
        full_tvd = ProxyMutualInformationTVD(data=pd.DataFrame(D, columns=[s_col, o_col, a_col])).calculate(
            s_col, o_col, a_col)
        candidates = self._get_relevant_tuples(D)
        candidates = list(set(candidates))
        avg_shapley_value_per_tuple = defaultdict(lambda: 0)

        for t in candidates:
            avg_shapley_value_per_tuple[t] = self._calculate_for_one_tuple(
                D, t, full_tvd, s_col, o_col, a_col=a_col, alpha=alpha, beta=beta, n=n)
        num_tuples_with_shapley_above_threshold = self._get_flagged_tuples(D, avg_shapley_value_per_tuple)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Layered Shapley: {num_tuples_with_shapley_above_threshold} tuples flagged above threshold "
              f"with alpha {alpha} and beta {beta}. Calculation took {elapsed_time:.3f} seconds.")
        return num_tuples_with_shapley_above_threshold

    # def _calculate_with_smart_threshold_helper(self, D, t, full_tvd, s_col, o_col, a_col, alpha=10, beta=10, n=1000,
    #                                            iterations=1):
    #     """
    #     Calculates whether a tuple t is statistically significant using permutation testing.
    #     Corresponds to the permutation test shown in the provided pseudocode.
    #
    #     Parameters:
    #     -----------
    #     t : pd.Series
    #         The tuple for which we are checking significance.
    #     s_col : str
    #         Name of the sensitive attribute (S).
    #     o_col : str
    #         Name of the outcome attribute (O).
    #     a_col : str
    #         Name of the admissible attribute (A).
    #     iterations : int
    #         Number of random permutations to perform (m in the pseudocode).
    #
    #     Returns:
    #     --------
    #     bool
    #         True if p-value < 0.05, otherwise False.
    #     """
    #
    #     # Step 1: Subset the dataset to only rows where A == t[A]
    #     # a_value = t[2]
    #     # D_a = []
    #     # for i in range(len(D)):
    #     #     if D[i][2] == a_value:
    #     #         D_a.append(D[i])
    #
    #     # Step 2: Compute the actual test statistic on the original data
    #     true_score = self._calculate_for_one_tuple(D, t, full_tvd, s_col, o_col, a_col=a_col, alpha=alpha, beta=beta,
    #                                                n=n)
    #
    #     # Step 3–5: Perform random permutations and compute statistics
    #     count = 0
    #     for _ in range(iterations):
    #         D_i = self.dataset.copy()
    #         D_i[s_col] = np.random.permutation(D_i[s_col].values)
    #         D_i[o_col] = np.random.permutation(D_i[o_col].values)
    #         D_i = D_i.to_numpy().tolist()
    #         for i in range(len(D_i)):
    #             D_i[i] = tuple(D_i[i])
    #         D_i = list(set(D_i))
    #
    #         permuted_score = self._calculate_for_one_tuple(D_i, t, full_tvd, s_col, o_col, a_col=a_col, alpha=alpha,
    #                                                        beta=beta, n=n)
    #         if permuted_score >= true_score:
    #             count += D.count(t)
    #
    #     # Step 6: Compute p-value
    #     p = (1 + count) / (1 + iterations)
    #
    #     # Step 7–9: Return True if statistically significant
    #     return p < 0.05
    #
    # def calculate_with_smart_threshold(self, s_col, o_col, a_col, data=None):
    #     """
    #     Computes the number of tuples that are statistically significant based on Shapley value
    #     using permutation-based p-value thresholding.
    #
    #     Parameters:
    #     -----------
    #     s_col : str
    #         Sensitive attribute.
    #     o_col : str
    #         Outcome attribute.
    #     a_col : str
    #         Admissible attribute.
    #     data : pd.DataFrame (optional)
    #         Optional dataset override.
    #
    #     Returns:
    #     --------
    #     int
    #         Count of tuples considered statistically significant.
    #     """
    #     cols = [s_col, o_col]
    #     if a_col is not None:
    #         cols += [a_col]
    #
    #     if data is None:
    #         self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)
    #         self.dataset.dropna(inplace=True, subset=cols)
    #         for col in cols:
    #             self.dataset[col] = LabelEncoder().fit_transform(self.dataset[col])
    #     self.dataset = self.dataset[cols]
    #
    #     D = self.dataset[cols].to_numpy().tolist() if data is None else data
    #     for i in range(len(D)):
    #         D[i] = tuple(D[i])
    #
    #     start_time = time.time()  # Record start time
    #
    #     D_shortened = list(set(D))
    #     full_tvd = ProxyMutualInformationTVD(data=pd.DataFrame(D, columns=[s_col, o_col, a_col])).calculate(
    #         s_col, o_col, a_col)
    #
    #     num_tuples_with_shapley_above_threshold = 0
    #     for t in D_shortened:
    #         if self._calculate_with_smart_threshold_helper(D, t, full_tvd, s_col, o_col, a_col):
    #             num_tuples_with_shapley_above_threshold += D.count(t)
    #
    #     end_time = time.time()  # Record end time
    #     elapsed_time = end_time - start_time
    #     print(f"Layered Shapley: {num_tuples_with_shapley_above_threshold} tuples flagged. "
    #           f"Calculation took {elapsed_time:.3f} seconds.")
    #     return num_tuples_with_shapley_above_threshold


class ShapleyValues:
    def __init__(self, datapath=None, data=None):
        """
        Initializes the Shapley value estimator with a dataset path.

        Parameters:
        -----------
        datapath : str
            Path to the CSV dataset file.
        """
        if (datapath is None and data is None) or (datapath is not None and data is not None):
            raise Exception("Usage: Should pass either datapath or data itself")
        if datapath is not None:
            self.dataset = pd.read_csv(datapath)
        else:
            self.dataset = data

    def calculate(self, s_col, o_col, a_col=None, threshold=0.01, sample_size=100, times=100, data=None):
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

        D = self.dataset[cols].to_numpy().tolist() if data is None else data
        for i in range(len(D)):
            D[i] = tuple(D[i])
        D_shortened = list(set(D))

        start_time = time.time()  # Record start time

        avg_shapley_values_per_tuple = defaultdict(lambda: 0)
        for _ in range(times):
            for t in D_shortened:
                D_tag = copy.deepcopy(D)
                D_tag.remove(t)

                S = random.sample(population=D_tag, k=sample_size)
                tvd_S_and_t = ProxyMutualInformationTVD(data=pd.DataFrame(S + [t], columns=[
                    s_col, o_col, a_col])).calculate(s_col, o_col, a_col)
                tvd_S = ProxyMutualInformationTVD(data=pd.DataFrame(S, columns=[
                    s_col, o_col, a_col])).calculate(s_col, o_col, a_col)

                shapley_value = abs(tvd_S_and_t - tvd_S)
                avg_shapley_values_per_tuple[t] += (shapley_value / times)

        num_tuples_with_shapley_above_threshold = 0
        for t in D_shortened:
            if avg_shapley_values_per_tuple[t] >= threshold:
                num_tuples_with_shapley_above_threshold += D.count(t)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Shapley: {num_tuples_with_shapley_above_threshold} tuples flagged above threshold {threshold} with "
              f"sample size {sample_size} and num iterations {times}. Calculation took {elapsed_time:.3f} seconds.")
        return num_tuples_with_shapley_above_threshold
