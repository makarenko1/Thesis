import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class TupleContribution:
    """
    Implementation of the Tuple Contribution metric
    for the case where the protected set, response set, and admissible set
    each contain at most one attribute (a single column).

    Computes unsigned marginal differences (MD) for given fairness criteria,
    optionally adds Laplace noise for differential privacy, and returns
    the sum of top-k MD values across criteria.
    """

    def __init__(self, datapath=None, data=None):
        if (datapath is None and data is None) or (datapath is not None and data is not None):
            raise ValueError("Usage: pass exactly one of datapath or data")
        self.dataset = pd.read_csv(datapath) if datapath is not None else data.copy()

    def calculate(self, fairness_criteria, k=250, epsilon=None, encode_and_clean=False):
        """
        Compute the top-k unsigned marginal differences for each fairness criterion.

        Each criterion consists of two or three column names:
            (protected, response) or (protected, response, admissible).

        If epsilon is given, Laplace noise is added for privacy.
        Prints a short summary and returns the resulting contribution value.
        """
        start_time = time.time()

        contribution = 0.0
        min_a_count = None

        for criterion in fairness_criteria:
            if len(criterion) not in [2, 3]:
                raise ValueError("Invalid input")

            protected_col, response_col, admissible_col = (criterion[0], criterion[1],
                                                           None if len(criterion) == 2 else criterion[2])
            cols = [protected_col, response_col] + ([admissible_col] if admissible_col is not None else [])
            if encode_and_clean:
                df = self._encode_and_clean(self.dataset, cols)
            else:
                df = self.dataset

            if admissible_col is not None:
                temp = int(df[admissible_col].value_counts().min())
                min_a_count = min(min_a_count, temp) if min_a_count is not None else temp

            if k is None:
                k = len(df)

            if admissible_col is None:
                values = self._calculate_unconditional_helper(df[protected_col].to_numpy(), df[response_col].to_numpy())
            else:
                values = self._calculate_conditional_helper(df[protected_col].to_numpy(), df[response_col].to_numpy(),
                                                            df[admissible_col].to_numpy())

            top_k = sorted(values, reverse=True)[:min(k, len(values))]
            contribution += float(np.sum(top_k))

        if epsilon is not None:
            if min_a_count is not None and min_a_count > 1:
                sensitivity = len(fairness_criteria) * ((3 * k / (min_a_count - 1)) + 2)
            else:
                n = len(self.dataset)
                sensitivity = len(fairness_criteria) * ((3 * k / n) + 2)
            contribution = contribution + np.random.laplace(loc=0.0, scale=sensitivity / float(epsilon))

        elapsed_time = time.time() - start_time
        print(f"Tuple Contribution for fairness criteria {fairness_criteria}: {contribution:.4f} with data size: "
              f"{len(self.dataset)} and epsilon: {epsilon if epsilon is not None else 'infinity'}. Calculation took "
              f"{elapsed_time:.3f} seconds.")
        return contribution

    @staticmethod
    def _encode_and_clean(df, cols):
        """
        Cleans and label-encode selected columns.

        Drops rows with missing values and converts categorical entries
        into integer codes for the specified columns.

        Returns
        -------
        pandas.DataFrame
            A cleaned and encoded copy of the input data.
        """
        df = df.replace(["NA", "N/A", ""], pd.NA).dropna(subset=cols).copy()
        for c in cols:
            df[c] = LabelEncoder().fit_transform(df[c])
        return df

    @staticmethod
    def _calculate_unconditional_helper(protected_col_values, response_col_values):
        """
        Compute unsigned marginal differences for all observed (S,O) pairs:
            MD(s,o) = |P(S,O) − P(S)P(O)|.
        """
        s_arr = np.asarray(protected_col_values, dtype=np.int64)
        o_arr = np.asarray(response_col_values, dtype=np.int64)
        N = len(s_arr)

        # joint counts C(s,o)
        joint_counts = Counter(zip(s_arr, o_arr))
        # marginals C(s), C(o)
        s_counts = Counter(s_arr)
        o_counts = Counter(o_arr)

        MD_vals = []
        for (s, o), c_so in joint_counts.items():
            p_so = c_so / N
            p_s = s_counts[s] / N
            p_o = o_counts[o] / N
            MD_vals.append(abs(p_so - p_s * p_o))

        return np.array(MD_vals, dtype=float)

    @staticmethod
    def _calculate_conditional_helper(protected_col_values, response_col_values, admissible_col_values):
        """
        Compute unsigned marginal differences for all observed (S,O,A) tuples:
            MD(s,o|a) = |P(S,O|A=a) − P(S|A=a)P(O|A=a)|.
        """
        s_arr = np.asarray(protected_col_values, dtype=np.int64)
        o_arr = np.asarray(response_col_values, dtype=np.int64)
        a_arr = np.asarray(admissible_col_values, dtype=np.int64)
        N = len(a_arr)

        # counts over triples (a,s,o)
        aso_counts = Counter(zip(a_arr, s_arr, o_arr))
        # counts per admissible value a
        a_counts = Counter(a_arr)
        # counts per (a,s)
        as_counts = Counter(zip(a_arr, s_arr))
        # counts per (a,o)
        ao_counts = Counter(zip(a_arr, o_arr))

        MD_vals = []
        for (a, s, o), c_aso in aso_counts.items():
            N_a = a_counts[a]  # total rows with this a
            p_so = c_aso / N_a  # P(S=s,O=o | A=a)
            p_s = as_counts[(a, s)] / N_a  # P(S=s | A=a)
            p_o = ao_counts[(a, o)] / N_a  # P(O=o | A=a)
            MD_vals.append(abs(p_so - p_s * p_o))

        return np.array(MD_vals, dtype=float)
