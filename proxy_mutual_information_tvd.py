import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ProxyMutualInformationTVD:
    """
    Implementation of the proxy mutual information metric using total variation distance (TVD).
    This simplified version assumes that the protected, response, and admissible sets
    each contain at most one attribute (a single column).

    The metric estimates dependence between protected and response variables, optionally
    conditioned on an admissible variable, by computing a TVD-based proxy for mutual information.
    Laplace noise can be added for differential privacy.
    """

    def __init__(self, datapath=None, data=None):
        if (datapath is None and data is None) or (datapath is not None and data is not None):
            raise ValueError("Usage: Should pass either datapath or data itself")
        if datapath is not None:
            self.dataset = pd.read_csv(datapath)
        else:
            self.dataset = data

    def calculate(self, fairness_criteria, epsilon=None, encode_and_clean=False):
        """
        Compute the total proxy mutual information (TVD-based) over all fairness criteria.

        Each criterion can include two or three columns:
          (protected, response) or (protected, response, admissible).

        If epsilon is provided, Laplace noise is added for differential privacy.
        Prints the total value and computation time.

        Returns
        -------
        float
            The total proxy mutual information value (rounded to 4 decimals).
        """
        start_time = time.time()

        tvd_proxy = 0.0

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

            if admissible_col is None:
                tvd_proxy += self._calculate_unconditional_helper(df[protected_col].to_numpy(),
                                                                  df[response_col].to_numpy())
            else:
                tvd_proxy += self._calculate_conditional_helper(df[protected_col].to_numpy(),
                                                                df[response_col].to_numpy(),
                                                                df[admissible_col].to_numpy())

        if epsilon is not None:
            n = len(self.dataset)
            sensitivity = len(fairness_criteria) * 16 / n
            tvd_proxy = tvd_proxy + np.random.laplace(loc=0, scale=sensitivity / epsilon)

        elapsed_time = time.time() - start_time
        print(
            f"TVD Proxy: Proxy Mutual Information for fairness criteria {fairness_criteria}: {tvd_proxy:.4f} with "
            f"data size: {len(self.dataset)} and epsilon: {epsilon if epsilon is not None else 'infinity'}. "
            f"Calculation took {elapsed_time:.3f} seconds."
        )
        return round(tvd_proxy, 4)

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
        Compute the unconditional TVD-based proxy mutual information between
        a protected attribute and a response attribute.

        Returns
        -------
        float
            The proxy MI value for the given attribute pair.
        """
        num_s = int(np.max(protected_col_values)) + 1
        num_o = int(np.max(response_col_values)) + 1
        joint = np.zeros((num_s, num_o))
        for s, o in zip(protected_col_values, response_col_values):
            joint[s, o] += 1
        joint /= len(protected_col_values)

        p_s = joint.sum(axis=1, keepdims=True)
        p_o = joint.sum(axis=0, keepdims=True)
        expected = p_s @ p_o

        tvd = 0.5 * np.sum(np.abs(joint - expected))
        return 2 * tvd**2

    def _calculate_conditional_helper(self, protected_col_values, response_col_values, admissible_col_values):
        """
        Compute the conditional TVD-based proxy mutual information, averaging
        the unconditional measure across admissible groups.

        Returns
        -------
        float
            The conditional proxy MI value weighted by group sizes.
        """
        total = len(admissible_col_values)
        tvd_total = 0.0

        for a_val in np.unique(admissible_col_values):
            mask = admissible_col_values == a_val
            s_sub = protected_col_values[mask]
            o_sub = response_col_values[mask]
            if len(s_sub) == 0:
                continue
            weight = len(s_sub) / total
            tvd_total += weight * self._calculate_unconditional_helper(s_sub, o_sub)

        return tvd_total
