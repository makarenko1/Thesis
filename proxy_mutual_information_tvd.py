import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ProxyMutualInformationTVD:
    """
    Computes a proxy for (conditional) mutual information using Total Variation Distance (TVD).
    If a_col is provided, computes TVD-based proxy for I(S;O|A). Otherwise, computes I(S;O).
    """

    def __init__(self, datapath=None, data=None):
        """
        Initializes the proxy estimator with a dataset path.

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

    def calculate(self, s_col, o_col, a_col=None, epsilon=None):
        """
        Calculates the TVD proxy for mutual information I(S;O) or conditional mutual information I(S;O|A).

        Parameters:
        -----------
        s_col : str
            Sensitive attribute (S).
        o_col : str
            Outcome attribute (O).
        a_col : str, optional
            Attribute to condition on (A). If None, computes unconditional MI.

        Returns:
        --------
        float
            Proxy score based on total variation distance.
        """
        start_time = time.time()

        # Clean and encode data
        self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)
        cols = [s_col, o_col]
        if a_col is not None:
            cols += [a_col]
        self.dataset.dropna(inplace=True, subset=cols)
        for col in cols:
            self.dataset[col] = LabelEncoder().fit_transform(self.dataset[col])

        s_col_values = self.dataset[s_col]
        o_col_values = self.dataset[o_col]
        a_col_values = self.dataset[a_col]

        if a_col is None:
            tvd = self._calculate_tvd_unconditional(s_col_values, o_col_values)
        else:
            tvd = self._calculate_tvd_conditional(s_col_values, o_col_values, a_col_values)

        if epsilon is not None:
            if a_col is None:
                sensitivity = 6 / (len(s_col_values) + 1)
            else:
                unique, counts = np.unique(a_col_values, return_counts=True)
                max_count = np.max(counts)
                sensitivity = (2 / (len(s_col_values) + 1)) + (6 / (max_count + 1))
            tvd = tvd + np.random.laplace(loc=0, scale=sensitivity / epsilon)

        elapsed_time = time.time() - start_time
        print(
            f"TVD Proxy: Proxy Mutual Information between '{s_col}' and '{o_col}'"
            + (f" conditioned on '{a_col}'" if a_col else "")
            + f": {tvd:.4f}. Calculation took {elapsed_time:.3f} seconds."
        )
        return round(tvd, 4)

    @staticmethod
    def _calculate_tvd_unconditional(s_col_values, o_col_values):
        """
        Computes unconditional TVD between P(S,O) and P(S)P(O)

        Parameters:
        -----------
        s : np.ndarray[int]
        o : np.ndarray[int]

        Returns:
        --------
        float : TVD score
        """
        num_s = np.max(s_col_values) + 1
        num_o = np.max(o_col_values) + 1
        joint = np.zeros((num_s, num_o))
        for s, o in zip(s_col_values, o_col_values):
            joint[s, o] += 1
        joint /= len(s_col_values)

        p_s = joint.sum(axis=1, keepdims=True)
        p_o = joint.sum(axis=0, keepdims=True)
        expected = p_s @ p_o

        tvd = 0.5 * np.sum(np.abs(joint - expected))
        return 2 * tvd**2

    def _calculate_tvd_conditional(self, s_col_values, o_col_values, a_col_values):
        """
        Computes conditional TVD as the expected TVD over each group of A=a.

        Parameters:
        -----------
        s_col_values : np.ndarray[int]
        o_col_values : np.ndarray[int]
        a_col_values : np.ndarray[int]

        Returns:
        --------
        float : Expected conditional TVD score
        """
        total = len(a_col_values)
        tvd_total = 0.0

        for a_val in np.unique(a_col_values):
            mask = a_col_values == a_val
            s_sub = s_col_values[mask]
            o_sub = o_col_values[mask]
            if len(s_sub) == 0:
                continue
            weight = len(s_sub) / total
            tvd_total += weight * self._calculate_tvd_unconditional(s_sub, o_sub)

        return tvd_total
