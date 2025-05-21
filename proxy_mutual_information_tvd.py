import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ProxyMutualInformationTVD:
    """
    Computes a proxy for (conditional) mutual information using Total Variation Distance (TVD).
    If a_col is provided, computes TVD-based proxy for I(S;O|A). Otherwise, computes I(S;O).
    """

    def __init__(self, datapath):
        """
        Initializes the proxy estimator with a dataset path.

        Parameters:
        -----------
        datapath : str
            Path to the CSV dataset file.
        """
        self.dataset = pd.read_csv(datapath)

    def calculate(self, s_col, o_col, a_col=None):
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
        columns = [s_col, o_col] + ([a_col] if a_col else [])
        self.dataset.dropna(inplace=True, subset=columns)

        s = LabelEncoder().fit_transform(self.dataset[s_col])
        o = LabelEncoder().fit_transform(self.dataset[o_col])

        if a_col is None:
            tvd = self._calculate_tvd_unconditional(s, o)
        else:
            a = LabelEncoder().fit_transform(self.dataset[a_col])
            tvd = self._calculate_tvd_conditional(s, o, a)

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

        tvd = np.sum(np.abs(joint - expected))

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
