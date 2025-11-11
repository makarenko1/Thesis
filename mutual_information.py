import math
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MutualInformation:
    """
    MutualInformation

    Simple implementation for computing mutual information (MI) as a fairness
    metric. The method assumes that each of the protected, response, and
    admissible sets contains at most one attribute (a single column).

    Computes both unconditional and conditional mutual information between
    the protected and response attributes, optionally adding Laplace noise
    for differential privacy.
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
        Compute mutual information for a list of fairness criteria.

        Each criterion is defined by two or three column names:
          (protected, response) or (protected, response, admissible).

        The method estimates empirical MI for each criterion and sums the results.
        If epsilon is provided, Laplace noise is added for differential privacy.

        Returns
        -------
        float
            The total mutual information value (possibly noised).
        """
        start_time = time.time()

        mi = 0.0

        for criterion in fairness_criteria:
            if len(criterion) not in [2, 3]:
                raise ValueError("Invalid input")

            protected_col, response_col, admissible_col = (criterion[0], criterion[1],
                                                           None if len(criterion) == 2 else criterion[2])
            cols = [protected_col, response_col] + ([admissible_col] if admissible_col is not None else [])
            if encode_and_clean:
                df = self._encode_and_clean(self.dataset, cols)

            mi += self._calculate_helper(df[protected_col].to_numpy(), df[response_col].to_numpy(),
                                         df[admissible_col].to_numpy() if admissible_col else None)

        if epsilon is not None:
            n = len(self.dataset)
            sensitivity = len(fairness_criteria) * ((2 / n) * math.log(n) + ((n - 1) / n) * math.log(n / (n - 1)))
            mi = mi + np.random.laplace(loc=0, scale=sensitivity / epsilon)

        elapsed_time = time.time() - start_time
        print(
            f"Mutual Information: MI for fairness criteria {fairness_criteria}: "
            f"{mi:.4f} with epsilon: {epsilon if epsilon is not None else 'infinity'}. "
            f"Calculation took {elapsed_time:.3f} seconds."
        )
        return round(mi, 4)

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

    def _calculate_helper(self, protected_col_values, response_col_values, admissible_col_values=None):
        """
        Compute empirical mutual information between protected and response
        attributes, optionally conditioned on an admissible attribute.

        For unconditional MI:
            MI(S;O) = Σ P(s,o) log₂( P(s,o) / (P(s)P(o)) )

        For conditional MI:
            MI(S;O|A) = Σ_a P(a) MI(S;O | A=a)

        Returns
        -------
        float
            Mutual information value (bits).
        """
        if admissible_col_values is None:
            size_X = protected_col_values.max() + 1
            size_Y = response_col_values.max() + 1
            counts = np.zeros((size_X, size_Y))
            for s_vals, o_vals in zip(protected_col_values, response_col_values):
                counts[s_vals, o_vals] += 1

            total = counts.sum()
            if total == 0:
                return 0.0

            P_xy = counts / total
            P_x = P_xy.sum(axis=1, keepdims=True)
            P_y = P_xy.sum(axis=0, keepdims=True)

            mask = P_xy > 0
            ratio = np.divide(P_xy, P_x @ P_y, out=np.ones_like(P_xy), where=mask)
            return np.sum(P_xy[mask] * np.log2(ratio[mask]))

        else:
            total_count = len(self.dataset)
            result = 0.0

            for a_val in np.unique(admissible_col_values):
                mask_z = admissible_col_values == a_val
                s_vals = protected_col_values[mask_z]
                o_vals = response_col_values[mask_z]
                n = len(s_vals)
                if n == 0:
                    continue

                size_X = s_vals.max() + 1
                size_Y = o_vals.max() + 1
                counts = np.zeros((size_X, size_Y))
                for xi, yi in zip(s_vals, o_vals):
                    counts[xi, yi] += 1

                P_xy = counts / n
                P_x = P_xy.sum(axis=1, keepdims=True)
                P_y = P_xy.sum(axis=0, keepdims=True)

                mask = P_xy > 0
                ratio = np.divide(P_xy, P_x @ P_y, out=np.ones_like(P_xy), where=mask)
                mi_z = np.sum(P_xy[mask] * np.log2(ratio[mask]))

                result += (n / total_count) * mi_z

            return result
