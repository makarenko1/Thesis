import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MutualInformation:
    """
    A class to compute Mutual Information (MI) or Conditional Mutual Information (CMI)
    between two categorical attributes in a dataset, with optional conditioning on a third attribute.
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

    def calculate(self, s_col, o_col, a_col=None):
        """
        Compute the Mutual Information (MI) between two columns, optionally conditioned on a third column.

        Parameters:
        -----------
        s_col : str
            Name of the first categorical column (S).
        o_col : str
            Name of the second categorical column (O).
        a_col : str, optional
            Name of the third categorical column (A) to condition on. Default is None.

        Returns:
        --------
        float
            The mutual information or conditional mutual information score rounded to 4 decimal places.
        """
        self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)
        cols = [s_col, o_col]
        if a_col is not None:
            cols += [a_col]
        self.dataset.dropna(inplace=True, subset=cols)
        for col in cols:
            self.dataset[col] = LabelEncoder().fit_transform(self.dataset[col])

        # Compute mutual information
        start_time = time.time()  # Record start time

        mi = self._getI(self.dataset[s_col].to_numpy(),
                        self.dataset[o_col].to_numpy(),
                        self.dataset[a_col].to_numpy() if a_col else None)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Regular Mutual Information for dependency '{s_col}' ⫫ '{o_col}'" +
              (f" | {a_col}" if a_col is not None else "") +
              f"is: {mi:.4f}. Calculation took {elapsed_time:.3f} seconds.")
        return round(mi, 4)

    def _getI(self, s_col_values, o_col_values, a_col_values=None):
        """
        Internal method to compute MI or CMI from label-encoded NumPy arrays.

        Parameters:
        -----------
        s_col_values : np.ndarray
            Encoded values of the S attribute.
        o_col_values : np.ndarray
            Encoded values of the O attribute.
        a_col_values : np.ndarray or None
            Encoded values of the A attribute, if conditioning is required.

        Returns:
        --------
        float
            The computed MI or CMI score in bits (base-2).
        """
        if a_col_values is None:
            size_X = s_col_values.max() + 1
            size_Y = o_col_values.max() + 1
            counts = np.zeros((size_X, size_Y))
            for s_vals, o_vals in zip(s_col_values, o_col_values):
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

            for a_val in np.unique(a_col_values):
                mask_z = a_col_values == a_val
                s_vals = s_col_values[mask_z]
                o_vals = o_col_values[mask_z]
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
