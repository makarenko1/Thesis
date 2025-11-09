import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class PMIThresholdDetector:
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

    def calculate(self, s_col, o_col, a_col, threshold=0.25):
        """
        Calculates the number of individuals where PMI(S,O|A) > threshold.

        Parameters:
        -----------
        s_col : str
            Name of the sensitive attribute (S).
        o_col : str
            Name of the outcome attribute (O).
        a_col : str
            Name of the admissible/context attribute (A).
        threshold : float
            PMI threshold to consider (default: 0.25).

        Returns:
        --------
        int
            Total count of tuples with PMI > threshold.
        """
        # Preprocess
        self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)
        self.dataset.dropna(inplace=True, subset=[s_col, o_col, a_col])
        self.dataset[s_col] = LabelEncoder().fit_transform(self.dataset[s_col])
        self.dataset[o_col] = LabelEncoder().fit_transform(self.dataset[o_col])
        self.dataset[a_col] = LabelEncoder().fit_transform(self.dataset[a_col])

        df = self.dataset[[s_col, o_col, a_col]].copy()
        if df.empty:
            print("Warning: Dataset is empty after cleaning.")
            return 0

        # Group into strata
        flagged_count = 0
        for a_val, group_df in df.groupby(a_col):
            total = len(group_df)
            if total == 0:
                continue

            # Empirical joint and marginal probabilities
            p_s = group_df[s_col].value_counts(normalize=True).to_dict()
            p_o = group_df[o_col].value_counts(normalize=True).to_dict()
            p_so = group_df.groupby([s_col, o_col]).size().div(total).to_dict()

            # For each row in group, compute PMI
            for _, row in group_df.iterrows():
                s, o = row[s_col], row[o_col]
                p_joint = p_so.get((s, o), 1e-10)
                p_s_val = p_s.get(s, 1e-10)
                p_o_val = p_o.get(o, 1e-10)

                pmi = np.log(p_joint / (p_s_val * p_o_val))
                if abs(pmi) > threshold:
                    flagged_count += 1

        print(f"PMI Threshold Detector: {flagged_count} individuals for dependency "
              f"{s_col} ⫫ {o_col} | {a_col} and threshold {threshold}.")
        return flagged_count
