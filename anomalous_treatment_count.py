import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class AnomalousTreatmentCount:
    def __init__(self, datapath):
        """
        Initializes the AnomalousTreatmentCount object.

        Parameters:
        -----------
        datapath : str
            Path to the CSV dataset file.
        """
        self.dataset = pd.read_csv(datapath)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def calculate(self, s_col, o_col, a_col, threshold=0.25):
        """
        Calculates the Anomalous Treatment Count score.

        Parameters:
        -----------
        s_col : str
            Name of the sensitive attribute (S).
        o_col : str
            Name of the outcome attribute (O).
        a_col : str
            Name of the admissible/context attribute (A).
        threshold : float
            Threshold for deviation from stratum outcome rate.

        Returns:
        --------
        int
            Total count of individuals in anomalous (S, A) groups.
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

        # ---------- OLD RATES WITHOUT FAVORABLE VALUE: ---------
        # # Compute stratum base rates
        # base_rates = df.groupby(a_col)[o_col].mean().astype(float)
        #
        # # Compute subgroup outcome rates
        # subgroup_rates = df.groupby([s_col, a_col])[o_col].mean().astype(float)

        favorable_value = df[o_col].value_counts().idxmax()

        # Compute stratum base rates (P(O = favorable | A))
        base_rates = df.groupby(a_col).apply(
            lambda g: (g[o_col] == favorable_value).sum() / len(g)
        )

        # Compute subgroup outcome rates (P(O = favorable | S, A))
        subgroup_rates = df.groupby([s_col, a_col]).apply(
            lambda g: (g[o_col] == favorable_value).sum() / len(g)
        )

        group_counts = df.groupby([s_col, a_col]).size()

        # ---------- OLD NOT SMOOTHED VERSION: -----------
        # # Count individuals in anomalous subgroups
        # anomalous_groups = []
        # for (s, a), subgroup_rate in subgroup_rates.items():
        #     base_rate = base_rates.loc[a]
        #     if abs(subgroup_rate - base_rate) > threshold:
        #         anomalous_groups.append((s, a))
        # # Filter individuals in anomalous groups and count
        # is_anomalous = df[[s_col, a_col]].apply(tuple, axis=1).isin(anomalous_groups)
        # count = is_anomalous.sum()

        smoothed_count = 0
        for (s, a), subgroup_rate in subgroup_rates.items():
            base_rate = base_rates.loc[a]
            deviation = abs(subgroup_rate - base_rate)
            prob = self.sigmoid((deviation - threshold) / 0.01)
            smoothed_count += group_counts.loc[(s, a)] * prob

        print(f"Anomalous Treatment Count: {smoothed_count} individuals in unfair groups for dependency "
              f"{s_col} ⫫ {o_col} | {a_col} and threshold {threshold}.")
        return smoothed_count
