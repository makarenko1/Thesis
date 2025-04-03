import math
import time

import numpy as np
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


class MutualInformation:

    def __init__(self):
        self.adult = fetch_ucirepo(id=2).data['original']
        self.adult.dropna(inplace=True)
        # Preprocess the 'income' column
        self.adult['income'] = self.adult['income'].apply(lambda x: 0 if x.startswith('<=50K') else 1)
        # Preprocess the 'sex' column
        self.adult['sex'] = self.adult['sex'].apply(lambda x: 0 if x.startswith('Male') else 1)

    def calculate(self, column_name_1, column_name_2):
        print(f"Computing mutual information between '{column_name_1}' and '{column_name_2}' treating both as "
              f"non-private")

        # Encode sex and income
        column_1_encoded = LabelEncoder().fit_transform(self.adult[column_name_1])
        column_2_encoded = LabelEncoder().fit_transform(self.adult[column_name_2])

        # Compute mutual information
        start_time = time.time()  # Record start time

        num_unique_column_1 = len(np.unique(column_1_encoded))
        num_unique_column_2 = len(np.unique(column_2_encoded))
        counts = np.zeros((num_unique_column_1, num_unique_column_2))
        for s, inc in zip(column_1_encoded, column_2_encoded):
            counts[s, inc] += 1
        # mi = self._getI(counts.flatten(), list(counts.shape))[0]
        mi = self._getI(column_name_1, column_name_2)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Non-Differentially Private Mutual Information between '{column_name_1}' and '{column_name_2}': "
              f"{mi:.4f}. Calculation took {elapsed_time:.3f} seconds.")

    def _getI(self, column_name_1, column_name_2):
        """
        Compute mutual information I(X; Y) from two categorical data columns using log base 2.
        :return: MI value
        """
        import numpy as np
        from sklearn.preprocessing import LabelEncoder

        # Encode to integers
        column_1_encoded = LabelEncoder().fit_transform(self.adult[column_name_1])
        column_2_encoded = LabelEncoder().fit_transform(self.adult[column_name_2])

        size_X = len(np.unique(column_1_encoded))
        size_Y = len(np.unique(column_2_encoded))

        # Build joint count table
        counts = np.zeros((size_X, size_Y))
        for xi, yi in zip(column_1_encoded, column_2_encoded):
            counts[xi, yi] += 1

        total = counts.sum()
        if total == 0:
            return [0.0]

        P_xy = counts / total
        P_x = P_xy.sum(axis=1, keepdims=True)
        P_y = P_xy.sum(axis=0, keepdims=True)

        mask = P_xy > 0
        ratio = np.divide(P_xy, P_x @ P_y, out=np.ones_like(P_xy), where=mask)
        MI = np.sum(P_xy[mask] * np.log2(ratio[mask]))

        return MI
