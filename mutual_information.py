import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MutualInformation:

    def __init__(self, datapath):
        self.dataset = pd.read_csv(datapath)

    def calculate(self, column_name_1, column_name_2):
        self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)
        self.dataset.dropna(inplace=True, subset=[column_name_1, column_name_2])
        self.dataset[column_name_1] = LabelEncoder().fit_transform(self.dataset[column_name_1])
        self.dataset[column_name_2] = LabelEncoder().fit_transform(self.dataset[column_name_2])

        # Compute mutual information
        start_time = time.time()  # Record start time

        mi = self._getI(column_name_1, column_name_2)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Regular Mutual Information between '{column_name_1}' and '{column_name_2}': "
              f"{mi:.4f}. Calculation took {elapsed_time:.3f} seconds.")

    def _getI(self, column_name_1, column_name_2):

        col1 = self.dataset[column_name_1]
        col2 = self.dataset[column_name_2]

        size_X = col1.max() + 1
        size_Y = col2.max() + 1

        counts = np.zeros((size_X, size_Y))
        for x, y in zip(col1, col2):
            counts[x, y] += 1

        total = counts.sum()
        if total == 0:
            return 0.0

        P_xy = counts / total
        P_x = P_xy.sum(axis=1, keepdims=True)
        P_y = P_xy.sum(axis=0, keepdims=True)

        mask = P_xy > 0
        ratio = np.divide(P_xy, P_x @ P_y, out=np.ones_like(P_xy), where=mask)
        MI = np.sum(P_xy[mask] * np.log2(ratio[mask]))

        return MI

