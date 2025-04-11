import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ProxyMutualInformationPrivbayes:

    def __init__(self, datapath):
        self.dataset = pd.read_csv(datapath)

    def calculate(self, column_name_1, column_name_2):
        self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)
        self.dataset.dropna(inplace=True, subset=[column_name_1, column_name_2])
        self.dataset[column_name_1] = LabelEncoder().fit_transform(self.dataset[column_name_1])
        self.dataset[column_name_2] = LabelEncoder().fit_transform(self.dataset[column_name_2])

        start_time = time.time()

        unique_1 = self.dataset[column_name_1].nunique()
        unique_2 = self.dataset[column_name_2].nunique()

        if unique_1 == 2 and unique_2 == 2:
            mi = self._getF(self.dataset[column_name_1], self.dataset[column_name_2])
        else:
            mi = self._getF_multiclass(column_name_1, column_name_2)
        mi += 0.5  # mapping

        elapsed_time = time.time() - start_time
        print(f"Privbayes: Proxy Mutual Information between '{column_name_1}' and '{column_name_2}': {mi:.4f}. "
              f"Calculation took {elapsed_time:.3f} seconds.")

    @staticmethod
    def _encode(values, widths):
        """Converts multidimensional index to 1D index"""
        index = 0
        base = 1
        for v, w in zip(reversed(values), reversed(widths)):
            index += v * base
            base *= w
        return index

    @staticmethod
    def _inc(values, bounds):
        """Increments a vector of indices with given bounds (like ++ on a multidimensional loop)"""
        for i in reversed(range(len(values))):
            if values[i] + 1 < bounds[i]:
                values[i] += 1
                for j in range(i + 1, len(values)):
                    values[j] = 0
                return True
        return False

    def _getF(self, col_1, col_2):
        """
        Compute F from two binary columns using the DP dynamic programming approach.
        """

        counts = np.zeros((2, 2))
        for xi, yi in zip(col_1, col_2):
            counts[xi, yi] += 1

        widths = list(counts.shape)
        counts = counts.flatten()
        total = counts.sum().item()
        ceil = (total + 1) // 2
        num_dims = len(widths)
        results = []

        for t in range(num_dims):
            bounds = list(widths)
            bounds[t] = 1  # always 0

            current_map = {0: 0}
            values = [0] * num_dims

            while True:
                conditional = []
                for col_1 in range(widths[t]):
                    values[t] = col_1
                    conditional.append(counts[self._encode(values, widths)])
                values[t] = 0

                next_map = defaultdict(int)
                for a, b in current_map.items():
                    a_new = min(a + int(conditional[0]), ceil)
                    b_new = min(b + int(conditional[1]), ceil)
                    next_map[a_new] = max(next_map.get(a_new, 0), b)
                    next_map[a] = max(next_map.get(a, 0), b_new)
                current_map = next_map

                if not self._inc(values, bounds):
                    break

            best = -total
            for a, b in current_map.items():
                best = max(best, a + b - total)
            results.append(best / total)

        return results[0]

    @staticmethod
    def int_to_binary_vector(val, num_bits):
        return [int(x) for x in format(val, f'0{num_bits}b')]

    @staticmethod
    def encode_column_to_bits(column):
        """
        Convert a categorical column to ⌈log2(n)⌉ binary features.
        Returns a 2D NumPy array of shape (n_samples, n_bits)
        """
        le = LabelEncoder()
        encoded = le.fit_transform(column)
        max_val = np.max(encoded)
        num_bits = int(np.ceil(np.log2(max_val + 1)))
        return np.array([ProxyMutualInformationPrivbayes.int_to_binary_vector(val, num_bits) for val in encoded])

    def _getF_multiclass(self, column_name_1, column_name_2):
        """
        Preprocess non-binary data into bitwise binary columns and compute F score per bit pair.
        """

        def preprocess_column(column, num_bins=16):
            """Binning if needed, then binary encode"""
            if column.dtype.kind in 'f':  # float = continuous
                # Equal-width binning
                binned = np.digitize(column, np.histogram_bin_edges(column, bins=num_bins)) - 1
            else:
                binned = LabelEncoder().fit_transform(column)
            return self.encode_column_to_bits(binned)

        col_1 = self.dataset[column_name_1]
        col_2 = self.dataset[column_name_2]

        col_1_bits = preprocess_column(np.array(col_1))
        col_2_bits = preprocess_column(np.array(col_2))

        n_bits_col1 = col_1_bits.shape[1]
        n_bits_col2 = col_2_bits.shape[1]

        F_scores = []

        for i in range(n_bits_col1):
            bit1 = col_1_bits[:, i]
            for j in range(n_bits_col2):
                bit2 = col_2_bits[:, j]

                # Skip degenerate bit-columns
                if len(np.unique(bit1)) < 2 or len(np.unique(bit2)) < 2:
                    continue

                f_score = self._getF(bit1, bit2)
                F_scores.append(f_score)

        return np.mean(F_scores) if F_scores else 0.0 # you can also return max(F_scores), etc.

    def _getF_multiclass_alternative(self, column_name_1, column_name_2):
        """
        One-vs-all generalization of F to multiclass variables.
        Returns the average F over all (one-vs-all x one-vs-all) pairs.
        """
        col_1 = self.dataset[column_name_1]
        col_2 = self.dataset[column_name_2]

        classes_1 = np.unique(col_1)
        classes_2 = np.unique(col_2)

        F_scores = []

        for class_1 in classes_1:
            binary_1 = (col_1 == class_1).astype(int)  # one-vs-all
            for class_2 in classes_2:
                binary_2 = (col_2 == class_2).astype(int)  # one-vs-all
                if len(np.unique(binary_1)) < 2 or len(np.unique(binary_2)) < 2:
                    continue
                f = self._getF(binary_1, binary_2)
                F_scores.append(f)

        return np.mean(F_scores) if F_scores else 0.0
