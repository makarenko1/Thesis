import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ProxyMutualInformationPrivbayesConditional:

    def __init__(self, datapath):
        self.dataset = pd.read_csv(datapath)

    def calculate(self, s_col, o_col, a_col):
        self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)
        self.dataset.dropna(inplace=True, subset=[s_col, o_col])
        self.dataset[s_col] = LabelEncoder().fit_transform(self.dataset[s_col])
        self.dataset[o_col] = LabelEncoder().fit_transform(self.dataset[o_col])
        self.dataset[a_col] = LabelEncoder().fit_transform(self.dataset[a_col])

        start_time = time.time()

        unique_1 = self.dataset[s_col].nunique()
        unique_2 = self.dataset[o_col].nunique()
        unique_3 = self.dataset[a_col].nunique()

        if unique_1 == 2 and unique_2 == 2 and unique_3 == 2:
            mi = self._getF(self.dataset[s_col], self.dataset[o_col], self.dataset[a_col])
        else:
            mi = self._getF_multiclass_conditional(s_col, o_col, a_col)
        mi += 0.5  # mapping

        elapsed_time = time.time() - start_time
        print(f"Privbayes: Proxy Mutual Information between '{s_col}' and '{o_col}': {mi:.4f}. "
              f"Calculation took {elapsed_time:.3f} seconds.")
        return round(mi, 4)

    @staticmethod
    def encode(values, widths):
        index = 0
        base = 1
        for v, w in zip(reversed(values), reversed(widths)):
            index += v * base
            base *= w
        return index

    @staticmethod
    def inc(values, bounds):
        for i in reversed(range(len(values))):
            if values[i] + 1 < bounds[i]:
                values[i] += 1
                for j in range(i + 1, len(values)):
                    values[j] = 0
                return True
        return False

    def _getF(self, s_col_values, o_col_values, a_col_values):
        d_s = len(np.unique(s_col_values))
        d_o = len(np.unique(o_col_values))
        d_a = len(np.unique(a_col_values))

        widths = [d_s, d_o, d_a]
        counts = np.zeros((d_s, d_o, d_a))
        for s, o, a in zip(s_col_values, o_col_values, a_col_values):
            counts[s, o, a] += 1

        flat_counts = counts.flatten()
        total_ans = 0.0
        grand_total = 0
        for a_val in range(d_a):
            filtered_counts = np.zeros_like(flat_counts)
            total_count = 0
            bounds = list(widths)
            values = [0, 0, 0]
            while True:
                if values[2] == a_val:
                    idx = self.encode(values, widths)
                    filtered_counts[idx] = flat_counts[idx]
                    total_count += int(flat_counts[idx])
                if not self.inc(values, bounds):
                    break
            if total_count == 0:
                continue

            now = {0: 0}
            ceil = (total_count + 1) // 2
            values = [0, 0, 0]
            bounds_conditional = list(widths)
            bounds_conditional[1] = 1  # iterate only over S
            while True:
                if values[2] != a_val:
                    if not self.inc(values, bounds_conditional):
                        break
                    continue

                conditional = []
                for o_val in range(d_o):
                    values[1] = o_val
                    conditional.append(filtered_counts[self.encode(values, widths)])
                values[1] = 0

                next_map = defaultdict(int)
                for a, b in now.items():
                    a_new = min(a + int(conditional[0]), ceil)
                    b_new = min(b + int(conditional[1]), ceil)
                    next_map[a_new] = max(next_map[a_new], b)
                    next_map[a] = max(next_map[a], b_new)
                now = next_map

                if not self.inc(values, bounds_conditional):
                    break

            best = -total_count
            for a, b in now.items():
                best = max(best, a + b - total_count)
            total_ans += (best / total_count) * total_count
            grand_total += total_count

        return total_ans / grand_total if grand_total > 0 else 0.0

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
        return np.array([ProxyMutualInformationPrivbayesConditional.int_to_binary_vector(val, num_bits) for val in encoded])

    def _getF_multiclass_conditional(self, s_col, o_col, a_col):
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

        col_1 = self.dataset[s_col]
        col_2 = self.dataset[o_col]
        col_3 = self.dataset[a_col]

        col_1_bits = preprocess_column(np.array(col_1))
        col_2_bits = preprocess_column(np.array(col_2))
        col_3_bits = preprocess_column(np.array(col_3))

        n_bits_col1 = col_1_bits.shape[1]
        n_bits_col2 = col_2_bits.shape[1]
        n_bits_col3 = col_3_bits.shape[1]

        F_scores = []

        for i in range(n_bits_col1):
            bit1 = col_1_bits[:, i]
            for j in range(n_bits_col2):
                bit2 = col_2_bits[:, j]
                for k in range(n_bits_col3):
                    bit3 = col_3_bits[:, k]

                    # Skip degenerate bit-columns
                    if len(np.unique(bit1)) < 2 or len(np.unique(bit2)) < 2:
                        continue

                    f_score = self._getF(bit1, bit2, bit3)
                    F_scores.append(f_score)

        return np.mean(F_scores) if F_scores else 0.0  # you can also return max(F_scores), etc.
