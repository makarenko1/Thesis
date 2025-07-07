import time
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ProxyMutualInformationPrivbayes:
    """
    Computes a proxy for mutual information or conditional mutual information using
    a dynamic programming-based approach inspired by PrivBayes. Supports binary and
    non-binary categorical features through binary encoding.
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
        Calculate proxy mutual information I(S;O) or conditional mutual information I(S;O|A).

        Parameters:
        -----------
        s_col : str
            Sensitive attribute (S).
        o_col : str
            Outcome attribute (O).
        a_col : str, optional
            Conditional attribute (A), if computing conditional MI.

        Returns:
        --------
        float
            Approximated mutual information score.
        """
        self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)
        cols = [s_col, o_col]
        if a_col is not None:
            cols += [a_col]
        self.dataset.dropna(inplace=True, subset=cols)
        for col in cols:
            self.dataset[col] = LabelEncoder().fit_transform(self.dataset[col])

        start_time = time.time()

        if a_col:
            unique_vals = [self.dataset[col].nunique() for col in [s_col, o_col, a_col]]
            if all(x == 2 for x in unique_vals):
                mi = self._getF_conditional(
                    self.dataset[s_col].to_numpy(),
                    self.dataset[o_col].to_numpy(),
                    self.dataset[a_col].to_numpy()
                )
            else:
                mi = self._getF_multiclass_conditional(
                    self.dataset[s_col].to_numpy(),
                    self.dataset[o_col].to_numpy(),
                    self.dataset[a_col].to_numpy()
                )
        else:
            unique_vals = [self.dataset[col].nunique() for col in [s_col, o_col]]
            if all(x == 2 for x in unique_vals):
                mi = self._getF_unconditional(
                    self.dataset[s_col].to_numpy(),
                    self.dataset[o_col].to_numpy()
                )
            else:
                mi = self._getF_multiclass_unconditional(s_col, o_col)

        mi += 0.5  # non-negative adjustment
        elapsed_time = time.time() - start_time
        print(f"Privbayes: Proxy Mutual Information between '{s_col}' and '{o_col}'"
              + (f" conditioned on '{a_col}'" if a_col else "")
              + f": {mi:.4f}. Calculation took {elapsed_time:.3f} seconds.")
        return round(mi, 4)

    @staticmethod
    def encode(values, widths):
        """
        Convert a multidimensional index to a flat index.

        Parameters:
        -----------
        values : list[int]
        widths : list[int]

        Returns:
        --------
        int : Encoded flat index.
        """
        index = 0
        base = 1
        for v, w in zip(reversed(values), reversed(widths)):
            index += v * base
            base *= w
        return index

    @staticmethod
    def inc(values, bounds):
        """
        Increment multidimensional index in lexicographic order.

        Parameters:
        -----------
        values : list[int]
        bounds : list[int]

        Returns:
        --------
        bool : True if increment succeeded; False if overflowed.
        """
        for i in reversed(range(len(values))):
            if values[i] + 1 < bounds[i]:
                values[i] += 1
                for j in range(i + 1, len(values)):
                    values[j] = 0
                return True
        return False

    @staticmethod
    def int_to_binary_vector(val, num_bits):
        """
        Convert integer to fixed-length binary list.

        Parameters:
        -----------
        val : int
        num_bits : int

        Returns:
        --------
        list[int] : Binary representation.
        """
        return [int(x) for x in format(val, f'0{num_bits}b')]

    @staticmethod
    def encode_column_to_bits(column):
        """
        Encode categorical values into binary feature vectors.

        Parameters:
        -----------
        column : np.ndarray[int]

        Returns:
        --------
        np.ndarray[int] : Binary matrix (n_samples, n_bits)
        """
        le = LabelEncoder()
        encoded = le.fit_transform(column)
        max_val = np.max(encoded)
        num_bits = int(np.ceil(np.log2(max_val + 1)))
        return np.array([
            ProxyMutualInformationPrivbayes.int_to_binary_vector(val, num_bits)
            for val in encoded
        ])

    def _getF_unconditional(self, s_col_values, o_col_values):
        """
        Compute proxy F score for binary unconditional mutual information.

        Parameters:
        -----------
        s_col_values : np.ndarray[int]
        o_col_values : np.ndarray[int]

        Returns:
        --------
        float : F score.
        """
        counts = np.zeros((2, 2))
        for s, o in zip(s_col_values, o_col_values):
            counts[s, o] += 1

        widths = list(counts.shape)
        flat_counts = counts.flatten()
        total = flat_counts.sum().item()
        ceil = (total + 1) // 2
        results = []

        for t in range(2):
            bounds = list(widths)
            bounds[t] = 1
            current_map = {0: 0}
            values = [0] * 2

            while True:
                conditional = []
                for s_val in range(2):
                    values[t] = s_val
                    idx = self.encode(values, widths)
                    conditional.append(flat_counts[idx])
                values[t] = 0
                next_map = defaultdict(int)
                for a, b in current_map.items():
                    a_new = min(a + int(conditional[0]), ceil)
                    b_new = min(b + int(conditional[1]), ceil)
                    next_map[a_new] = max(next_map[a_new], b)
                    next_map[a] = max(next_map[a], b_new)
                current_map = next_map

                if not self.inc(values, bounds):
                    break

            best = max((a + b - total) for a, b in current_map.items())
            results.append(best / total)

        return results[0]

    def _getF_conditional(self, s_col_values, o_col_values, a_col_values):
        """
        Compute proxy F score for binary conditional mutual information.

        Parameters:
        -----------
        s_col_values : np.ndarray[int]
        o_col_values : np.ndarray[int]
        a_col_values : np.ndarray[int]

        Returns:
        --------
        float : F score.
        """
        d_s = len(np.unique(s_col_values))
        d_o = len(np.unique(o_col_values))
        d_a = len(np.unique(a_col_values))

        counts = np.zeros((d_s, d_o, d_a))
        for s, o, a in zip(s_col_values, o_col_values, a_col_values):
            counts[s, o, a] += 1

        total_ans = 0
        grand_total = 0

        for a_val in range(d_a):
            sub_counts = counts[:, :, a_val]
            widths = list(sub_counts.shape)
            flat_counts = sub_counts.flatten()
            total = flat_counts.sum().item()
            if total == 0:
                continue
            ceil = (total + 1) // 2
            num_dims = len(widths)

            for t in range(num_dims):
                bounds = list(widths)
                bounds[t] = 1
                current_map = {0: 0}
                values = [0] * num_dims

                while True:
                    conditional = [flat_counts[self.encode(values[:t] + [s] + values[t+1:], widths)]
                                   for s in range(widths[t])]
                    next_map = defaultdict(int)
                    for a, b in current_map.items():
                        a_new = min(a + int(conditional[0]), ceil)
                        b_new = min(b + int(conditional[1]), ceil)
                        next_map[a_new] = max(next_map[a_new], b)
                        next_map[a] = max(next_map[a], b_new)
                    current_map = next_map

                    if not self.inc(values, bounds):
                        break

                best = max((a + b - total) for a, b in current_map.items())
                total_ans += (best / total) * total
                grand_total += total

        return total_ans / grand_total if grand_total > 0 else 0.0

    def _getF_multiclass_unconditional(self, s_col, o_col):
        """
        Handle non-binary attributes by encoding to bits and averaging over bitwise F scores.

        Parameters:
        -----------
        s_col : str
        o_col : str

        Returns:
        --------
        float : Average F score over bitwise combinations.
        """
        def preprocess_column(column):
            if column.dtype.kind in 'f':
                binned = np.digitize(column, np.histogram_bin_edges(column, bins=16)) - 1
            else:
                binned = LabelEncoder().fit_transform(column)
            return self.encode_column_to_bits(binned)

        s_bits = preprocess_column(self.dataset[s_col])
        o_bits = preprocess_column(self.dataset[o_col])

        scores = []
        for i in range(s_bits.shape[1]):
            for j in range(o_bits.shape[1]):
                bit1, bit2 = s_bits[:, i], o_bits[:, j]
                if len(np.unique(bit1)) < 2 or len(np.unique(bit2)) < 2:
                    continue
                scores.append(self._getF_unconditional(bit1, bit2))
        return np.mean(scores) if scores else 0.0

    def _getF_multiclass_conditional(self, s_col_values, o_col_values, a_col_values):
        """
        Handle non-binary conditional MI by averaging F scores across all bitwise triplets.

        Parameters:
        -----------
        s_col_values : np.ndarray
        o_col_values : np.ndarray
        a_col_values : np.ndarray

        Returns:
        --------
        float : Average F score over bitwise combinations.
        """
        def preprocess_column(column):
            if column.dtype.kind in 'f':
                binned = np.digitize(column, np.histogram_bin_edges(column, bins=16)) - 1
            else:
                binned = LabelEncoder().fit_transform(column)
            return self.encode_column_to_bits(binned)

        s_bits = preprocess_column(s_col_values)
        o_bits = preprocess_column(o_col_values)
        a_bits = preprocess_column(a_col_values)

        scores = []
        for i in range(s_bits.shape[1]):
            for j in range(o_bits.shape[1]):
                for k in range(a_bits.shape[1]):
                    bit1, bit2, bit3 = s_bits[:, i], o_bits[:, j], a_bits[:, k]
                    if len(np.unique(bit1)) < 2 or len(np.unique(bit2)) < 2:
                        continue
                    scores.append(self._getF_conditional(bit1, bit2, bit3))
        return np.mean(scores) if scores else 0.0
