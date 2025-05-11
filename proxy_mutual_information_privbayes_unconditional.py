import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ProxyMutualInformationPrivbayesUnconditional:
    """
    Approximates mutual information using a dynamic programming-based proxy,
    inspired by the winning approach of the NIST PrivBayes competition.
    Supports binary and non-binary attributes using bitwise decomposition.
    """

    def __init__(self, datapath):
        """
        Initializes the class with a dataset loaded from the given CSV file.

        Parameters:
        -----------
        datapath : str
            Path to the CSV dataset file.
        """
        self.dataset = pd.read_csv(datapath)

    def calculate(self, s_col, o_col):
        """
        Computes the proxy mutual information between two attributes.

        Parameters:
        -----------
        s_col : str
            Name of the first attribute (S).
        o_col : str
            Name of the second attribute (O).

        Returns:
        --------
        float
            Proxy mutual information score.
        """
        self.dataset.replace(["NA", "N/A", ""], pd.NA, inplace=True)
        self.dataset.dropna(inplace=True, subset=[s_col, o_col])
        self.dataset[s_col] = LabelEncoder().fit_transform(self.dataset[s_col])
        self.dataset[o_col] = LabelEncoder().fit_transform(self.dataset[o_col])

        start_time = time.time()

        unique_1 = self.dataset[s_col].nunique()
        unique_2 = self.dataset[o_col].nunique()

        if unique_1 == 2 and unique_2 == 2:
            mi = self.getF(self.dataset[s_col].to_numpy(), self.dataset[o_col].to_numpy())
        else:
            mi = self._getF_multiclass_unconditional(s_col, o_col)

        mi += 0.5  # Offset to ensure non-negative output

        elapsed_time = time.time() - start_time
        print(f"Privbayes: Proxy Mutual Information between '{s_col}' and '{o_col}': {mi:.4f}. "
              f"Calculation took {elapsed_time:.3f} seconds.")
        return round(mi, 4)

    @staticmethod
    def encode(values, widths):
        """
        Converts a multidimensional index to a single flattened index.

        Parameters:
        -----------
        values : list[int]
        widths : list[int]

        Returns:
        --------
        int : Flattened index.
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
        Increments a multidimensional index vector in lexicographic order.

        Parameters:
        -----------
        values : list[int]
        bounds : list[int]

        Returns:
        --------
        bool : True if increment succeeded, False if overflowed.
        """
        for i in reversed(range(len(values))):
            if values[i] + 1 < bounds[i]:
                values[i] += 1
                for j in range(i + 1, len(values)):
                    values[j] = 0
                return True
        return False

    def getF(self, s_col_values, o_col_values):
        """
        Compute the F score from two binary-encoded columns using a DP approach.

        Parameters:
        -----------
        s_col_values : np.ndarray[int]
        o_col_values : np.ndarray[int]

        Returns:
        --------
        float : F-score proxy for mutual information.
        """
        counts = np.zeros((2, 2))
        for s, o in zip(s_col_values, o_col_values):
            counts[s, o] += 1

        widths = list(counts.shape)
        counts = counts.flatten()
        total = counts.sum().item()
        ceil = (total + 1) // 2
        num_dims = len(widths)
        results = []

        for t in range(num_dims):
            bounds = list(widths)
            bounds[t] = 1

            current_map = {0: 0}
            values = [0] * num_dims

            while True:
                conditional = []
                for s in range(widths[t]):
                    values[t] = s
                    conditional.append(counts[self.encode(values, widths)])
                values[t] = 0

                next_map = defaultdict(int)
                for a, b in current_map.items():
                    a_new = min(a + int(conditional[0]), ceil)
                    b_new = min(b + int(conditional[1]), ceil)
                    next_map[a_new] = max(next_map.get(a_new, 0), b)
                    next_map[a] = max(next_map.get(a, 0), b_new)
                current_map = next_map

                if not self.inc(values, bounds):
                    break

            best = -total
            for a, b in current_map.items():
                best = max(best, a + b - total)
            results.append(best / total)

        return results[0]

    @staticmethod
    def int_to_binary_vector(val, num_bits):
        """
        Converts an integer to a fixed-length binary vector.

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
        Converts a categorical column into binary-encoded vectors.

        Parameters:
        -----------
        column : np.ndarray[int]

        Returns:
        --------
        np.ndarray[int] : Array of binary vectors (n_samples, n_bits).
        """
        le = LabelEncoder()
        encoded = le.fit_transform(column)
        max_val = np.max(encoded)
        num_bits = int(np.ceil(np.log2(max_val + 1)))
        return np.array([
            ProxyMutualInformationPrivbayesUnconditional.int_to_binary_vector(val, num_bits)
            for val in encoded
        ])

    def _getF_multiclass_unconditional(self, s_col, o_col):
        """
        Compute the average F score for multiclass attributes using binary decomposition.

        Parameters:
        -----------
        s_col : str
        o_col : str

        Returns:
        --------
        float : Averaged F-score across bitwise binary encodings.
        """
        def preprocess_column(column, num_bins=16):
            if column.dtype.kind in 'f':
                binned = np.digitize(column, np.histogram_bin_edges(column, bins=num_bins)) - 1
            else:
                binned = LabelEncoder().fit_transform(column)
            return self.encode_column_to_bits(binned)

        s_col_values = self.dataset[s_col].to_numpy()
        o_col_values = self.dataset[o_col].to_numpy()

        s_col_bits = preprocess_column(np.array(s_col_values))
        o_col_bits = preprocess_column(np.array(o_col_values))

        n_bits_s_col = s_col_bits.shape[1]
        n_bits_o_col = o_col_bits.shape[1]

        F_scores = []

        for i in range(n_bits_s_col):
            bit1 = s_col_bits[:, i]
            for j in range(n_bits_o_col):
                bit2 = o_col_bits[:, j]
                if len(np.unique(bit1)) < 2 or len(np.unique(bit2)) < 2:
                    continue
                f_score = self.getF(bit1, bit2)
                F_scores.append(f_score)

        return np.mean(F_scores) if F_scores else 0.0

    def _getF_multiclass_alternative(self, s_col, o_col):
        """
        Computes F score using a one-vs-all strategy for each class pair.

        Parameters:
        -----------
        s_col : str
        o_col : str

        Returns:
        --------
        float : Averaged F-score over all (class x class) binary combinations.
        """
        s_col_values = self.dataset[s_col].to_numpy()
        o_col_values = self.dataset[o_col].to_numpy()

        s_col_classes = np.unique(s_col_values)
        o_col_classes = np.unique(o_col_values)

        F_scores = []

        for class_1 in s_col_classes:
            binary_1 = (s_col_values == class_1).astype(int)
            for class_2 in o_col_classes:
                binary_2 = (o_col_values == class_2).astype(int)
                if len(np.unique(binary_1)) < 2 or len(np.unique(binary_2)) < 2:
                    continue
                f = self.getF(binary_1, binary_2)
                F_scores.append(f)

        return np.mean(F_scores) if F_scores else 0.0
