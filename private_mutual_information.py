import math
import time
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


class PrivateMutualInformation:

    def __init__(self):
        self.adult = fetch_ucirepo(id=2).data['original']
        self.adult.dropna(inplace=True)
        # Preprocess the 'income' column
        self.adult['income'] = self.adult['income'].apply(lambda x: 0 if x.startswith('<=50K') else 1)
        # Preprocess the 'sex' column
        self.adult['sex'] = self.adult['sex'].apply(lambda x: 0 if x.startswith('Male') else 1)

    def calculate(self, column_name_1, column_name_2):
        print(f"Computing mutual information between '{column_name_1}' and '{column_name_2}' treating them as private")

        # Encode sex and income
        column_1_encoded = LabelEncoder().fit_transform(self.adult[column_name_1])
        column_2_encoded = LabelEncoder().fit_transform(self.adult[column_name_2])

        # Compute F
        start_time = time.time()  # Record start time

        num_unique_column_1 = len(np.unique(column_1_encoded))
        num_unique_column_2 = len(np.unique(column_2_encoded))
        counts = np.zeros((num_unique_column_1, num_unique_column_2))
        for s, inc in zip(column_1_encoded, column_2_encoded):
            counts[s, inc] += 1
        if [num_unique_column_1, num_unique_column_2] == [2, 2]:
            mi = self._getF(counts.flatten(), list(counts.shape))[0]
        else:
            mi = self._getF_multiclass(counts)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Differentially-Private Mutual Information between '{column_name_1}' and '{column_name_2}': {mi:.4f}. "
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

    @staticmethod
    def _getF(counts, widths):
        """
        Python translation of table::getF from C++
        :param counts: 1D list or np.array of flattened counts
        :param widths: list of ints (width per dimension)
        :returns: list of F scores, one for each variable
        """
        total = counts.sum().item()
        ceil = (total + 1) // 2
        num_dims = len(widths)
        results = []

        for t in range(num_dims):
            bounds = list(widths)
            bounds[t] = 1  # always 0

            current_map = {0: 0}  # current_map[a] = b means that for partial sum a of class 0 the best
            # achievable partial sum for class 1 is b
            values = [0] * num_dims

            while True:
                # Build conditional vector along target dimension t
                conditional = []
                for x in range(widths[t]):
                    values[t] = x
                    conditional.append(counts[PrivateMutualInformation._encode(values, widths)])
                values[t] = 0  # reset

                # Update DP map
                next_map = defaultdict(int)
                for a, b in current_map.items():
                    a_new = min(a + int(conditional[0]), ceil)
                    b_new = min(b + int(conditional[1]), ceil)

                    next_map[a_new] = max(next_map.get(a_new, 0), b)
                    next_map[a] = max(next_map.get(a, 0), b_new)
                current_map = next_map

                if not PrivateMutualInformation._inc(values, bounds):
                    break

            # Compute best score
            best = -total
            for a, b in current_map.items():
                best = max(best, a + b - total)
            results.append(best / total)

        return results

    @staticmethod
    def _getF_multiclass(counts_2d, ceil=None):
        """
        Generalized version of getF for multiclass targets.
        :param counts_2d: 2D NumPy array where counts[i, j] is the count of feature value i and class j
        :param ceil: Optional ceil for DP constraint. If None, computed as (total + 1) // 2
        :return: averaged F-score across all classes
        """
        num_feat_vals, num_classes = counts_2d.shape
        total = int(np.sum(counts_2d))
        if ceil is None:
            ceil = (total + 1) // 2

        # Store the F-score for each class vs the rest
        class_fs = []

        for target_class in range(num_classes):
            # Create a 2D version where:
            #   class 0 = current class
            #   class 1 = all others combined
            collapsed_counts = np.zeros((num_feat_vals, 2))
            for i in range(num_feat_vals):
                collapsed_counts[i, 0] = counts_2d[i, target_class]
                collapsed_counts[i, 1] = np.sum(counts_2d[i]) - collapsed_counts[i, 0]

            # Apply original F logic on this binary-like table
            current_map = {0: 0}
            for i in range(num_feat_vals):
                val0 = int(collapsed_counts[i, 0])
                val1 = int(collapsed_counts[i, 1])
                next_map = defaultdict(int)
                for a, b in current_map.items():
                    a_new = min(a + val0, ceil)
                    b_new = min(b + val1, ceil)
                    next_map[a_new] = max(next_map[a_new], b)
                    next_map[a] = max(next_map[a], b_new)
                current_map = next_map

            best = -total
            for a, b in current_map.items():
                best = max(best, a + b - total)
            f_score = best / total
            class_fs.append(f_score)

        # Return the average score across all one-vs-rest binary tasks
        return np.mean(class_fs)

    @staticmethod
    def _mutual_info(joint):
        n = len(joint)
        m = len(joint[0]) if n > 0 else 0

        nsum = [0.0] * n
        msum = [0.0] * m
        total = 0.0

        for i in range(n):
            for j in range(m):
                val = joint[i][j]
                nsum[i] += val
                msum[j] += val
                total += val

        result = 0.0
        for i in range(n):
            for j in range(m):
                if joint[i][j] > 0:
                    p_ij = joint[i][j] / total
                    expected = (nsum[i] * msum[j]) / (total * total)
                    result += p_ij * math.log2(p_ij / expected)
        return result

    @staticmethod
    def _getI(counts, widths):
        """
        Translated version of table::getI from C++
        :param counts: flattened list of counts
        :param widths: number of categories per dimension
        :return: list of mutual information scores (1 per dimension)
        """
        results = []
        num_dims = len(widths)

        for t in range(num_dims):
            bounds = list(widths)
            bounds[t] = 1  # always 0

            values = [0] * num_dims
            joint = []

            while True:
                conditional = []
                for x in range(widths[t]):
                    values[t] = x
                    idx = PrivateMutualInformation._encode(values, widths)
                    conditional.append(counts[idx])
                values[t] = 0  # reset

                joint.append(conditional)

                if not PrivateMutualInformation._inc(values, bounds):
                    break

            results.append(PrivateMutualInformation._mutual_info(joint))

        return results
