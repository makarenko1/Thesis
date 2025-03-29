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

    def calculate(self):
        print("Computing mutual information between 'sex' and 'income' treating them as private")

        # Encode sex and income
        sex_encoded = LabelEncoder().fit_transform(self.adult['sex'])
        income_encoded = LabelEncoder().fit_transform(self.adult['income'])

        # Compute F
        start_time = time.time()  # Record start time

        counts = np.zeros((len(np.unique(sex_encoded)), len(np.unique(income_encoded))))
        for s, inc in zip(sex_encoded, income_encoded):
            counts[s, inc] += 1
        mi = self._getF(counts.flatten(), [2, 2])[0]

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Differentially-Private Mutual Information between 'sex' and 'income': {mi:.4f}. "
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

            current_map = {(0, 0): 0}  # current_map[(a, b)] = val means that for partial sum a of class 0 the best
            # achievable partial sum for class 1 is val, so that val >= b
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
                for (a, b), val in current_map.items():
                    a_new = min(a + int(conditional[0]), ceil)
                    b_new = min(b + int(conditional[1]), ceil)
                    next_map[(a_new, b)] = max(next_map[(a_new, b)], b_new)
                    next_map[(a, b_new)] = max(next_map[(a, b_new)], b)
                current_map = next_map

                if not PrivateMutualInformation._inc(values, bounds):
                    break

            # Compute best score
            best = -total
            for a, b in current_map:
                best = max(best, a + b - total)
            results.append(best / total)

        return results

    @staticmethod
    def _getFBinary(counts):
        """
        Simplified version of getF for 2x2 contingency tables
        :param counts: 2D NumPy array of shape [2, 2], where counts[i, j] is the number of samples with variable of
                       interest = i and label = j
        """
        total = int(counts.sum())
        ceil = (total + 1) // 2
        current_map = {(0, 0): 0}

        for sex_val in [0, 1]:  # variable of interest
            next_map = defaultdict(int)
            for (a, b), val in current_map.items():
                a_new = min(a + int(counts[sex_val, 0]), ceil)
                b_new = min(b + int(counts[sex_val, 1]), ceil)
                next_map[(a_new, b)] = max(next_map[(a_new, b)], b_new)
                next_map[(a, b_new)] = max(next_map[(a, b_new)], b)
            current_map = next_map

        best = -total
        for (a, b) in current_map:
            best = max(best, a + b - total)

        return best / total
