import math
import time

import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


class NonPrivateMutualInformation:

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

        # mi = mutual_info_score(column_1_encoded, column_2_encoded)
        num_unique_column_1 = len(np.unique(column_1_encoded))
        num_unique_column_2 = len(np.unique(column_2_encoded))
        counts = np.zeros((num_unique_column_1, num_unique_column_2))
        for s, inc in zip(column_1_encoded, column_2_encoded):
            counts[s, inc] += 1
        mi = self._getI(counts.flatten(), list(counts.shape))[0]

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Non-Differentially Private Mutual Information between '{column_name_1}' and '{column_name_2}': "
              f"{mi:.4f}. Calculation took {elapsed_time:.3f} seconds.")

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
                    idx = NonPrivateMutualInformation._encode(values, widths)
                    conditional.append(counts[idx])
                values[t] = 0  # reset

                joint.append(conditional)

                if not NonPrivateMutualInformation._inc(values, bounds):
                    break

            results.append(NonPrivateMutualInformation._mutual_info(joint))

        return results

    # def calculate_in_c(self):
    #     def write_domain_file(df, discrete_cols=None):
    #         with open("adult.domain", 'w') as f:
    #             for col in df.columns:
    #                 if discrete_cols is None:
    #                     is_discrete = df[col].dtype == 'object'
    #                 else:
    #                     is_discrete = col in discrete_cols
    #
    #                 if is_discrete:
    #                     unique_vals = sorted(df[col].dropna().unique())
    #                     line = "D " + " ".join(str(v) for v in unique_vals)
    #                 else:
    #                     min_val = df[col].min()
    #                     max_val = df[col].max()
    #                     line = f"C {min_val} {max_val}"
    #                 f.write(line + "\n")
    #
    #     def write_dat_file(df):
    #         with open("adult.dat", 'w') as f:
    #             for _, row in df.iterrows():
    #                 line = "\t".join(str(v) for v in row)
    #                 f.write(line + "\n")
    #
    #     print("Computing mutual information between 'sex' and 'income' treating sex as non-private in C++")
    #
    #     write_domain_file(self.adult)
    #     write_dat_file(self.adult)
    #     args = ['0']
    #
    #     start_time = time.time()
    #
    #     result = subprocess.run(['./mutual_information'] + args, capture_output=True, text=True)
    #
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #
    #     if result.returncode == 0:
    #         print(f"Non-Differentially Private Mutual Information between 'sex' and 'income': "
    #               f"{result.stdout.strip():.4f}. "
    #               f"Calculation took {elapsed_time:.3f} seconds.")
    #     else:
    #         print("Error:", result.stderr)
