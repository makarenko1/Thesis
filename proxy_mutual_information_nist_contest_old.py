import time

import numpy as np
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


class ProxyMutualInformationNistContest:

    def __init__(self):
        self.adult = fetch_ucirepo(id=2).data['original']
        self.adult.dropna(inplace=True)
        # Preprocess the 'income' column
        self.adult['income'] = self.adult['income'].apply(lambda x: 0 if x.startswith('<=50K') else 1)
        # Preprocess the 'sex' column
        self.adult['sex'] = self.adult['sex'].apply(lambda x: 0 if x.startswith('Male') else 1)

    def calculate(self, column_name_1, column_name_2):
        print(f"Computing mutual information between '{column_name_1}' and '{column_name_2}' treating "
              f"them as private")

        col1 = self.adult[column_name_1]
        col2 = self.adult[column_name_2]

        start_time = time.time()

        mi = self._getF(col1, col2)
        mi += 0.5  # mapping

        elapsed_time = time.time() - start_time
        print(f"Differentially-Private Mutual Information between '{column_name_1}' and '{column_name_2}': {mi:.4f}. "
              f"Calculation took {elapsed_time:.3f} seconds.")

    def measure_1way_marginals(df, C, w_C, sigma):
        """
        Implements Algorithm 1 to measure 1-way marginals with Gaussian noise.

        Parameters:
            df: pandas DataFrame (sensitive dataset D)
            C: list of [col] (collection of 1-way marginals, each as a list)
            w_C: list of floats (weight for each marginal)
            sigma: float (noise scale σ)

        Returns:
            log: list of tuples (w_C_normalized, noisy_marginal, sigma, [col])
        """
        log = []
        w_C = np.array(w_C)
        w_C_normalized = w_C / np.sqrt(np.sum(w_C ** 2))

        for i, [col] in enumerate(C):
            weight = w_C_normalized[i]

            # Encode column to integers
            data = LabelEncoder().fit_transform(df[col])
            domain_size = len(np.unique(data))

            # Compute true marginal
            counts = np.bincount(data, minlength=domain_size)
            true_marginal = counts / counts.sum()

            # Add Gaussian noise
            noise = np.random.normal(loc=0.0, scale=sigma, size=domain_size)
            noisy_marginal = weight * true_marginal + noise

            # Store measurement log
            log.append((weight, noisy_marginal, sigma, [col]))

        return log

    @staticmethod
    def _getF(data1, data2, sigma=0.1):
        """
        Estimate F = -1/2 * ||M_ij(D) - M̂_ij||_1
        Where:
          - M_ij(D) is the true 2-way marginal from data
          - M̂_ij is estimated using independent 1-way marginals + noise
        """

        # Step 1: Encode data
        x = LabelEncoder().fit_transform(data1)
        y = LabelEncoder().fit_transform(data2)

        x_card = len(np.unique(x))
        y_card = len(np.unique(y))
        n = len(x)

        # Step 2: Compute noisy 1-way marginals (simulate DP noise)
        x_counts = np.bincount(x, minlength=x_card)
        y_counts = np.bincount(y, minlength=y_card)

        x_marginal = x_counts / n + np.random.normal(0, sigma, size=x_card)
        y_marginal = y_counts / n + np.random.normal(0, sigma, size=y_card)

        x_marginal = np.clip(x_marginal, 0, None)
        y_marginal = np.clip(y_marginal, 0, None)

        x_marginal /= x_marginal.sum()
        y_marginal /= y_marginal.sum()

        # Step 3: Estimate 2-way marginal assuming independence
        estimated_joint = np.outer(x_marginal, y_marginal)  # P(x)P(y)

        # Step 4: Compute empirical joint
        true_counts = np.zeros((x_card, y_card))
        for xi, yi in zip(x, y):
            true_counts[xi, yi] += 1
        true_joint = true_counts / n

        # Step 5: Compute F as L1 distance
        return np.abs(true_joint - estimated_joint).sum()
