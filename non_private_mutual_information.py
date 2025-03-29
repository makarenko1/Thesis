import subprocess
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

    def calculate(self):
        print("Computing mutual information between 'sex' and 'income' treating both as non-private")

        # Encode sex and income
        sex_encoded = LabelEncoder().fit_transform(self.adult['sex'])
        income_encoded = LabelEncoder().fit_transform(self.adult['income'])

        # Compute mutual information
        start_time = time.time()  # Record start time

        mi = mutual_info_score(sex_encoded, income_encoded)

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Non-Differentially Private Mutual Information between 'sex' and 'income': {mi:.4f}. "
              f"Calculation took {elapsed_time:.3f} seconds.")

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
