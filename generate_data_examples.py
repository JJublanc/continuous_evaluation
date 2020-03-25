import os
import numpy as np
import pandas as pd

if not "data" in os.listdir("./"):
    os.mkdir("./data")

lambda_a = 0.41
lambda_b = 0.4
lambda_c = 0.6
lambda_d = 0.6001

results_a = np.random.binomial(1, lambda_a, 10000)
results_b = np.random.binomial(1, lambda_b, 10000)
results_c = np.random.binomial(1, lambda_c, 10000)
results_d = np.random.binomial(1, lambda_d, 10000)

data_test = pd.DataFrame({"A": results_a,
                          "B": results_b,
                          "C": results_c,
                          "D": results_d})

data_test.to_csv("./data/data_test_example_0.csv", index=False)
data_test = pd.read_csv("./data/data_test_example_0.csv")