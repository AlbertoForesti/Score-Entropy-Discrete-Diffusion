import numpy as np
from scipy.stats import bernoulli
from scipy.stats._multivariate import multi_rv_frozen

class XORRandomVariable(multi_rv_frozen):
    def __init__(self, p, n_minus_1):
        super().__init__()
        self.p = p
        self.n_minus_1 = n_minus_1

    def rvs(self, size=None, random_state=None):
        # Generate the initial array of shape (size, n-1)
        X = bernoulli.rvs(p=self.p, size=(size, self.n_minus_1), random_state=random_state)

        # Generate the Y array of shape (size, 1)
        Y = bernoulli.rvs(p=self.p, size=(size, 1), random_state=random_state)

        # Compute the XOR of all elements along the second dimension for each sample
        X_xor = np.concatenate([X, Y], axis=1)
        X_xor = np.bitwise_xor.reduce(X_xor, axis=1, keepdims=True)

        # Append the XOR result as a new column to the original array
        X = np.concatenate([X, X_xor], axis=1)
        
        return X, Y

class IsingLoaderRandomVariable:

    def __init__(self, path):
        if path.endswith(".npz"):
            self.values = np.load(path)["arr_0"]
        else:
            self.values = np.load(path)
        self.values[self.values == -1] = 0
    
    def rvs(self, size=None, random_state=None):
        if len(self.values) < size:
            raise ValueError(f"Number of samples requested is greater than the number of samples in the dataset ({len(self.values)}).")
        else:
            self.values = self.values[:size]
        return self.values[:size]