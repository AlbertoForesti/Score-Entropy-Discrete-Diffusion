import numpy as np
import gzip
import os
from scipy.stats import bernoulli
from scipy.stats._multivariate import multi_rv_frozen
from datasets import load_dataset

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

class NumpyLoaderRandomVariable:

    def __init__(self, path, p_random=0):
        if path.endswith(".npz"):
            self.values = np.load(path)["arr_0"]
        elif path.endswith(".gz"):
            with gzip.open(path, 'rb') as f:
                self.values = np.load(f)
        else:
            self.values = np.load(path)
        self.p_random = p_random
    
    def rvs(self, size=None, random_state=None):
        if len(self.values) < size:
            raise ValueError(f"Number of samples requested is greater than the number of samples in the dataset ({len(self.values)}).")
        else:
            self.values = self.values[:size]
        data = self.values[:size]
        X = data[:,:-1]
        Y = data[:,-1]
        Y = Y.reshape(-1,1)
        if self.p_random > 0:
            Y_random = np.random.randint(np.min(data[:,-1]), np.max(data[:,-1])+1, size=Y.shape)
            random_indices = np.random.choice(len(data), int(len(data) * self.p_random), replace=False)
            Y[random_indices] = Y_random[random_indices]

        return X, Y

class GenomicBenchmarkRV:

    def __init__(self, path, split='train', p_random=0):
        self.dataset = load_dataset(path)[split]
        self.nucleotide_to_int = {
            'A': 0,
            'C': 1,
            'G': 2,
            'T': 3,
            'N': 4,
        }
        self.values = np.array([np.array([self.nucleotide_to_int[nuc] for nuc in seq]) for seq in self.dataset['seq']])
        self.values = np.concatenate([self.values, np.array(self.dataset['label']).reshape(-1, 1)], axis=1)
        self.p_random = p_random
        if self.p_random > 0:
            random_indices = np.random.choice(len(self.values), int(len(self.values) * self.p_random), replace=False)
            self.values[random_indices, -1] = np.random.randint(np.min(self.values[:,-1]), np.max(self.values[:,-1])+1, size=len(random_indices))
    
    def rvs(self, size=None, random_state=None):
        if len(self.values) < size:
            raise ValueError(f"Number of samples requested is greater than the number of samples in the dataset ({len(self.values)}).")
        else:
            self.values = self.values[:size]
        data = self.values[:size]
        X = data[:,:-1]
        Y = data[:,-1]
        Y = Y.reshape(-1,1)
        return X, Y