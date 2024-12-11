import numpy as np
from scipy.stats import bernoulli
from scipy.stats._multivariate import multi_rv_frozen
from tqdm import tqdm

class Ising(multi_rv_frozen):

    def __init__(self, lattice_width=None, temperature=None, mc_steps=None, path=None):
        super().__init__()

        assert path is not None or (lattice_width is not None and temperature is not None), "Either path or lattice_width and temperature must be provided"
        
        if path is not None:
            self.path = path
        else:
            self.l = lattice_width
            self.t = temperature
            if mc_steps is not None:
                self.mc_steps = mc_steps
            else:
                self.mc_steps = 1000*self.l**2
    
    def rvs(self, size=None, random_state=None):
        if self.path is not None:
            lattices = np.load(self.path)
            if len(lattices) <= size:
                # Choose randomly size lattices from the loaded ones without replacement
                lattices = np.random.choice(lattices, size=size, replace=False)
                return lattices
        spins = np.random.choice([1, -1], size=(size, self.l, self.l))
        lattices = self.sweep(spins)
        return lattices

    def sweep(self, spins):
        n_samples = spins.shape[0]
        sample_indeces = np.arange(n_samples)
        for _ in tqdm(range(self.mc_steps)):
            row = np.random.randint(0, self.l, size=n_samples)
            column = np.random.randint(0, self.l, size=n_samples)

            top = np.roll(spins, 1, axis=1)[sample_indeces,row,column]
            bottom = np.roll(spins, -1, axis=1)[sample_indeces,row,column]
            left = np.roll(spins, 1, axis=2)[sample_indeces,row,column]
            right = np.roll(spins, -1, axis=2)[sample_indeces,row,column]

            selected_to_move = spins[sample_indeces,row,column]
            dE = 2*selected_to_move*(top + bottom + left + right)
            move_chance = np.exp(-dE/self.t)
            flip_indices = np.random.rand(n_samples) < move_chance
            
            row = row[flip_indices]
            column = column[flip_indices]
            spins[flip_indices, row, column] *= -1
        return spins
    
    @property
    def entropy(self):
        pass