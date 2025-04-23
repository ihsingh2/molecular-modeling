import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def hamiltonian(x, p, k, m):
    return (k * x**2) / 2 + p**2 / (2 * m)

k = 10
m = 10

dt = 0.001
num_steps = 2500000
grid_size = 1e-3

combinations = {
    'X': [15, 20],
    'P': [15, 20]
}
true_probabilities = []
probabilities = []

for (X_0, P_0) in tqdm([(X_0, P_0) for X_0 in combinations['X'] for P_0 in combinations['P']], desc='Generating trajectories'):
    X = [X_0, ]
    P = [P_0, ]

    for step in range(num_steps):
        x = X[-1]
        p = P[-1]
        X.append(x + (p / m) * dt - 0.5 * (k * x / m) * (dt ** 2))
        P.append(p - k * x * dt - 0.5 * (k * p / m) * (dt ** 2))

    i = np.floor(np.array(X) / grid_size).astype(int)
    j = np.floor(np.array(P) / grid_size).astype(int)
    grid_cells = np.stack((i, j), axis=-1)
    unique_cells, counts = np.unique(grid_cells, axis=0, return_counts=True)
    true_probabilities.append(1 / len(unique_cells))
    probabilities.append(counts / len(grid_cells))

plt.boxplot(probabilities, showmeans=True, label='Median Empirical Probability')
plt.plot(range(1, len(true_probabilities) + 1), true_probabilities, 'bo', label='Uniform Theoretical Probability')
plt.title('Comparision of Empirical and Theoretical Probability Distributions')
plt.ylabel('Probability of an Accessible Microstate')
plt.xlabel('Different Initial Conditions of the System')
plt.legend()
plt.savefig('probability.png', bbox_inches='tight')
plt.close()
plt.clf()
