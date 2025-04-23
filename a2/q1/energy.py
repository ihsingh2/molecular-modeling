import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def hamiltonian(x, p, k, m):
    return (k * x**2) / 2 + p**2 / (2 * m)

k = 10
m = 10

dt = 0.001
num_steps = 250000
grid_size = 1e-2

combinations = {
    'X': [10, 20, 30, 40],
    'P': [10, 20, 30, 40]
}
energy = []
num_states = []

for (X_0, P_0) in tqdm([(X_0, P_0) for X_0 in combinations['X'] for P_0 in combinations['P']], desc='Generating trajectories'):
    X = [X_0, ]
    P = [P_0, ]

    for step in range(num_steps):
        x = X[-1]
        p = P[-1]
        X.append(x + (p / m) * dt - 0.5 * (k * x / m) * (dt ** 2))
        P.append(p - k * x * dt - 0.5 * (k * p / m) * (dt ** 2))

    energy.append(hamiltonian(X[0], P[0], k, m))
    i = np.floor(np.array(X) / grid_size).astype(int)
    j = np.floor(np.array(P) / grid_size).astype(int)
    grid_cells = np.stack((i, j), axis=-1)
    unique_cells = np.unique(grid_cells, axis=0)
    num_states.append(len(unique_cells))

plt.plot(energy, num_states)
plt.title('Relation between Energy and Accessible Microstates')
plt.ylabel('Number of Accessible Microstates')
plt.xlabel('Energy')
plt.savefig('energy.png', bbox_inches='tight')
plt.close()
plt.clf()
