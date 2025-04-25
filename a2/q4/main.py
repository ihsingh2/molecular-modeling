import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

np.random.seed(42)

def find_coordinate_distance(r_i, r_j, L):
    dr = r_i - r_j
    for kdx in range(3):
        # https://en.wikipedia.org/wiki/Periodic_boundary_conditions#(A)_Restrict_particle_coordinates_to_the_simulation_box
        if (dr[kdx] > 0.5 * L):
            dr[kdx] -= L
        elif (dr[kdx] <= -0.5 * L):
            dr[kdx] += L
    return dr

def find_distance(r_i, r_j, L):
    return np.linalg.norm(find_coordinate_distance(r_i, r_j, L))

def pairwise_distances(r1, r2, L):
    distances = np.empty((len(r1), len(r2)))
    for idx in range(len(r1)):
        for jdx in range(len(r2)):
            distances[idx, jdx] = find_distance(r1[idx], r2[jdx], L)
    return distances

def constraint_satisfied(r, candidate, L, threshold=3):
    return (pairwise_distances(r, candidate.reshape(1, 3), L) >= threshold).all()

def check_all_constraints(r, L, threshold=3):
    distances = pairwise_distances(r, r, L)
    np.fill_diagonal(distances, threshold + 1)
    return (distances >= threshold).all()

def potential_energy(r, eps, sigma, L):
    energy = 0.0
    distances = pairwise_distances(r, r, L)
    for idx in range(len(r) - 1):
        for jdx in range(idx + 1, len(r)):
            energy += 4 * eps * ((sigma / distances[idx, jdx]) ** 12 - (sigma / distances[idx, jdx]) ** 6)
    return energy

N = 108
L = 18.7        # Armstrong
eps = 0.238     # kcal/mol
sigma = 3.4     # Armstrong

k_B = 1.98e-3   # kcal/mol.K
T = 300
m = 10

step_size = 1e-2
num_opt_steps = 1000

r = np.array([], dtype=np.float64).reshape(0, 3)

for idx in tqdm(range(N), desc='Generating particles'):
    candidate = np.random.random(3) * L
    while not constraint_satisfied(r, candidate, L):
        # print(f'Resmapling for particle {idx + 1}')
        candidate = np.random.random(3) * L
    r = np.concatenate((r, [candidate]))
assert check_all_constraints(r, L)

distances = pairwise_distances(r, r, L)
distances = np.triu(distances)
distances = distances[~np.isclose(distances, 0)]
assert len(distances) == N * (N - 1) / 2

std = np.sqrt(3 * N * k_B * T * m)
p = np.random.normal(0, std, size=(N, 3))

plt.hist(distances, bins=15)
plt.axvline(x=3, color='red', linestyle='--', label='Minimum Permissible Distance')
plt.axvline(x=np.sqrt(3) * L / 2, color='green', linestyle='--', label='Maximum Permissible Distance')
plt.title('Histogram of Initial Pairwise Distances')
plt.ylabel('Frequency')
plt.xlabel('Distance')
plt.grid(linewidth=0.35)
plt.legend(loc='upper right')
plt.savefig('initial-distances.png', bbox_inches='tight')
plt.close()
plt.clf()

plt.hist(p.flatten(), bins=15)
plt.title('Histogram of Initial Axiswise Momenta')
plt.ylabel('Frequency')
plt.xlabel('Momenta')
plt.grid(linewidth=0.35)
plt.savefig('initial-momenta.png', bbox_inches='tight')
plt.close()
plt.clf()

plt.hist(np.linalg.norm(p, axis=1), bins=15)
plt.title('Histogram of Magnitude of Initial Momenta')
plt.ylabel('Frequency')
plt.xlabel('Momenta')
plt.grid(linewidth=0.35)
plt.savefig('initial-momenta-magnitude.png', bbox_inches='tight')
plt.close()
plt.clf()

start_time = time.time()
U_curr = potential_energy(r, eps, sigma, L)
U = [U_curr, ]
for step in range(num_opt_steps):
    while True:
        r_candidate = r + step_size * np.random.randn(*r.shape)
        U_candidate = potential_energy(r_candidate, eps, sigma, L)
        U_delta = U_candidate - U_curr
        if U_delta < 0 or np.random.random() <= np.exp(- U_delta / (k_B * T)):
            r = r_candidate
            U_curr = U_candidate
            U.append(U_curr)
            break
    print(f'Step {step:03d} | Potential Energy: {U_curr:.2f}')
end_time = time.time()
print(f'Time Elapsed: {end_time - start_time:.2f} seconds')

plt.plot(U)
plt.title(f'Change in Potential Energy with Monte Carlo Simulation (T={T})')
plt.ylabel('Potential Energy')
plt.xlabel('Steps')
plt.xlim(0, len(U) - 1)
plt.grid(linewidth=0.35)
plt.savefig(f'monte-carlo-{T}.png', bbox_inches='tight')
plt.close()
plt.clf()
