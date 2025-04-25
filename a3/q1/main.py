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

def potential_energy_gradient(r, eps, sigma, L):
    gradient = np.zeros_like(r)
    for idx in range(len(r)):
        for jdx in range(len(r)):
            if idx != jdx:
                distance = find_distance(r[idx], r[jdx], L)
                difference = find_coordinate_distance(r[idx], r[jdx], L)
                gradient[idx] -= (48 * eps * sigma**12 / distance ** 13) * (difference / distance)
                gradient[idx] += (24 * eps * sigma**6 / distance ** 7) * (difference / distance)
    return gradient

N = 108
L = 18.7        # Armstrong
eps = 0.238     # kcal/mol
sigma = 3.4     # Armstrong

k_B = 1.98e-3   # kcal/mol.K
T = 300
m = 10
Q = 0.05
alpha = np.full((N, 1), 0.5)

dt = 0.005
step_size = 1e-2
num_steps = 1000
num_opt_steps = 100

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

std = np.sqrt(k_B * T * m)
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
U = []
for step in range(num_opt_steps):
    U_curr = potential_energy(r, eps, sigma, L)
    U_grad = potential_energy_gradient(r, eps, sigma, L)
    U.append(U_curr)
    r -= step_size * (1 - step / (2 * num_opt_steps)) * U_grad
    print(f'Step {step:02d} | Potential Energy: {U_curr:.2f} | Gradient: {np.linalg.norm(U_grad):.2f}')
end_time = time.time()
print(f'Time Elapsed: {end_time - start_time:.2f} seconds')

plt.plot(U)
plt.title('Change in Potential Energy with Steepest Descent')
plt.ylabel('Potential Energy')
plt.xlabel('Optimization Steps')
plt.xlim(0, num_opt_steps)
plt.grid(linewidth=0.35)
plt.savefig('descent.png', bbox_inches='tight')
plt.close()
plt.clf()

R = [r, ]
P = [p, ]
U = [potential_energy(r, eps, sigma, L)]
KE = [np.sum(p ** 2) / (2 * m)]
E = [U[-1] + KE[-1]]

for _ in tqdm(range(num_steps), desc='Generating trajectory'):
    r = R[-1]
    p = P[-1]
    alpha = alpha + ((1 / Q) * ((np.sum(p ** 2, axis=1, keepdims=True) / m) - (3 * k_B * T))) * dt
    R.append(np.clip(r + (p / m) * dt, 0, L))
    P.append(p - (potential_energy_gradient(r, eps, sigma, L) + alpha * p) * dt)
    U.append(potential_energy(R[-1], eps, sigma, L))
    KE.append(np.sum(P[-1] ** 2) / (2 * m))
    E.append(U[-1] + KE[-1])

P_array = np.vstack(P)

plt.hist(P_array.reshape(-1), bins=50)
plt.title('Distribution of Momenta in Trajectory')
plt.ylabel('Frequency')
plt.xlabel('Momenta')
plt.grid(linewidth=0.35)
plt.savefig('momenta.png', bbox_inches='tight')
plt.close()
plt.clf()

P_square = np.sum(P_array ** 2, axis=1)
plt.hist(P_square, bins=50)
plt.axvline(x=np.mean(P_square), color='orange', linestyle='--', label=f'Second Moment')
plt.axvline(x=3 * m * k_B * T, color='green', linestyle='--', label=f'Expected Value (3mkT)')
plt.title('Distribution of Momenta Squared in Trajectory')
plt.ylabel('Frequency')
plt.xlabel('Momenta Square')
plt.grid(linewidth=0.35)
plt.legend(loc='upper right')
plt.savefig('momenta-square.png', bbox_inches='tight')
plt.close()
plt.clf()

plt.plot(U)
plt.title('Variation in Potential Energy with Time')
plt.ylabel('Potential Energy')
plt.xlabel('Timestep')
plt.grid(linewidth=0.35)
plt.savefig('potential-energy.png', bbox_inches='tight')
plt.close()
plt.clf()

plt.plot(KE)
plt.title('Variation in Kinetic Energy with Time')
plt.ylabel('Kinetic Energy')
plt.xlabel('Timestep')
plt.grid(linewidth=0.35)
plt.savefig('kinetic-energy.png', bbox_inches='tight')
plt.close()
plt.clf()

plt.plot(E)
plt.title('Variation in Total Energy with Time')
plt.ylabel('Total Energy')
plt.xlabel('Timestep')
plt.grid(linewidth=0.35)
plt.savefig('total-energy.png', bbox_inches='tight')
plt.close()
plt.clf()

with open('trajectory.xyz', 'w') as f:
    for timestep in range(num_steps + 1):
        r = R[timestep]
        f.write(f"{N}\n")
        f.write(f"Time Step {timestep + 1}\n")
        for _, (x, y, z) in enumerate(r):
            f.write(f'H {x:.4f} {y:.4f} {z:.4f}\n')
