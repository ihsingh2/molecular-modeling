import matplotlib.pyplot as plt
import numpy as np
import time
from numba import njit, prange
from tqdm import tqdm

np.random.seed(42)

def get_coordinates_from_pdb(pdb_file):
    coordinates = []
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coordinates.append([x, y, z])
                    except ValueError:
                        print(f"Warning: Could not parse coordinate from line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File not found: {pdb_file}")
        exit()
    return np.array(coordinates)

@njit
def find_coordinate_distance(r_i, r_j, L):
    return r_i - r_j

@njit
def find_distance(r_i, r_j, L):
    return np.linalg.norm(find_coordinate_distance(r_i, r_j, L))

@njit(parallel=True)
def pairwise_distances(r1, r2, L):
    distances = np.empty((len(r1), len(r2)))
    for idx in prange(len(r1)):
        for jdx in range(len(r2)):
            distances[idx, jdx] = find_distance(r1[idx], r2[jdx], L)
    return distances

@njit
def constraint_satisfied(r, candidate, L, threshold=3):
    return (pairwise_distances(r, candidate.reshape(1, 3), L) >= threshold).all()

@njit
def check_all_constraints(r, L, threshold=3):
    distances = pairwise_distances(r, r, L)
    np.fill_diagonal(distances, threshold + 1)
    return (distances >= threshold).all()

@njit
def potential_energy(r, eps, sigma, L, l_0, theta_0):
    energy = 0.0
    distances = pairwise_distances(r, r, L)
    for idx in range(0, len(r), 3):
        energy += 1.389e3 * q_H * q_O / (distances[idx, idx + 1] + 1e-6)
        energy += 1.389e3 * q_H * q_O / (distances[idx, idx + 2] + 1e-6)
        energy += 1.389e3 * q_H * q_H / (distances[idx + 1, idx + 2] + 1e-6)
        energy += 1/2 * (distances[idx, idx + 1] - l_0) ** 2
        energy += 1/2 * (distances[idx, idx + 2] - l_0) ** 2
        # v1 = find_coordinate_distance(r[idx + 1], r[idx], L)
        # v2 = find_coordinate_distance(r[idx + 2], r[idx], L)
        # theta = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1., 1.)))
        # energy += 1/2 * (theta - theta_0) ** 2
        for jdx in range(idx + 3, len(r), 3):
            energy += 4 * eps * ((sigma / (distances[idx, jdx] + 1e-6)) ** 12 - (sigma / (distances[idx, jdx] + 1e-6)) ** 6)
            energy += 1.389e3 * q_O * q_O / (distances[idx, jdx] + 1e-6)
            energy += 1.389e3 * q_H * q_O / (distances[idx, jdx + 1] + 1e-6)
            energy += 1.389e3 * q_H * q_O / (distances[idx, jdx + 2] + 1e-6)
    return energy

@njit
def potential_energy_gradient(r, eps, sigma, L, l_0, theta_0):
    gradient = np.zeros_like(r)
    for idx in range(0, len(r), 3):
        gradient[idx] -= (1.389e3 * q_H * q_O / (find_distance(r[idx], r[idx + 1], L) ** 3 + 1e-6)) * find_coordinate_distance(r[idx], r[idx + 1], L)
        gradient[idx + 1] += (1.389e3 * q_H * q_O / (find_distance(r[idx], r[idx + 1], L) ** 3 + 1e-6)) * find_coordinate_distance(r[idx], r[idx + 1], L)
        gradient[idx] -= (1.389e3 * q_H * q_O / (find_distance(r[idx], r[idx + 2], L) ** 3 + 1e-6)) * find_coordinate_distance(r[idx], r[idx + 2], L)
        gradient[idx + 2] += (1.389e3 * q_H * q_O / (find_distance(r[idx], r[idx + 2], L) ** 3 + 1e-6)) * find_coordinate_distance(r[idx], r[idx + 2], L)
        gradient[idx + 1] -= (1.389e3 * q_H * q_H / (find_distance(r[idx + 1], r[idx + 2], L) ** 3 + 1e-6)) * find_coordinate_distance(r[idx + 1], r[idx + 2], L)
        gradient[idx + 2] += (1.389e3 * q_H * q_H / (find_distance(r[idx + 1], r[idx + 2], L) ** 3 + 1e-6)) * find_coordinate_distance(r[idx + 1], r[idx + 2], L)
        gradient[idx] += find_coordinate_distance(r[idx], r[idx + 1], L)
        gradient[idx + 1] -= find_coordinate_distance(r[idx], r[idx + 1], L)
        gradient[idx] += find_coordinate_distance(r[idx], r[idx + 2], L)
        gradient[idx + 2] -= find_coordinate_distance(r[idx], r[idx + 2], L)
        for jdx in range(0, len(r), 3):
            if idx != jdx:
                distance = find_distance(r[idx], r[jdx], L) + 1e-6
                difference = find_coordinate_distance(r[idx], r[jdx], L)
                gradient[idx] -= (48 * eps * sigma**12 / distance ** 13) * (difference / distance)
                gradient[idx] += (24 * eps * sigma**6 / distance ** 7) * (difference / distance)
                gradient[idx] -= (1.389e3 * q_O * q_O / (find_distance(r[idx], r[jdx], L) ** 3 + 1e-6)) * find_coordinate_distance(r[idx], r[jdx], L)
                gradient[idx] -= (1.389e3 * q_H * q_O / (find_distance(r[idx], r[jdx + 1], L) ** 3 + 1e-6)) * find_coordinate_distance(r[idx], r[jdx + 1], L)
                gradient[jdx + 1] += (1.389e3 * q_H * q_O / (find_distance(r[idx], r[jdx + 1], L) ** 3 + 1e-6)) * find_coordinate_distance(r[idx], r[jdx + 1], L)
                gradient[idx] -= (1.389e3 * q_H * q_O / (find_distance(r[idx], r[jdx + 2], L) ** 3 + 1e-6)) * find_coordinate_distance(r[idx], r[jdx + 2], L)
                gradient[jdx + 2] += (1.389e3 * q_H * q_O / (find_distance(r[idx], r[jdx + 2], L) ** 3 + 1e-6)) * find_coordinate_distance(r[idx], r[jdx + 2], L)
    return np.clip(gradient, -0.5, 0.5)

N = 972
L = 13.5            # Angstrom
eps = 0.650         # kJ/mol
sigma = 3.166       # Angstrom
q_H = 0.4236
q_O = -0.8476
l_0 = 1.0000        # Angstrom
theta_0 = 109.47    # Degrees

k_B = 8.3e-3        # kJ/mol.K
T = 300
m = 10
Q = 0.05
alpha = np.full((N, 1), 0.5)

dt = 0.001
num_steps = 1000

r = get_coordinates_from_pdb('ICES-hex.pdb')
std = np.sqrt(k_B * T * m)
p = np.random.normal(0, std, size=(N, 3))

R = [r, ]
P = [p, ]
U = [potential_energy(r, eps, sigma, L, l_0, theta_0)]
KE = [np.sum(p ** 2) / (2 * m)]
E = [U[-1] + KE[-1]]

for _ in tqdm(range(num_steps), desc='Generating trajectory'):
    r = R[-1]
    p = P[-1]
    alpha = alpha + ((1 / Q) * ((np.sum(p ** 2, axis=1, keepdims=True) / m) - (3 * k_B * T))) * dt
    R.append(np.clip(r + (p / m) * dt, -L, L))
    P.append(p - (potential_energy_gradient(r, eps, sigma, L, l_0, theta_0) + alpha * p) * dt)
    U.append(potential_energy(R[-1], eps, sigma, L, l_0, theta_0))
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
