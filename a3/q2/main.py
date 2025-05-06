import matplotlib.pyplot as plt
import numpy as np
import time
from numba import njit, prange
from tqdm import tqdm

np.random.seed(42)

@njit
def find_coordinate_distance(r_i, r_j, h):
    f_i = np.linalg.solve(h, r_i)
    f_j = np.linalg.solve(h, r_j)
    df = f_i - f_j
    df -= np.round(df)
    return h.dot(df)

@njit
def find_distance(r_i, r_j, h):
    return np.linalg.norm(find_coordinate_distance(r_i, r_j, h))

@njit(parallel=True)
def pairwise_distances(r1, r2, h):
    distances = np.empty((len(r1), len(r2)))
    for idx in prange(len(r1)):
        for jdx in range(len(r2)):
            distances[idx, jdx] = find_distance(r1[idx], r2[jdx], h)
    return distances

@njit
def constraint_satisfied(r, candidate, h, threshold=3):
    return (pairwise_distances(r, candidate.reshape(1, 3), h) >= threshold).all()

@njit
def check_all_constraints(r, h, threshold=3):
    distances = pairwise_distances(r, r, h)
    np.fill_diagonal(distances, threshold + 1)
    return (distances >= threshold).all()

@njit
def potential_energy(r, eps, sigma, L):
    # Lennard Jones Potential
    energy = 0.0
    distances = pairwise_distances(r, r, L)
    for idx in range(len(r) - 1):
        for jdx in range(idx + 1, len(r)):
            energy += 4 * eps * ((sigma / distances[idx, jdx]) ** 12 - (sigma / distances[idx, jdx]) ** 6)
    return energy

@njit(parallel=True)
def potential_energy_gradient(r, eps, sigma, h):
    gradient = np.zeros_like(r)
    for idx in prange(len(r)):
        for jdx in range(len(r)):
            if idx != jdx:
                distance = find_distance(r[idx], r[jdx], h)
                difference = find_coordinate_distance(r[idx], r[jdx], h)
                gradient[idx] -= (48 * eps * sigma**12 / distance ** 13) * (difference / distance)
                gradient[idx] += (24 * eps * sigma**6 / distance ** 7) * (difference / distance)
    return gradient

@njit
def pressure_tensor(r, p, h, eps, sigma, v):
    # https://manual.gromacs.org/documentation/2019-rc1/reference-manual/algorithms/molecular-dynamics.html#kinetic-energy-and-temperature
    kinetic = np.zeros((3, 3))
    for i in range(len(r)):
        kinetic += np.outer(p[i], p[i])
    kinetic /= m

    vir = np.zeros((3,3))
    for i in range(len(r) - 1):
        for j in range(i + 1, len(r)):
            rij = find_coordinate_distance(r[i], r[j], h)
            dist = np.linalg.norm(rij)
            fij = 48 * eps * (sigma ** 12 / dist**14 - 0.5 * sigma ** 6 / dist ** 8) * rij
            vir += np.outer(rij, fij)
    return (kinetic + vir) / v

N = 108
L = 18.7        # Angstrom
eps = 0.238     # kcal/mol
sigma = 3.4     # Angstrom

k_B = 1.98e-3   # kcal/mol.K
T = 300
m = 10

dt = 0.005
step_size = 1e-2
num_steps = 1000
num_opt_steps = 100

Q = 0.5
alpha = 0.0

W = 1.0
P_target = 0.0025

h = np.eye(3) * L
G = np.zeros((3, 3))

r = np.array([], dtype=np.float64).reshape(0, 3)

for idx in tqdm(range(N), desc='Generating particles'):
    candidate = np.random.random(3) * L
    while not constraint_satisfied(r, candidate, h):
        # print(f'Resmapling for particle {idx + 1}')
        candidate = np.random.random(3) * L
    r = np.concatenate((r, [candidate]))
assert check_all_constraints(r, h)

distances = pairwise_distances(r, r, h)
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
    U_curr = potential_energy(r, eps, sigma, h)
    U_grad = potential_energy_gradient(r, eps, sigma, h)
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
V = [np.linalg.det(h), ]
U = [potential_energy(r, eps, sigma, h)]
KE = [np.sum(p ** 2) / (2 * m)]
E = [U[-1] + KE[-1]]
Pr = [np.trace(pressure_tensor(r, p, h, eps, sigma, np.linalg.det(h))) / 3, ]

for _ in tqdm(range(num_steps), desc='Generating trajectory'):

    v = np.linalg.det(h)
    alpha += 0.5 * dt * ( (np.sum(p**2)/m) - 3*N*k_B*T ) / Q
    gradU = potential_energy_gradient(r, eps, sigma, h)
    P_inst = pressure_tensor(r, p, h, eps, sigma, v)
    p -= (0.5 * dt * (gradU + p.dot(G.dot(np.linalg.inv(h))) + alpha * p))
    G -= (0.5 * dt * (P_inst - np.eye(3) * P_target))

    h += (dt * G / W)
    p -= (dt * p.dot(G.dot(np.linalg.inv(h))))
    p *= np.exp(-alpha * dt)
    r += ((dt * (p / m)) + (dt * r.dot(G.dot(np.linalg.inv(h)))))

    v = np.linalg.det(h)
    gradU = potential_energy_gradient(r, eps, sigma, h)
    P_inst = pressure_tensor(r, p, h, eps, sigma, v)
    p -= (0.5 * dt * (gradU + p.dot(G.dot(np.linalg.inv(h))) + alpha * p))
    G -= (0.5 * dt * (P_inst - np.eye(3) * P_target))
    alpha += (0.5 * dt * ((np.sum(p**2) / m) - 3 * N * k_B * T) / Q)

    R.append(r)
    P.append(p)
    V.append(v)
    U.append(potential_energy(R[-1], eps, sigma, h))
    KE.append(np.sum(P[-1] ** 2) / (2 * m))
    E.append(U[-1] + KE[-1])
    Pr.append(np.trace(P_inst) / 3)

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
ax = plt.gca()
ax.ticklabel_format(useOffset=False, style='plain')
plt.savefig('total-energy.png', bbox_inches='tight')
plt.close()
plt.clf()

plt.plot(E)
plt.title('Variation in Total Energy with Time')
plt.ylabel('Total Energy')
plt.xlabel('Timestep')
plt.grid(linewidth=0.35)
ax = plt.gca()
ax.ticklabel_format(useOffset=False, style='plain')
plt.savefig('total-energy.png', bbox_inches='tight')
plt.close()
plt.clf()

plt.plot(Pr)
plt.title('Variation in Pressure with Time')
plt.ylabel('Pressure')
plt.xlabel('Timestep')
plt.grid(linewidth=0.35)
ax = plt.gca()
ax.ticklabel_format(useOffset=False, style='plain')
plt.savefig('pressure.png', bbox_inches='tight')
plt.close()
plt.clf()

plt.plot(V)
plt.title('Variation in Volume with Time')
plt.ylabel('Volume')
plt.xlabel('Timestep')
plt.grid(linewidth=0.35)
ax = plt.gca()
ax.ticklabel_format(useOffset=False, style='plain')
plt.savefig('volume.png', bbox_inches='tight')
plt.close()
plt.clf()
