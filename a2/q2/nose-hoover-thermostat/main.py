import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def hamiltonian(x, p, k, m):
    return (k * x**2) / 2 + p**2 / (2 * m)

k = 10
m = 10
Q = 0.5
alpha = 0.5

k_B = 1.38e-3     # Scaled for Armstrong units
T = 300

dt = 0.001
num_steps = 500000

X = [1, ]
P = [1, ]
KE = [P[0] ** 2 / (2 * m)]

for _ in tqdm(range(num_steps), desc='Generating trajectory'):
    x = X[-1]
    p = P[-1]
    F = - (k * x)
    alpha = alpha + ((1 / Q) * ((p ** 2 / m) - (k_B * T))) * dt
    X.append(x + (p / m) * dt + 0.5 * (-k * x / m) * (dt ** 2))
    P.append(p + (-k * x - alpha * p) * dt)
    KE.append(P[-1] ** 2 / (2 * m))

plt.plot(X, P)
plt.scatter(X[0], P[0], label='Initial Point', s=15, c='red')
plt.title('Trajectory in State Space')
plt.ylabel('Momentum')
plt.xlabel('Position')
plt.grid(linewidth=0.35)
plt.legend(loc='upper right')
plt.savefig('trajectory.png', bbox_inches='tight')
plt.close()
plt.clf()

plt.hist(P[num_steps // 5:], bins=50)
plt.title('Distribution of Momentum in Trajectory')
plt.ylabel('Frequency')
plt.xlabel('Momentum')
plt.grid(linewidth=0.35)
plt.savefig('momentum.png', bbox_inches='tight')
plt.close()
plt.clf()

P_square = np.array(P[num_steps // 5:]) ** 2
plt.hist(P_square[P_square <= np.percentile(P_square, 85)], bins=50)
plt.axvline(x=np.mean(P_square), color='orange', linestyle='--', label=f'Second Moment')
plt.axvline(x=m * k_B * T, color='green', linestyle='--', label=f'Expected Value (mkT)')
plt.title('Distribution of Momentum Squared in Trajectory')
plt.ylabel('Frequency')
plt.xlabel('Momentum Square')
plt.grid(linewidth=0.35)
plt.legend(loc='upper right')
plt.savefig('momentum-square.png', bbox_inches='tight')
plt.close()
plt.clf()

plt.plot(KE[num_steps // 5:])
plt.title('Variation in Kinetic Energy with Time')
plt.ylabel('Kinetic Energy')
plt.xlabel('Timestep')
plt.grid(linewidth=0.35)
plt.savefig('kinetic-energy.png', bbox_inches='tight')
plt.close()
plt.clf()
