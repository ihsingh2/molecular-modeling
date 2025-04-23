import matplotlib.pyplot as plt
from tqdm import tqdm

def hamiltonian(x, p, k, m):
    return (k * x**2) / 2 + p**2 / (2 * m)

k = 10
m = 10

dt = 0.001
num_steps = 250000

X = [15, ]
P = [15, ]
KE = [P[0] ** 2 / (2 * m)]

for _ in tqdm(range(num_steps), desc='Generating trajectory'):
    x = X[-1]
    p = P[-1]
    F = - (k * x)
    alpha = (F * p) / (p * p)
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

plt.plot(KE)
plt.title('Variation in Kinetic Energy with Time')
plt.ylabel('Kinetic Energy')
plt.xlabel('Timestep')
plt.grid(linewidth=0.35)
plt.savefig('kinetic-energy.png', bbox_inches='tight')
plt.close()
plt.clf()
