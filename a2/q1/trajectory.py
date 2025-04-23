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

for _ in tqdm(range(num_steps), desc='Generating trajectory'):
    x = X[-1]
    p = P[-1]
    X.append(x + (p / m) * dt - 0.5 * (k * x / m) * (dt ** 2))
    P.append(p - k * x * dt - 0.5 * (k * p / m) * (dt ** 2))

print(f'Initial Energy: {hamiltonian(X[0], P[0], k, m):.2f}')
print(f'Final Energy: {hamiltonian(X[-1], P[-1], k, m):.2f}')

plt.plot(X, P)
plt.title('Trajectory in the Phase Space')
plt.ylabel('Momentum')
plt.xlabel('Position')
plt.savefig('trajectory.png', bbox_inches='tight')
plt.close()
plt.clf()
