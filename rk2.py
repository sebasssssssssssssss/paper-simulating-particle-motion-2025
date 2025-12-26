import numpy as np
import matplotlib.pyplot as plt

def rk2_step(x, t, dt, f):
    k1 = f(x, t)
    k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
    return x + dt * k2

# Example ODE: dx/dt = -x, exact solution x(t) = x0 * exp(-t)
def f(x, t):
    return -x

t_end = 5.0
x0    = 1.0

# high‑resolution exact solution
ts_exact = np.linspace(0, t_end, 1000)
xs_exact = x0 * np.exp(-ts_exact)

dt_values = [0.05, 0.5, 1.0]
colors    = ['C1', 'C2', 'C3']

plt.figure(figsize=(8, 5))

# Plot exact solution thicker and solid
plt.plot(ts_exact, xs_exact,
         color='blue', linewidth=2.5, label='Exact')

# Plot RK2 approximations as dotted lines with hollow circles
for dt, c in zip(dt_values, colors):
    n_steps = int(t_end / dt)
    xs = [x0]
    ts = [0.0]
    x, t = x0, 0.0
    for _ in range(n_steps):
        x = rk2_step(x, t, dt, f)
        t += dt
        xs.append(x)
        ts.append(t)

    plt.plot(ts, xs,
             linestyle=':',        # dotted
             marker='o',
             color=c,
             markersize=5,
             markerfacecolor='none',
             label=f'RK2 dt={dt}')

plt.xlabel('Time (S)', fontsize=14)
plt.ylabel('X(t)', fontsize=14)
plt.title('Exact and Approximate Solutions (dx/dt = -x)', fontsize=16)
plt.legend(frameon=True, fontsize=14)
plt.grid(False)
plt.tight_layout()
plt.savefig("figures/rk2.png", dpi=300, bbox_inches='tight')
plt.show()
