# Re-import libraries after reset
import numpy as np
import matplotlib.pyplot as plt

# Reload data generation process and visualization setup
np.random.seed(1)
tau = 60
bins = np.arange(-0.5, 256.5, 1)

# Define the function g
g = lambda x: np.exp(-((x / 255 - 0.2) ** 2 + 0.01) / 0.05 * (x < tau) - (x / 255 - 0.7) ** 2 / 0.06 * (x >= tau) + np.random.normal(1, 5, len(x)))

# Generate samples using MCMC
mc = [np.random.randint(0, 256)]
for _ in range(10000):
    x_temp = np.random.randint(0, 256)
    u = np.random.rand()
    p_acc = min(1, g(np.array([x_temp]))[0] / g(np.array([mc[-1]]))[0])
    mc.append(x_temp if u < p_acc else mc[-1])

# Calculate histogram data
x = np.array(mc)
counts, edges = np.histogram(x, bins=bins)
midpoints = (edges[:-1] + edges[1:]) / 2

# Separate data for coloring
below_tau = midpoints[midpoints < tau]
above_tau = midpoints[midpoints >= tau]
counts_below_tau = counts[midpoints < tau]
counts_above_tau = counts[midpoints >= tau]

# Convert counts to density (frequency / total samples)
density_below_tau = counts_below_tau / len(mc)
density_above_tau = counts_above_tau / len(mc)

# Plot the histogram with density on the y-axis
plt.figure(figsize=(10, 6))
plt.bar(below_tau, density_below_tau, width=1, color='lightsteelblue', label=r"$x_i < \tau$")
plt.bar(above_tau, density_above_tau, width=1, color='steelblue', label=r"$x_i \geq \tau$")
plt.axvline(x=tau, color='red', linestyle='--', linewidth=2, label=r"$\tau = 60$")

# Enhance aesthetics
plt.xlabel(r"$x_i$", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Remove unnecessary empty space
plt.xlim(0, 255)

# Display the plot
plt.tight_layout()
plt.show()
