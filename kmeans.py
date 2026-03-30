import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from scipy.interpolate import griddata
import matplotlib.cm as cm

# Set seed for reproducibility (Required for accurate MCQ answers) [cite: 41]
np.random.seed(42)

# 1. Create dense graph (600 nodes) [cite: 13]
sizes = [85, 95, 80, 90, 75, 90, 85]   
p_in = [0.30, 0.28, 0.32, 0.29, 0.31, 0.27, 0.30]
p_out = 0.05
G = nx.stochastic_block_model(sizes, np.full((7, 7), p_out, dtype=float), seed=42)
base_pos = nx.spring_layout(G, seed=42, k=0.11, iterations=300)
nodes = list(G.nodes())
xy = np.array([base_pos[n] for n in nodes])

# Normalize and scale coordinates [cite: 6]
x, y = xy[:, 0] * 100, xy[:, 1] * 100
energy = np.random.randint(10, 101, size=len(nodes))

# 2. Build DataFrame and Cluster (k=7) [cite: 7, 9]
df = pd.DataFrame({"Node": [f"N{n}" for n in nodes], "X": x, "Y": y, "Energy": energy})
k = 7
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df[["X", "Y"]])
centroids = kmeans.cluster_centers_

# --- PRINT RESULTS FOR MCQS [cite: 35, 36] ---
print("\n" + "="*40)
print("CMPG 313 - LAB 2 ANALYSIS RESULTS")
print("="*40)
print("\n[NODE COUNTS PER CLUSTER]")
print(df["Cluster"].value_counts().sort_index())
print("\n[ENERGY STATISTICS PER CLUSTER]")
print(df.groupby("Cluster")["Energy"].agg(['mean', 'min', 'max']))
print("="*40 + "\n")

# 3. PLOT 1: 2D Network [cite: 15]
plt.figure(figsize=(8, 6))
plt.scatter(df["X"], df["Y"], c=df["Cluster"], cmap='tab10', s=35, alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c="black", marker="X", label="Centroids")
plt.title(f"2D Clustered Network Topology (k={k})")
plt.legend()
plt.show() # CLOSE THIS WINDOW TO SEE THE 3D PLOT

# 4. PLOT 2: 3D Energy Surface 
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create grid for smooth surface
grid_x, grid_y = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
grid_z = griddata((df["X"], df["Y"]), df["Energy"], (grid_x, grid_y), method='cubic')

surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cm.viridis, alpha=0.8, antialiased=True)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Energy Level")
ax.set_title("3D Smooth Energy Surface")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Energy")
plt.show()