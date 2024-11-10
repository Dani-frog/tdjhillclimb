import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Initialize global variables
highest_point = 0  
max_peak_count = 0  
found_peak_count = 0
given_seed = 340
sim_delay_sec = 0.05

# Function to set the seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Generate a random integer between min and max
def generate_random_int(min_val, max_val):
    return np.random.randint(min_val, max_val)

# Terrain generation with integer-based Gaussian peaks and valleys
def generate_terrain(x, y):
    num_peaks = generate_random_int(15, 30)
    terrain = np.zeros_like(x, dtype=int) + 8

    for _ in range(num_peaks):
        peak_x = np.random.uniform(0, 30)
        peak_y = np.random.uniform(0, 30)
        height = generate_random_int(1, 3)
        width = generate_random_int(1, 2)
        terrain += np.round(height * np.exp(-((x - peak_x) ** 2 + (y - peak_y) ** 2) / (2 * width ** 2))).astype(int)

    num_valleys = generate_random_int(5, 10)
    for _ in range(num_valleys):
        valley_x = np.random.uniform(0, 30)
        valley_y = np.random.uniform(0, 30)
        depth = generate_random_int(1, 3)
        terrain -= np.round(depth * np.exp(-((x - valley_x) ** 2 + (y - valley_y) ** 2) / (2 * 1 ** 2))).astype(int)

    return terrain

def find_highest_peak(Z):
    global highest_point
    highest_point = np.max(Z)
    return highest_point

def count_max_peaks(Z, max_height):
    return np.sum(Z == max_height)

# Initialize the plot
def init_plot(X, Y, Z, seed):
    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot_surface(X, Y, Z, cmap='plasma', alpha=0.6, edgecolor='k', linewidth=0.3)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Terrain height")
    ax3d.set_title(f"3D Terrain with Peaks and Valleys (Seed={seed})", color="darkblue", fontsize=14)
    return fig, ax3d

def plot_terrain(ax3d, Z, dot_coords=None):
    for collection in ax3d.collections:
        collection.remove()
    ax3d.plot_surface(X, Y, Z, cmap='plasma', alpha=0.6, edgecolor='k', linewidth=0.3)
    if dot_coords is not None:
        x, y = dot_coords
        x_idx = (np.abs(x_vals - x)).argmin()
        y_idx = (np.abs(y_vals - y)).argmin()
        z = Z[y_idx, x_idx]
        ax3d.scatter(x, y, z, color='black', s=150, label="Hegymászó")
        ax3d.legend(loc="upper left", fontsize=10)
    plt.draw()
    plt.pause(0.1)

# Lépésenkénti megállás a felhasználói bevitelre
def move_climbing_dot():
    current_x = np.random.randint(0, 30)
    current_y = np.random.randint(0, 30)
    
    step_count = 0
    found_peaks = set()

    # Matrix to keep track of visited cells
    visited = np.zeros((30, 30), dtype=bool)
    global max_peak_count
    found_peak_count = 0

    while max_peak_count != found_peak_count:
        visited[current_y, current_x] = True
        current_height = Z[current_y, current_x]
        neighbors = [] 

        # Szomszédok keresése
        if current_y < 29 and not visited[current_y + 1, current_x]:
            neighbors.append(('up', current_x, current_y + 1, Z[current_y + 1, current_x]))
        if current_y > 0 and not visited[current_y - 1, current_x]:
            neighbors.append(('down', current_x, current_y - 1, Z[current_y - 1, current_x]))
        if current_x > 0 and not visited[current_y, current_x - 1]:
            neighbors.append(('left', current_x - 1, current_y, Z[current_y, current_x - 1]))
        if current_x < 29 and not visited[current_y, current_x + 1]:
            neighbors.append(('right', current_x + 1, current_y, Z[current_y, current_x + 1]))

        # Legjobb lépés kiválasztása, ha van
        if neighbors:
            best_move = max(neighbors, key=lambda n: n[3])
            direction, new_x, new_y, new_height = best_move
        else:
            # Ha nincs további lépés, visszatérés a legközelebbi fel nem fedezett cellára
            unvisited = np.argwhere(~visited)
            if len(unvisited) == 0:
                print("Nincs több fel nem fedezett cella")
                break
            new_y, new_x = unvisited[0]
            new_height = Z[new_y, new_x]

        current_x, current_y = new_x, new_y
        step_count += 1

        if new_height == highest_point and (current_x, current_y) not in found_peaks:
            found_peaks.add((current_x, current_y))
            found_peak_count += 1

        print(f"Step {step_count}: Position=({current_x}, {current_y}), Height={new_height}, Global Max={highest_point}, Peaks Found={found_peak_count}")
        plot_terrain(ax3d, Z, dot_coords=(current_x, current_y))
        
        # Megállítás a felhasználói bevitelre
        input("Nyomj Entert a következő lépéshez...")

        if max_peak_count == found_peak_count:
            print(f"All peaks found in {step_count} steps.")
            break

# Set up seed and terrain
seed = given_seed
set_seed(seed)

# Generate terrain and initialize plot
x_vals = np.linspace(0, 30, 30)
y_vals = np.linspace(0, 30, 30)
X, Y = np.meshgrid(x_vals, y_vals)
Z = generate_terrain(X, Y)

highest_point = find_highest_peak(Z)
max_peak_count = count_max_peaks(Z, highest_point)

# Plot initialization and move dot
fig, ax3d = init_plot(X, Y, Z, seed)
move_climbing_dot()
