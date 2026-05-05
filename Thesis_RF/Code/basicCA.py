import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

# Grid size
GRID_SIZE = 100
SEED = 42

# Cell states based on proposal
STATE_NON_BURNABLE = 1      # 1: Not possible to burn 
STATE_NOT_YET_BURNING = 2   # 2: Not yet burning 
STATE_IGNITED = 3           # 3: Ignited 
STATE_BLAZING = 4           # 4: Blazing 
STATE_EXTINGUISHED = 5      # 5: Extinguished 

PROB_IGNITION = 0.15 # orig kay 0.10 para lang mas paspas ang pag visualize
PROB_BURNOUT = 0.05

# Visualization settings
PAUSE_TIME = 0.05

rng = np.random.default_rng(SEED)

def initialize_grid(size):
    grid = np.full((size, size), STATE_NOT_YET_BURNING, dtype=np.int8)
    mask_unburn = rng.random((size, size)) < 0.1
    grid[mask_unburn] = STATE_NON_BURNABLE
    
    print(f"Initialized a {size}x{size} grid.")
    return grid

def set_ignition_point(grid, x, y):
    if grid[x, y] == STATE_NOT_YET_BURNING:
        grid[x, y] = STATE_BLAZING
        print(f"Ignition point set at ({x}, {y})")
    else:
        print(f"Cannot ignite at ({x}, {y}): Cell is non-burnable.")
    return grid

def simulation_step(grid):
    new_grid = grid.copy()
    size = grid.shape[0]

    for i in range(size):
        for j in range(size):
            state = grid[i, j]

            # Rule 1 Combustible cell (2) can ignite (3)
            if state == STATE_NOT_YET_BURNING:
                has_blazing_neighbor = False
                for ni in range(max(0, i - 1), min(size, i + 2)):
                    for nj in range(max(0, j - 1), min(size, j + 2)):
                        if (ni, nj) == (i, j): 
                            continue
                        if grid[ni, nj] == STATE_BLAZING:
                            has_blazing_neighbor = True
                            break
                    if has_blazing_neighbor:
                        break
                
                # If a neighbor is blazing, apply the ignition probability
                if has_blazing_neighbor and np.random.rand() < PROB_IGNITION:
                    new_grid[i, j] = STATE_IGNITED

            # Rule 2: Ignited cell (3) becomes Blazing (4)
            elif state == STATE_IGNITED:
                new_grid[i, j] = STATE_BLAZING

            elif state == STATE_BLAZING:
                if np.random.rand() < PROB_BURNOUT:
                    new_grid[i, j] = STATE_EXTINGUISHED
            
    return new_grid

def setup_visualization(size):
    colors = {
        STATE_NON_BURNABLE: '#808080',      # 1: Gray (Non-burnable)
        STATE_NOT_YET_BURNING: '#008000', # 2: Green (Combustible)
        STATE_IGNITED: '#FFFF00',       # 3: Yellow (Ignited)
        STATE_BLAZING: '#FF0000',         # 4: Red (Blazing)
        STATE_EXTINGUISHED: '#000000'     # 5: Black (Burned out)
    }
    
    cmap = mcolors.ListedColormap([colors[i] for i in sorted(colors.keys())])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    patches = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in colors]
    labels = [
        "1: Non-Burnable",
        "2: Combustible",
        "3: Ignited",
        "4: Blazing",
        "5: Extinguished"
    ]
    ax.legend(patches, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xticks(np.arange(-.5, size, 10), minor=False)
    ax.set_yticks(np.arange(-.5, size, 10), minor=False)
    ax.grid(True, which='major', color='white', linewidth=0.5, alpha=0.3)
    ax.set_xticklabels(np.arange(0, size+1, 10))
    ax.set_yticklabels(np.arange(0, size+1, 10))
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    
    return fig, ax, cmap, norm

def update_visualization(fig, ax, cmap, norm, grid, step):

    ax.set_title(f"Fire Spread Simulation - Step {step}")
    if not hasattr(ax, 'im'):
        ax.im = ax.imshow(grid, cmap=cmap, norm=norm, animated=True)
    else:
        ax.im.set_data(grid)
    
    plt.pause(PAUSE_TIME)

def main():
    num_steps = 300
    simulation_grid = initialize_grid(GRID_SIZE)
    simulation_grid = set_ignition_point(simulation_grid, 75, 50) 
    
    # For second ignition point
    # simulation_grid = set_ignition_point(simulation_grid, 75, 75)simulation_grid = set_ignition_point(simulation_grid, 75, 75)

    fig, ax, cmap, norm = setup_visualization(GRID_SIZE)
    plt.ion()
    plt.show()

    for step in range(num_steps):
        update_visualization(fig, ax, cmap, norm, simulation_grid, step)
        
        simulation_grid = simulation_step(simulation_grid)

        if (np.count_nonzero(simulation_grid == STATE_IGNITED) == 0 and
            np.count_nonzero(simulation_grid == STATE_BLAZING) == 0):
            print(f"Fire burned out at step {step}.")
            break
            
    print("--- Simulation Finished ---")
    print("--- Visualizing Final State ---")
    update_visualization(fig, ax, cmap, norm, simulation_grid, step + 1)
    plt.ioff() 
    print("Close the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    main()