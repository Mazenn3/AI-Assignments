import matplotlib.pyplot as plt
import numpy as np
import math
import heapq
import time

# Define the goal state and goal positions
GOAL_STATE = (0, 1, 2, 3, 4, 5, 6, 7, 8)       #tuple
GOAL_POSITIONS = {val: (i // 3, i % 3) for i, val in enumerate(GOAL_STATE)}    #dict

# Helper function to find the position of a tile in the 3x3 grid
def get_position(state, tile):
    index = state.index(tile)
    return (index // 3, index % 3)
#// ---> gimme the ROW num   &&       % ---> gimme the CoLUMN num
# Heuristic functions
def manhattan_distance(state):
    distance = 0
    for i, tile in enumerate(state):
        if tile != 0:
            current_pos = (i // 3, i % 3)
            goal_pos = GOAL_POSITIONS[tile]
            distance += abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
    return distance

def euclidean_distance(state):
    distance = 0
    for i, tile in enumerate(state):
        if tile != 0:
            current_pos = (i // 3, i % 3)
            goal_pos = GOAL_POSITIONS[tile]
            distance += math.sqrt((current_pos[0] - goal_pos[0]) ** 2 + (current_pos[1] - goal_pos[1]) ** 2)
    return distance

# Helper function to generate new states from a given state
def get_neighbors(state):
    neighbors = []
    empty_pos = state.index(0)
    empty_row, empty_col = empty_pos // 3, empty_pos % 3
    moves = {
        'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)
    }
    for move, (dr, dc) in moves.items():
        new_row, new_col = empty_row + dr, empty_col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_pos = new_row * 3 + new_col
            new_state = list(state)
            new_state[empty_pos], new_state[new_pos] = new_state[new_pos], new_state[empty_pos]
            neighbors.append(tuple(new_state))
    return neighbors

# A* search function with additional metrics
def a_star_search(start_state, heuristic_func):
    start_time = time.time()
    priority_queue = []
    heapq.heappush(priority_queue, (0, start_state, []))
    visited = set()
    nodes_expanded = 0

    while priority_queue:
        cost, state, path = heapq.heappop(priority_queue)

        if state in visited:
            continue
        visited.add(state)
        nodes_expanded += 1

        if state == GOAL_STATE:
            end_time = time.time()
            running_time = end_time - start_time
            path_cost = len(path)
            search_depth = path_cost  # depth is the length of the path in this case
            return {
                "path": path + [state],
                "path_cost": path_cost,
                "nodes_expanded": nodes_expanded,
                "search_depth": search_depth,
                "running_time": running_time
            }

        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                new_path = path + [state]
                g_cost = len(new_path)
                h_cost = heuristic_func(neighbor)
                f_cost = g_cost + h_cost
                heapq.heappush(priority_queue, (f_cost, neighbor, new_path))

    return None

# Visualization function to display solution path
def visualize_solution_path(metrics, heuristic_name):
    path = metrics["path"]
    path_cost = metrics["path_cost"]
    nodes_expanded = metrics["nodes_expanded"]
    search_depth = metrics["search_depth"]
    running_time = metrics["running_time"]

    num_steps = len(path)
    cols = 5
    rows = (num_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    fig.suptitle(f"A* Solution Path using {heuristic_name} Heuristic\n"
                 f"Path Cost: {path_cost}, Nodes Expanded: {nodes_expanded}, "
                 f"Search Depth: {search_depth}, Running Time: {running_time:.4f} seconds", fontsize=12)

    axes = axes.flatten()

    for i, state in enumerate(path):
        ax = axes[i]
        grid = np.array(state).reshape(3, 3)

        ax.imshow(grid == 0, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

        for x in range(3):
            for y in range(3):
                num = grid[x, y]
                if num != 0:
                    ax.text(y, x, str(num), ha='center', va='center', fontsize=16, color='black')

        ax.set_title(f"Step {i+1}", fontsize=10)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

# Initial puzzle state
start_state = (1, 2, 3, 4, 5, 6, 0, 7, 8)

# Solve the puzzle using both heuristics
manhattan_metrics = a_star_search(start_state, manhattan_distance)
euclidean_metrics = a_star_search(start_state, euclidean_distance)

# Visualize the results for Manhattan distance
print("Using Manhattan Distance Heuristic:")
visualize_solution_path(manhattan_metrics, "Manhattan Distance")

# Visualize the results for Euclidean distance
print("\nUsing Euclidean Distance Heuristic:")
visualize_solution_path(euclidean_metrics, "Euclidean Distance")
