import numpy as np

# Grid parameters
GRID_SIZE = 5

# Elements on the grid
EMPTY = 0
MOUSE = 1
CHEESE = 2
TRAP = 3
WALL = 4
ORGANIC_CHEESE = 5

# Number of traps, walls, and organic cheeses
NUM_TRAPS = 2
NUM_WALLS = 2
NUM_ORGANIC_CHEESE = 1
NUM_CHEESE = 2

ACTIONS = ['up', 'down', 'left', 'right']
ACTION_TO_DELTA = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}


def initialize_grid_with_cheese_types(grid_size=GRID_SIZE, num_traps=NUM_TRAPS, num_walls=NUM_WALLS, 
                                      num_cheese= NUM_CHEESE, num_organic_cheese=NUM_ORGANIC_CHEESE):
    grid = np.zeros((grid_size, grid_size), dtype=int)

    # Randomly place mouse
    while True:
        mouse_pos = tuple(np.random.randint(0, grid_size, size=2))
        if grid[mouse_pos] == EMPTY:
            grid[mouse_pos] = MOUSE
            break

    # Randomly place normal cheese
    for _ in range(num_cheese):
        while True:
            cheese_pos = tuple(np.random.randint(0, grid_size, size=2))
            if grid[cheese_pos] == EMPTY:
                grid[cheese_pos] = CHEESE
                break

    # Place organic cheese
    organic_cheese_positions = []
    for _ in range(num_organic_cheese):
        while True:
            pos = tuple(np.random.randint(0, grid_size, size=2))
            if grid[pos] == EMPTY:
                grid[pos] = ORGANIC_CHEESE
                organic_cheese_positions.append(pos)
                break

    # Place traps
    for _ in range(num_traps):
        while True:
            trap_pos = tuple(np.random.randint(0, grid_size, size=2))
            if grid[trap_pos] == EMPTY:
                grid[trap_pos] = TRAP
                break

    # Place walls
    for _ in range(num_walls):
        while True:
            wall_pos = tuple(np.random.randint(0, grid_size, size=2))
            if grid[wall_pos] == EMPTY:
                grid[wall_pos] = WALL
                break

    return grid, mouse_pos, cheese_pos, organic_cheese_positions


def print_grid_with_cheese_types(grid):
    symbols = {
        EMPTY: '.',
        MOUSE: 'M',
        CHEESE: 'C',
        TRAP: 'T',
        WALL: '#',
        ORGANIC_CHEESE: 'O'
    }
    for row in grid:
        print(' '.join(symbols[cell] for cell in row))
        
        
        
def move(action, grid):
    delta = ACTION_TO_DELTA[action]

    # Find mouse position
    mouse_pos = tuple(np.argwhere(grid == MOUSE)[0])
    new_pos = (mouse_pos[0] + delta[0], mouse_pos[1] + delta[1])

    cell_content = grid[mouse_pos]
    # Check bounds and wall
    if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE:
        if grid[new_pos] != WALL:
            
            cell_content = grid[new_pos]
            
            grid[mouse_pos] = EMPTY  # Clear old position
            grid[new_pos] = MOUSE    # Move mouse
            
            return grid, cell_content

    return grid, cell_content


# Reward function
def get_reward(cell_content):
    if cell_content == CHEESE or cell_content == ORGANIC_CHEESE:
        return 10
    elif cell_content == TRAP:
        return -50 
    else:
        return -0.2


# Example usage
grid, mouse_pos, cheese_pos, organic_cheese_positions = initialize_grid_with_cheese_types()
print_grid_with_cheese_types(grid)
