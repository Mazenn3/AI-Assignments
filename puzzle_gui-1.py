from collections import deque  # For BFS queue
import time  # For measuring how long solutions take
import math  # For calculating distances
import heapq  # For A* priority queue


class PuzzleState:
    """
    Represents a state of the 8-puzzle board.
    Think of this as taking a snapshot of the board at any moment.
    """

    def __init__(self, state, parent=None, move=None, depth=0, cost=0):
        self.state = tuple(state)  # The actual board configuration (0 represents blank)
        self.parent = parent  # Previous board state
        self.move = move  # What move led to this state
        self.depth = depth  # How many moves from start
        self.cost = cost  # For A* search

    def __eq__(self, other):
        # Two states are equal if they have the same board configuration
        return self.state == other.state

    def __hash__(self):
        # Needed to use states in sets (for keeping track of visited states)
        return hash(self.state)

    def __lt__(self, other):
        # Needed for A* to compare states
        return self.cost < other.cost

    def get_blank_pos(self):
        """Find where the blank (0) is on the board."""
        return self.state.index(0)

    def get_possible_moves(self):
        """Figure out what moves are possible from current state."""
        moves = []
        blank_pos = self.get_blank_pos()
        row = blank_pos // 3  # Get row of blank
        col = blank_pos % 3  # Get column of blank

        # Check each possible direction
        if row > 0:
            moves.append('UP')
        if row < 2:
            moves.append('DOWN')
        if col > 0:
            moves.append('LEFT')
        if col < 2:
            moves.append('RIGHT')

        return moves
    #ميثود بتخليه يتحرك
    def apply_move(self, move):
        """Make a move and return new board state."""
        new_state = list(self.state)
        blank_pos = self.get_blank_pos()

        # Calculate where blank will move to
        if move == 'UP':
            new_pos = blank_pos - 3
        elif move == 'DOWN':
            new_pos = blank_pos + 3
        elif move == 'LEFT':
            new_pos = blank_pos - 1
        elif move == 'RIGHT':
            new_pos = blank_pos + 1

        # Swap blank with tile in new position
        new_state[blank_pos], new_state[new_pos] = new_state[new_pos], new_state[blank_pos]
        return PuzzleState(new_state, self, move, self.depth + 1)

    def print_state(self):
        """Show the board in a nice grid format."""
        print("-" * 13) #عشان يفصل مبينهم بخط متقطع و بيدي شكل حلو
        for i in range(0, 9, 3): #دي لوب بترص الارقام
            print(f"| {self.state[i]} | {self.state[i + 1]} | {self.state[i + 2]} |")
            print("-" * 13)


def manhattan_distance(state):
    """
    Calculate how many rows and columns each tile is from its goal position.
    This helps A* search estimate how far we are from the solution.
    """
    distance = 0
    for i, tile in enumerate(state):
        if tile != 0:  # Skip blank tile
            current_row = i // 3
            current_col = i % 3
            goal_row = tile // 3
            goal_col = tile % 3
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance


def euclidean_distance(state):
    """
    Calculate straight-line distance each tile is from its goal.
    Another way to help A* search estimate distance to goal.
    """
    distance = 0
    for i, tile in enumerate(state):
        if tile != 0:  # Skip blank tile
            current_row = i // 3
            current_col = i % 3
            goal_row = tile // 3
            goal_col = tile % 3
            distance += math.sqrt((current_row - goal_row) ** 2 + (current_col - goal_col) ** 2)
    return distance


def is_goal(state):
    """Check if we've reached the solved state (0-8 in order)."""
    return state.state == tuple(range(9))


def get_solution_path(state):
    """
    Trace back from goal to start to get solution path.
    Returns list of states from start to goal.
    """
    path = []
    current = state
    while current:
        path.append(current)
        current = current.parent
    return list(reversed(path))

#الميثود بتاعت ال BFS
def bfs(initial_state):
    """
    Breadth-First Search: Explores all moves at current depth before going deeper.
    Guarantees shortest solution but might take longer.
    """
    start_time = time.time() #بيحسب وقت الرن بتاع BFS لوحده
    frontier = deque([PuzzleState(initial_state)])  # Queue of states to explore
    explored = set()  # Keep track of states we've seen
    nodes_expanded = 0

    while frontier:
        current_state = frontier.popleft()  # Get next state to explore

        if current_state.state in explored:
            continue  # Skip if we've seen this before

        explored.add(current_state.state)
        nodes_expanded += 1

        # Check if we found the solution
        if is_goal(current_state):
            solution_path = get_solution_path(current_state)
            print("\nSolution Path:")
            for i, state in enumerate(solution_path):
                print(f"\nStep {i}: {state.move if state.move else 'Start'}")
                state.print_state()

            # Print search results after showing solution path
            print(f"\nBFS Search Results:")
            print("=" * 50)
            print(f"Path Cost: {len(solution_path) - 1}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Search Depth: {current_state.depth}")
            print(f"Running Time: {time.time() - start_time:.4f} seconds")

            return {
                "path": solution_path,
                "path_cost": len(solution_path) - 1,
                "nodes_expanded": nodes_expanded,
                "search_depth": current_state.depth,
                "running_time": time.time() - start_time
            }

        # Try all possible moves from current state
        for move in current_state.get_possible_moves():
            new_state = current_state.apply_move(move)
            if new_state.state not in explored:
                frontier.append(new_state)

    return None

#المثود بتاعت ال DFS
def dfs(initial_state):
    """
    Depth-First Search: Explores as far as possible along each path before backtracking.
    Might not find the shortest solution but could be faster sometimes.
    """
    start_time = time.time() #عشان تحسب وقت رن الالجوريزم لوحده
    frontier = [PuzzleState(initial_state)]  # Stack of states to explore
    explored = set()  # Keep track of states we've seen
    nodes_expanded = 0

    while frontier:#بيحفظ هو كان فين قبل كده عشان ميرجعلهاش تاني
        current_state = frontier.pop()  # Get next state to explore

        if current_state.state in explored:
            continue  # Skip if we've seen this before

        explored.add(current_state.state)
        nodes_expanded += 1

        # Check if we found the solution
        if is_goal(current_state):
            solution_path = get_solution_path(current_state)
            print("\nSolution Path:")
            for i, state in enumerate(solution_path):
                print(f"\nStep {i}: {state.move if state.move else 'Start'}")
                state.print_state()

            # Print search results after showing solution path
            print(f"\nDFS Search Results:")
            print("=" * 50)
            print(f"Path Cost: {len(solution_path) - 1}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Search Depth: {current_state.depth}")
            print(f"Running Time: {time.time() - start_time:.4f} seconds")

            return {
                "path": solution_path,
                "path_cost": len(solution_path) - 1,
                "nodes_expanded": nodes_expanded,
                "search_depth": current_state.depth,
                "running_time": time.time() - start_time
            }

        # Try all possible moves from current state (in reverse order for DFS)
        for move in reversed(current_state.get_possible_moves()):
            new_state = current_state.apply_move(move)
            if new_state.state not in explored:
                frontier.append(new_state)

    return None


def a_star(initial_state, heuristic_func):
    """
    A* Search: Uses a heuristic to estimate which moves look most promising.
    Tries to combine speed of DFS with accuracy of BFS.
    """
    start_time = time.time()
    initial = PuzzleState(initial_state)
    frontier = [(heuristic_func(initial.state), initial)]  # Priority queue
    explored = set()  # Keep track of states we've seen
    nodes_expanded = 0

    while frontier:
        _, current_state = heapq.heappop(frontier)  # Get most promising state

        if current_state.state in explored:
            continue  # Skip if we've seen this before

        explored.add(current_state.state)
        nodes_expanded += 1

        # Check if we found the solution
        if is_goal(current_state):
            solution_path = get_solution_path(current_state)
            print("\nSolution Path:")
            for i, state in enumerate(solution_path):
                print(f"\nStep {i}: {state.move if state.move else 'Start'}")
                state.print_state()

            # Print search results after showing solution path
            print(f"\nA* Search Results:")
            print("=" * 50)
            print(f"Path Cost: {len(solution_path) - 1}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Search Depth: {current_state.depth}")
            print(f"Running Time: {time.time() - start_time:.4f} seconds")

            return {
                "path": solution_path,
                "path_cost": len(solution_path) - 1,
                "nodes_expanded": nodes_expanded,
                "search_depth": current_state.depth,
                "running_time": time.time() - start_time
            }

        # Try all possible moves from current state
        for move in current_state.get_possible_moves():
            new_state = current_state.apply_move(move)
            if new_state.state not in explored:
                new_state.cost = new_state.depth + heuristic_func(new_state.state)
                heapq.heappush(frontier, (new_state.cost, new_state))

    return None

#بيهندل اي حاجه اليوزر مكن يدخلها
def validate_input(numbers):
    """Make sure the puzzle input is valid."""
    if len(numbers) != 9:
        return False, "Must enter exactly 9 numbers"
    if not all(isinstance(n, int) and 0 <= n <= 8 for n in numbers):
        return False, "All numbers must be integers between 0 and 8"
    if len(set(numbers)) != 9:
        return False, "Each number must appear exactly once"
    return True, ""


def main():
    """Main program that handles user interaction."""
    print("\n8-Puzzle Solver")
    print("=" * 50)

    # Get starting puzzle state from user
    while True:
        try:
            print("\nEnter the initial state (0-8, where 0 represents the blank space)")
            print("Enter 9 numbers separated by spaces:")
            initial_state = [int(x) for x in input().strip().split()]
            valid, error_message = validate_input(initial_state)
            if valid:
                break
            print(f"Error: {error_message}")
        except ValueError:
            print("Error: Please enter valid numbers")

    # Let user choose which search method to use
    while True:
        print("\nSelect search algorithm:")
        print("1. Breadth-First Search (BFS)")
        print("2. Depth-First Search (DFS)")
        print("3. A* Search with Manhattan Distance")
        print("4. A* Search with Euclidean Distance")
        print("5. Exit")
    #error handling
        try:
            choice = int(input("\nEnter your choice (1-5): "))

            if choice == 5:
                print("\nExiting program...")
                break

            if choice not in [1, 2, 3, 4]:
                print("Invalid choice! Please enter a number between 1 and 5.")
                continue

            # Run the chosen algorithm
            if choice == 1:
                metrics = bfs(initial_state)
            elif choice == 2:
                metrics = dfs(initial_state)
            elif choice == 3:
                metrics = a_star(initial_state, manhattan_distance)
            else:  # choice == 4
                metrics = a_star(initial_state, euclidean_distance)

            if not metrics:
                print("\nNo solution found!")

            # Ask if user wants to try another method
            retry = input("\nWould you like to try another algorithm? (y for yes, anything for no): ")
            if retry.lower() != 'y':
                break

        except ValueError:
            print("Invalid input! Please enter a number.")


if __name__ == "__main__":
    main()
