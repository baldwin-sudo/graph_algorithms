import tkinter as tk
import random
from collections import deque

from utils import Stack
import heapq
# Constants
CELL_SIZE = 40  # Size of each cell in the maze
MAZE_WIDTH = 15   # Width of the maze (in cells)
MAZE_HEIGHT = 15  # Height of the maze (in cells)
BUTTON_WIDTH = 30  # Width of buttons
DELAY = 100  # Delay in milliseconds for visualization

BFS_COLOR = "blue"
DFS_COLOR = "purple"
ASTAR_COLOR = "red"

class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Algorithms")

        self.canvas = tk.Canvas(root, width=CELL_SIZE * MAZE_WIDTH, height=CELL_SIZE * MAZE_HEIGHT, bg="lightgray")
        self.canvas.pack(side=tk.LEFT)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.RIGHT, padx=10)

        self.maze = self.generate_maze()  # Generate the initial maze
        self.start = (0, 0)  # Starting point
        self.end = (MAZE_HEIGHT - 1, MAZE_WIDTH - 1)  # Ending point
        self.draw_maze()

        # Create buttons
        self.create_buttons()

        # Flag to indicate if an algorithm is running
        self.is_running = False
        self.heuristic=euclidean_distance
    def generate_maze(self):
        # Create a grid filled with walls
        maze = [[1 for _ in range(MAZE_WIDTH)] for _ in range(MAZE_HEIGHT)]

        # Carve 2 paths using DFS
        self.carve_path(maze, 0, 0)
        

        # Ensure start and end points are paths
        maze[0][0] = 0
        maze[MAZE_HEIGHT - 1][MAZE_WIDTH - 1] = 0
        maze[MAZE_HEIGHT - 1][MAZE_WIDTH - 2] = 0
        maze[MAZE_HEIGHT - 2][MAZE_WIDTH - 1] = 0
        self.carve_second_path(maze)
        
        return maze

    def carve_path(self, maze, x, y):
        # Directions: up, down, left, right
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)  # Shuffle to create random maze paths
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            random.shuffle(directions)
            # Check if the new position is within bounds and is a wall
            if 0 <= nx < MAZE_HEIGHT and 0 <= ny < MAZE_WIDTH and maze[nx][ny] == 1:
                maze[nx][ny] = 0  # Carve a path
                maze[x + dx // 2][y + dy // 2] = 0  # Also carve between current and new position
                self.carve_path(maze, nx, ny)  # Recursively carve from the new position
    def carve_second_path(self, maze):
        """
        Carves a second path from the start to the end to ensure multiple paths exist.
        This method tries to avoid overlapping too much with the first path.
        """
        # Directions: up, down, left, right
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        
        # Use BFS to try to find an alternative path to carve
        queue = deque([(0, 0)])
        visited = [[False for _ in range(MAZE_WIDTH)] for _ in range(MAZE_HEIGHT)]
        visited[0][0] = True
        
        while queue:
            x, y = queue.popleft()
            
            random.shuffle(directions)  # Shuffle to create random maze paths
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # Check if the new position is within bounds and is a wall
                if 0 <= nx < MAZE_HEIGHT and 0 <= ny < MAZE_WIDTH and maze[nx][ny] == 1:
                    # Only carve this new path if it doesn't overlap too much with the previous path
                    if not visited[nx][ny] and random.random() > 0.3:  # Add some randomness
                        maze[nx][ny] = 0  # Carve a path
                        maze[x + dx // 2][y + dy // 2] = 0  # Carve the wall between current and new position
                        visited[nx][ny] = True
                        queue.append((nx, ny))

        # Ensure the end is reachable
        maze[MAZE_HEIGHT - 1][MAZE_WIDTH - 1] = 0
    def draw_maze(self):
        self.is_running=False
        self.canvas.delete("all")  # Clear previous drawings
        for row in range(MAZE_HEIGHT):
            for col in range(MAZE_WIDTH):
                color = "black" if self.maze[row][col] == 1 else "white"
                x1 = col * CELL_SIZE
                y1 = row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
        self.color_goal_and_start()

    def create_buttons(self):
        tk.Button(self.button_frame, text="BFS", width=BUTTON_WIDTH, command=self.bfs,bg=BFS_COLOR,fg="white",font=("Helvetica", 16, "bold")).pack(pady=5)
        tk.Button(self.button_frame, text="DFS", width=BUTTON_WIDTH, command=self.dfs,bg=DFS_COLOR,fg="white",font=("Helvetica", 16, "bold")).pack(pady=5)
        tk.Button(self.button_frame, text="A *", width=BUTTON_WIDTH, command=self.a_star,bg=ASTAR_COLOR,fg="white",font=("Helvetica", 16, "bold")).pack(pady=5)
        # Buttons for selecting heuristic
        tk.Button(self.button_frame, text="Heuristic: Euclidean", width=BUTTON_WIDTH,fg="red", command=self.set_euclidean, font=("Helvetica", 16)).pack(pady=5)
        tk.Button(self.button_frame, text="Heuristic: Manhattan", width=BUTTON_WIDTH, fg="red",command=self.set_manhattan, font=("Helvetica", 16)).pack(pady=5)
        tk.Button(self.button_frame, text="Heuristic: Diagonal", width=BUTTON_WIDTH, fg="red",command=self.set_diagonal, font=("Helvetica", 16)).pack(pady=5)
        tk.Button(self.button_frame, text="Generate New Maze",bg="black",fg="white", width=BUTTON_WIDTH, command=self.reset_and_generate,font=("Helvetica", 16)).pack(pady=5)
        tk.Button(self.button_frame, text="Reset", bg="black",fg="white",width=BUTTON_WIDTH, command=self.draw_maze,font=("Helvetica", 16)).pack(pady=5)
    def set_euclidean(self):
        self.heuristic = euclidean_distance
        print("Heuristic set to Euclidean")

    def set_manhattan(self):
        self.heuristic = manhattan_distance
        print("Heuristic set to Manhattan")
    def set_diagonal(self):
        self.heuristic = diagonal_distance
        print("Heuristic set to diagonal")
    def bfs(self):
        if self.is_running:  # Prevent starting a new search if one is already running
            return

        self.is_running = True
        visited = [[False] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]
        queue = deque([self.start])
        visited[self.start[0]][self.start[1]] = True
        path = []  # To store the path taken

        while queue and self.is_running:  # Check if the algorithm should continue
            current = queue.popleft()
            path.append(current)  # Add current cell to the path

            if current == self.end:
                break  # Reached the end

            # Get neighbors
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                x, y = neighbor
                if not visited[x][y]:
                    visited[x][y] = True
                    queue.append(neighbor)

        # Only visualize the path if the algorithm finished without interruption
        if current == self.end:
            self.visualize_algorithm(path, BFS_COLOR)

        self.is_running = False  # Mark the algorithm as not running after finishing

    def get_neighbors(self, position):
        x, y = position
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < MAZE_HEIGHT and 0 <= ny < MAZE_WIDTH and self.maze[nx][ny] == 0:
                neighbors.append((nx, ny))

        return neighbors

    def visualize_algorithm(self, path, color):
        for step in path:
            if not self.is_running:  # Check if the algorithm should stop
                break

            x, y = step
            x1 = y * CELL_SIZE
            y1 = x * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE

            # Color the cell to indicate traversal
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
            self.color_goal_and_start()
            self.root.update()  # Update the display
            self.root.after(DELAY)  # Wait for a short delay

    def reset_and_generate(self):
        self.is_running = False  # Stop any running algorithms
        self.maze = self.generate_maze()  # Generate a new maze
        self.draw_maze()  # Draw the new maze

    def color_goal_and_start(self):
        # Coordinates for the start and end points
        start_x = 0
        start_y = 0
        finish_x = MAZE_WIDTH - 1
        finish_y = MAZE_HEIGHT - 1

        # Color the start point (top-left corner)
        x1_start = start_x * CELL_SIZE
        y1_start = start_y * CELL_SIZE
        x2_start = x1_start + CELL_SIZE
        y2_start = y1_start + CELL_SIZE
        self.canvas.create_rectangle(x1_start, y1_start, x2_start, y2_start, fill="yellow")  # Start colored green

        # Color the finish point (bottom-right corner)
        x1_finish = finish_x * CELL_SIZE
        y1_finish = finish_y * CELL_SIZE
        x2_finish = x1_finish + CELL_SIZE
        y2_finish = y1_finish + CELL_SIZE
        self.canvas.create_rectangle(x1_finish, y1_finish, x2_finish, y2_finish, fill="green")  # Finish colored red

    def dfs(self):
        if self.is_running:  # Prevent starting a new search if one is already running
            return

        self.is_running = True
        visited = [[False] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]
        stack = Stack()
        stack.push(self.start)
        visited[self.start[0]][self.start[1]] = True
        path = []
        while stack.size() > 0 and self.is_running:
            current = stack.pop()
            path.append(current)
            if current == self.end:
                break
            neighbors = self.get_neighbors(current)
            for n in neighbors:
                x, y = n
                if not visited[x][y]:
                    stack.push(n)
                    visited[x][y] = True
        # Only visualize the path if the algorithm finished without interruption
        if current == self.end:
            self.visualize_algorithm(path, DFS_COLOR)

        self.is_running = False  # Mark the algorithm as not running after finishing
        

    def a_star(self):
        #  a priority queue
        if self.is_running:  # Prevent starting a new search if one is already running
            return

        self.is_running = True
        priority_queue =[]
        
        # to keep track of the path :
        path=[]
        g_score={}
        f_score={}
        for i in range(MAZE_HEIGHT):
            for j in range(MAZE_WIDTH):
                g_score[(i,j)]=float('inf')
                f_score[(i,j)]=float('inf')
        
        # g score is the actuall distance (weight) of the path
        g_score[self.start]=0
        # f score is equal : g_score + the heuristic 
        f_score[self.start]=self.heuristic(self.start,self.end)
        # lets push the self.start in the priority queue
        heapq.heappush(priority_queue,(f_score[self.start],self.start))
        while len(priority_queue)>0 :
            current_f_score,current_node = heapq.heappop(priority_queue)
            path.append(current_node) 
            if current_f_score > f_score[current_node]:
                continue
            if current_node==self.end:
                break
            for neighbor in self.get_neighbors(current_node):
                new_g_score = g_score[current_node]+1
                if new_g_score < g_score[neighbor]:
                    new_f_score = new_g_score + self.heuristic(neighbor,self.end)
                    g_score[neighbor]=new_g_score
                    f_score[neighbor]=new_f_score
                    heapq.heappush(priority_queue,(f_score[neighbor],neighbor))
        if current_node == self.end:
            self.visualize_algorithm(path=path,color=ASTAR_COLOR)
        
        self.is_running = False
def euclidean_distance(y,x):
    return ((x[0]-y[0])**2+(x[1]-y[1])**2)**(1/2)
def manhattan_distance(x,y):
    return (abs(x[0]-y[0])+abs(x[1]-y[1]))        

def diagonal_distance(x, y):
    return max(abs(x[0] - y[0]), abs(x[1] - y[1]))

def main():
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
