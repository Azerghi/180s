import matplotlib.pyplot as plt
import numpy as np
import random
import time
import heapq
from collections import deque
import math

def generate_large_map(rows, cols, obstacle_prob=0.3):
    return [[1 if random.random() < obstacle_prob else 0 for _ in range(cols)] for _ in range(rows)]

def bfs(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    visited = [[False]*cols for _ in range(rows)]
    prev = [[None]*cols for _ in range(rows)]
    queue = deque([start])
    visited[start[0]][start[1]] = True

    while queue:
        y, x = queue.popleft()
        if (y, x) == goal:
            break
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < rows and 0 <= nx < cols and not visited[ny][nx] and grid[ny][nx] == 0:
                visited[ny][nx] = True
                prev[ny][nx] = (y, x)
                queue.append((ny, nx))

    path = []
    at = goal
    while at and at != start:
        path.append(at)
        at = prev[at[0]][at[1]]
    if at == start:
        path.append(start)
        path.reverse()
        return path
    return []

def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    def h(pos): return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = current[0]+dy, current[1]+dx
            neighbor = (ny, nx)
            if 0 <= ny < rows and 0 <= nx < cols and grid[ny][nx] == 0:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + h(neighbor)
                    heapq.heappush(open_set, (f, neighbor))
    return []

def line_of_sight(grid, p1, p2):
    """Line of sight with strict diagonal corner-cutting prevention"""
    x0, y0 = p1[1], p1[0]
    x1, y1 = p2[1], p2[0]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1

    x, y = x0, y0
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            if not is_free(grid, y, x):
                return False
            if x != x0 and y != y0:
                # Check both adjacent tiles to prevent corner-cutting
                if not is_free(grid, y, x - sx) and not is_free(grid, y - sy, x):
                    return False
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            if not is_free(grid, y, x):
                return False
            if x != x0 and y != y0:
                if not is_free(grid, y, x - sx) and not is_free(grid, y - sy, x):
                    return False
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    return is_free(grid, y1, x1)





def is_free(grid, y, x):
    return 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] == 0

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def lazy_theta_star(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: start}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            ny, nx = current[0]+dy, current[1]+dx
            neighbor = (ny, nx)
            if not is_free(grid, ny, nx):
                continue

            parent = came_from[current]
            if line_of_sight(grid, parent, neighbor):
                tentative_g = g_score[parent] + euclidean(parent, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = parent
                    f = tentative_g + euclidean(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
            else:
                tentative_g = g_score[current] + euclidean(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f = tentative_g + euclidean(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
    return []

def test_and_visualize(rows, cols, obstacle_prob=0.3):
    path_bfs, path_astar, path_ltheta = [], [], []
    while not path_bfs or not path_astar or not path_ltheta:
        grid = generate_large_map(rows, cols, obstacle_prob)
        start, goal = (0, 0), (rows-1, cols-1)

        t0 = time.perf_counter()
        path_bfs = bfs(grid, start, goal)
        bfs_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        path_astar = a_star(grid, start, goal)
        astar_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        path_ltheta = lazy_theta_star(grid, start, goal)
        ltheta_time = time.perf_counter() - t0

    print(f"BFS   -> Time: {bfs_time:.4f}s, Path length: {len(path_bfs)}")
    print(f"A*    -> Time: {astar_time:.4f}s, Path length: {len(path_astar)}")
    print(f"Lazy Theta* -> Time: {ltheta_time:.4f}s, Path length: {len(path_ltheta)}")

    grid_np = np.array(grid)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_np, cmap='Greys', origin='upper')

    def draw_path(path, color, label, style='-'):
        if path:
            y, x = zip(*path)
            ax.plot(x, y, style, label=label, color=color, linewidth=1)

    draw_path(path_bfs, 'blue', 'BFS')
    draw_path(path_astar, 'red', 'A*', '--')
    draw_path(path_ltheta, 'lime', 'Lazy Theta*', ':')

    ax.scatter([0], [0], color='green', s=50, label='Start')
    ax.scatter([cols-1], [rows-1], color='purple', s=50, label='Goal')

    ax.set_title("Pathfinding: BFS vs A* vs Lazy Theta*")
    ax.legend()
    plt.tight_layout()
    plt.show()

# Run visualization
test_and_visualize(1000, 1000, obstacle_prob=0.4)
