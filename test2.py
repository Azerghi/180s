import pygame
import heapq
import time
from collections import deque
import random as rd


ROWS, COLS = 31, 55
TILE_SIZE = 35
WIDTH, HEIGHT = COLS * TILE_SIZE, ROWS * TILE_SIZE

pygame.init()
font = pygame.font.SysFont("Arial", 24)
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
info = pygame.display.Info()
screen_width, screen_height = info.current_w, info.current_h

# Choisir la plus grande taille possible de cellule sans dépasser
cell_width = screen_width // COLS
cell_height = screen_height // ROWS
CELL_SIZE = max(cell_width, cell_height)  # pour couvrir l’écran au mieux

# Recalculer les dimensions réelles de la grille
grid_width = CELL_SIZE * COLS
grid_height = CELL_SIZE * ROWS

# Centrer la grille
offset_x = (screen_width - grid_width) // 2
offset_y = (screen_height - grid_height) // 2

clock = pygame.time.Clock()
pygame.display.set_caption("Carte du Donjon")

# --- Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 150, 255)
DARK_BLUE = (0, 100, 200)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)

# --- Grid
def generate_grid(rows, cols, obstacle_prob=0.3):

    rd.seed(1231) 
    return [[0 if rd.random() > obstacle_prob else 1 for _ in range(cols)] for _ in range(rows)]

# Load these once at the start of your script (after pygame.init())
floor_img = pygame.transform.scale(pygame.image.load("floor.png"), (TILE_SIZE, TILE_SIZE))
wall_img = pygame.transform.scale(pygame.image.load("wall.png"), (TILE_SIZE, TILE_SIZE))

def draw_grid(grid, path, visited, frontier, start, goal, current=None):
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if (row, col) == start:
                pygame.draw.rect(screen, BLUE, rect)
            elif (row, col) == goal:
                pygame.draw.rect(screen, RED, rect)
            elif (row, col) in path:
                pygame.draw.rect(screen, PURPLE, rect)
            elif (row, col) in visited:
                pygame.draw.rect(screen, GREEN, rect)
            elif (row, col) in frontier:
                pygame.draw.rect(screen, ORANGE, rect)
            elif grid[row][col] == 1:
                screen.blit(wall_img, rect)
            else:
                screen.blit(floor_img, rect)

            pygame.draw.rect(screen, GRAY, rect, 1)

    if current:
        crow, ccol = current
        pygame.draw.rect(screen, YELLOW, (ccol * TILE_SIZE, crow * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3)

    pygame.display.flip()


# --- Algorithms
def bfs_generator(grid, start, goal):
    queue = deque([start])
    came_from = {start: None}
    visited = set()
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        visited.add(current)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = current[0]+dy, current[1]+dx
            neighbor = (ny, nx)
            if 0 <= ny < ROWS and 0 <= nx < COLS and grid[ny][nx] == 0 and neighbor not in came_from:
                queue.append(neighbor)
                came_from[neighbor] = current
        yield current, visited, queue, came_from
    yield goal, visited, [], came_from

def a_star_generator(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_score = {start: 0}
    visited = set()

    def h(p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break
        visited.add(current)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = current[0]+dy, current[1]+dx
            neighbor = (ny, nx)
            if 0 <= ny < ROWS and 0 <= nx < COLS and grid[ny][nx] == 0:
                tentative = g_score[current] + 1
                if neighbor not in g_score or tentative < g_score[neighbor]:
                    g_score[neighbor] = tentative
                    f = tentative + h(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current
        yield current, visited, [node for _, node in open_set], came_from
    yield goal, visited, [], came_from

def reconstruct_path(came_from, end):
    path = []
    while end in came_from and came_from[end] is not None:
        path.append(end)
        end = came_from[end]
    path.reverse()
    return path

# --- Main loop
# Replace the `run_visualization` function with this:
def run_visualization(algo='bfs'):
    grid = generate_grid(ROWS, COLS, obstacle_prob=0.3)
    start, goal = (0, 0), (ROWS-1, COLS-1)
    while grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        grid = generate_grid(ROWS, COLS, obstacle_prob=0.3)

    if algo == 'bfs':
        algo_gen = bfs_generator(grid, start, goal)
    else:
        algo_gen = a_star_generator(grid, start, goal)

    came_from = {}
    path = []
    running = True
    done = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

        if not done:
            try:
                current, visited, frontier, came_from = next(algo_gen)
                if goal in came_from:
                    path = reconstruct_path(came_from, goal)
                screen.fill(WHITE)
                draw_grid(grid, path, visited, frontier, start, goal, current)
                clock.tick(60)
            except StopIteration:
                done = True
                if goal in came_from:
                    path = reconstruct_path(came_from, goal)
                else:
                    print("No path found.")
        else:
            draw_grid(grid, path, visited, [], start, goal, None)
            print(len(visited))



run_visualization('a_star')  # or 'a_star'bfs
