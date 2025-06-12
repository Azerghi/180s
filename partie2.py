# main.py
import pygame
from map import generate_static_map, ROWS, COLS
from collections import deque
import time
import heapq


CELL_SIZE = 25
WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE
FPS = 30

class Explorer:
    def __init__(self, start_pos, map_grid):
        self.pos = start_pos
        self.map = map_grid
        self.path = []

    def a_star(self, start, goal):
        #start a timer for the A* algorithm
        time_start = time.time()
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path, time.time() - time_start

            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = current[0] + dy, current[1] + dx
                neighbor = (ny, nx)
                if 0 <= ny < len(self.map) and 0 <= nx < len(self.map[0]):
                    if self.map[ny][nx] < 1:  # sol ou porte
                        tentative_g = g_score[current] + 1
                        if neighbor not in g_score or tentative_g < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f = tentative_g + heuristic(neighbor, goal)
                            heapq.heappush(open_set, (f, neighbor))

        return []

    def bfs(self, start, goal):
        time_start = time.time()
        queue = deque()
        queue.append(start)
        visited = set()
        visited.add(start)
        came_from = {}

        while queue:
            current = queue.popleft()
            if current == goal:
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path, time.time() - time_start

            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = current[0] + dy, current[1] + dx
                neighbor = (ny, nx)
                if 0 <= ny < len(self.map) and 0 <= nx < len(self.map[0]):
                    if self.map[ny][nx] < 1 and neighbor not in visited:
                        visited.add(neighbor)
                        came_from[neighbor] = current
                        queue.append(neighbor)
        return []

def main():
    pygame.init()
    font = pygame.font.SysFont("Arial", 24)
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h

    cell_width = screen_width // COLS
    cell_height = screen_height // ROWS
    CELL_SIZE = max(cell_width, cell_height)
    grid_width = CELL_SIZE * COLS
    grid_height = CELL_SIZE * ROWS
    offset_x = (screen_width - grid_width) // 2
    offset_y = (screen_height - grid_height) // 2

    clock = pygame.time.Clock()
    pygame.display.set_caption("Phase 2 : Pathfinding vers la sortie")

    wall_img = pygame.image.load("wall.png").convert()
    wall_img = pygame.transform.scale(wall_img, (CELL_SIZE, CELL_SIZE))
    floor_img = pygame.image.load("floor.png").convert()
    floor_img = pygame.transform.scale(floor_img, (CELL_SIZE, CELL_SIZE))
    explorer_img = pygame.image.load("knight.png").convert_alpha()
    explorer_img = pygame.transform.scale(explorer_img, (CELL_SIZE, CELL_SIZE))
    door_img = pygame.image.load("Door.png").convert_alpha()
    door_img = pygame.transform.scale(door_img, (CELL_SIZE, CELL_SIZE))

    grid = generate_static_map()
    start_pos = (5, 43)   # Position du boss
    goal = (26, 0)        # Sortie

    explorer = Explorer(start_pos, grid)
    explorer.path, explore_time = explorer.bfs(start_pos, goal)
    path_len = len(explorer.path)  # Pour déclencher le calcul du chemin

    start_time = time.time()
    end_time = None
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        screen.fill((0, 0, 0))

        # Dessin de la carte
        for row in range(ROWS):
            for col in range(COLS):
                x = offset_x + col * CELL_SIZE
                y = offset_y + row * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                val = grid[row][col]
                if val == 1:
                    screen.blit(wall_img, rect)
                elif val == 2:
                    screen.blit(door_img, rect)
                else:
                    screen.blit(floor_img, rect)

        # Déplacement
        if explorer.path:
            explorer.pos = explorer.path.pop(0)
        elif end_time is None:
            end_time = time.time()
            print(f"Arrivé à la sortie en {end_time - start_time:.2f} secondes, pour un chemin de {path_len} cases, et {explore_time} de calcul.")

        # Affichage de l'explorateur
        ey, ex = explorer.pos
        ex_px = offset_x + ex * CELL_SIZE
        ey_px = offset_y + ey * CELL_SIZE
        screen.blit(explorer_img, (ex_px, ey_px))

        # Timer
        elapsed = (end_time or time.time()) - start_time
        timer_text = font.render(f"Time: {elapsed:.2f}s", True, (255, 255, 255))
        screen.blit(timer_text, (20, 20))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
