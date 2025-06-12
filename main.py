# main.py
import pygame
from map import generate_static_map, ROWS, COLS
import heapq
import math


# Configuration
CELL_SIZE = 25
WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE
FPS = 30



class Explorer:
    def __init__(self, start_pos, map_grid):
        self.pos = start_pos
        self.map = map_grid
        self.known_map = [[-1 for _ in range(len(map_grid[0]))] for _ in range(len(map_grid))]  # -1 = inconnu
        self.view_radius = 5
        self.path = []

    def update_visibility(self):
        for dy in range(-self.view_radius, self.view_radius + 1):
            for dx in range(-self.view_radius, self.view_radius + 1):
                nx = self.pos[1] + dx
                ny = self.pos[0] + dy
                if 0 <= nx < len(self.map[0]) and 0 <= ny < len(self.map):
                    if self.line_of_sight(self.pos, (ny, nx)):
                        self.known_map[ny][nx] = self.map[ny][nx]
                    """elif self.map[ny][nx] == 1:
                        self.known_map[ny][nx] = 1  # Still mark the wall as visible"""


    def line_of_sight(self, start, end):
        x0, y0 = start[1], start[0]
        x1, y1 = end[1], end[0]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if x == x1 and y == y1:
                return True
            if self.map[y][x] == 1:
                return False  # Wall is visible, but blocks further sight
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return True


    def get_frontiers(self):
        frontiers = []
        for y in range(1, len(self.known_map) - 1):
            for x in range(1, len(self.known_map[0]) - 1):
                if self.known_map[y][x] == 0:
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = x + dx, y + dy
                        if self.known_map[ny][nx] == -1:
                            frontiers.append((y, x))
                            break
        return frontiers


def draw_grid(screen, grid, wall_img, floor_img):
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if grid[row][col] == 1:
                screen.blit(wall_img, rect)
            else:
                screen.blit(floor_img, rect)
            #pygame.draw.rect(screen, GRID_COLOR, rect, 1)  # facultatif : grille par-dessus

def main():
    pygame.init()
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

    wall_img = pygame.image.load("wall.png").convert()
    wall_img = pygame.transform.scale(wall_img, (CELL_SIZE, CELL_SIZE))

    floor_img = pygame.image.load("floor.png").convert()
    floor_img = pygame.transform.scale(floor_img, (CELL_SIZE, CELL_SIZE))

    explorer_img = pygame.image.load("knight.png").convert_alpha()
    explorer_img = pygame.transform.scale(explorer_img, (CELL_SIZE, CELL_SIZE))

    grid = generate_static_map()
    explorer = Explorer((26,0), grid)
    explorer.update_visibility()


    def a_star(start, goal, grid):
        heap = []
        heapq.heappush(heap, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while heap:
            _, current = heapq.heappop(heap)

            if current == goal:
                break

            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = current[0] + dy, current[1] + dx
                if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):
                    if grid[ny][nx] != 1:  # mur bloquant
                        new_cost = cost_so_far[current] + 1
                        next_node = (ny, nx)
                        if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                            cost_so_far[next_node] = new_cost
                            priority = new_cost + heuristic(goal, next_node)
                            heapq.heappush(heap, (priority, next_node))
                            came_from[next_node] = current

        # Reconstruire le chemin
        path = []
        node = goal
        while node and node in came_from:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path[1:]  # on ignore la case actuelle

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # distance de Manhattan

    def draw_grid(screen, known_map, wall_img, floor_img):
        for row in range(ROWS):
            for col in range(COLS):
                x = offset_x + col * CELL_SIZE
                y = offset_y + row * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                val = known_map[row][col]
                if val == -1:
                    pygame.draw.rect(screen, (30, 30, 30), rect)  # inconnu = noir
                elif val == 1:
                    screen.blit(wall_img, rect)
                else:
                    screen.blit(floor_img, rect)

        ey, ex = explorer.pos
        ex_px = offset_x + ex * CELL_SIZE
        ey_px = offset_y + ey * CELL_SIZE
        screen.blit(explorer_img, (ex_px, ey_px))



    running = True
    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        screen.fill((0, 0, 0))
        if not explorer.path:
            frontiers = explorer.get_frontiers()
            if frontiers:
                target = frontiers[0]  # on pourrait choisir plus intelligent plus tard
                explorer.path = a_star(explorer.pos, target, explorer.known_map)

        if explorer.path:
            next_pos = explorer.path.pop(0)
            explorer.pos = next_pos
            explorer.update_visibility()


        draw_grid(screen, explorer.known_map, wall_img, floor_img)
        pygame.display.flip()

    pygame.quit()



if __name__ == "__main__":
    main()
