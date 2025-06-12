# main.py
import pygame
from map import generate_static_map, ROWS, COLS
import heapq
import math
from collections import deque

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
        visited = [[False for _ in range(len(self.known_map[0]))] for _ in range(len(self.known_map))]
        frontiers = []

        def is_frontier(y, x):
            if self.known_map[y][x] != 0:
                return False
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < len(self.known_map) and 0 <= nx < len(self.known_map[0]):
                    if self.known_map[ny][nx] == -1:
                        return True
            return False

        for y in range(1, len(self.known_map) - 1):
            for x in range(1, len(self.known_map[0]) - 1):
                if is_frontier(y, x) and not visited[y][x]:
                    # On lance un BFS pour regrouper cette zone frontière
                    queue = deque()
                    group = []
                    queue.append((y, x))
                    visited[y][x] = True

                    while queue:
                        cy, cx = queue.popleft()
                        group.append((cy, cx))
                        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ny, nx = cy + dy, cx + dx
                            if (0 <= ny < len(self.known_map) and
                                0 <= nx < len(self.known_map[0]) and
                                not visited[ny][nx] and is_frontier(ny, nx)):
                                visited[ny][nx] = True
                                queue.append((ny, nx))

                    # Calcul du centroïde
                    if group:
                        sum_y = sum(pos[0] for pos in group)
                        sum_x = sum(pos[1] for pos in group)
                        cy = int(round(sum_y / len(group)))
                        cx = int(round(sum_x / len(group)))
                        frontiers.append((cy, cx))

        return frontiers
    
    def choose_target(self):
        frontiers = self.get_frontiers()
        if not frontiers:
            return None

        def manhattan(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        # On choisit la frontière la plus proche
        closest = min(frontiers, key=lambda f: manhattan(self.pos, f))
        return closest
    
    def a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # On reconstruit le chemin
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = current[0] + dy, current[1] + dx
                neighbor = (ny, nx)
                if 0 <= ny < len(self.known_map) and 0 <= nx < len(self.known_map[0]):
                    if self.known_map[ny][nx] != 1:  # Peut marcher ici
                        tentative_g = g_score[current] + 1
                        if neighbor not in g_score or tentative_g < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f = tentative_g + abs(goal[0]-ny) + abs(goal[1]-nx)
                            heapq.heappush(open_set, (f, neighbor))
        return []

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

        # Step 1: Save the known map before update
        old_known_map = [row.copy() for row in explorer.known_map]

        # Step 2: Decide what to do
        if not explorer.path:
            # Pick the closest frontier centroid as target
            target = explorer.choose_target()

            if target:
                # Plan a path
                explorer.path = explorer.a_star(explorer.pos, target)

                # Optional: draw the target (e.g. yellow circle)
                tx, ty = target[1], target[0]
                tx_px = offset_x + tx * CELL_SIZE
                ty_px = offset_y + ty * CELL_SIZE
                pygame.draw.circle(screen, (255, 255, 0), (tx_px + CELL_SIZE // 2, ty_px + CELL_SIZE // 2), CELL_SIZE // 4)
        else:
            # Step forward on the path
            explorer.pos = explorer.path.pop(0)
            explorer.update_visibility()

        # Step 3: Check if anything new is discovered
        map_changed = any(
            old_known_map[y][x] != explorer.known_map[y][x]
            for y in range(len(explorer.known_map))
            for x in range(len(explorer.known_map[0]))
        )

        # Optional re-evaluation after new area is discovered
        if map_changed and not explorer.path:
            target = explorer.choose_target()
            if target:
                explorer.path = explorer.a_star(explorer.pos, target)


        draw_grid(screen, explorer.known_map, wall_img, floor_img)
        frontiers = explorer.get_frontiers()
        for fy, fx in frontiers:
            fx_px = offset_x + fx * CELL_SIZE
            fy_px = offset_y + fy * CELL_SIZE
            pygame.draw.circle(screen, (0, 255, 255), (fx_px + CELL_SIZE // 2, fy_px + CELL_SIZE // 2), CELL_SIZE // 4)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
