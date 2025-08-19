import os, math, heapq, json, time, random
from collections import deque

# Ensure the window always opens at the same screen position
os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'

import pygame

#configuration
DIM_X, DIM_Y = 20, 20
CELL         = 30
W, H         = DIM_X * CELL, DIM_Y * CELL

WHITE      = (255,255,255)
GRAY       = (200,200,200)
BLACK      = (  0,  0,  0)
RED        = (255,  0,  0)
DARK_GREEN = (  0,180,  0)
BLUE       = (  0,  0,255)
YELLOW     = (255,255,  0)
PURPLE     = (128,  0,128)
ORANGE     = (255,165,  0)

# terrain_map[x][y] , cost multiplier
terrain_map    = [[1 for _ in range(DIM_Y)] for _ in range(DIM_X)]
terrain_colors = {
    1: WHITE,
    2: (144,238,144),
    3: (189,183,107),
    4: (169,169,169),
}

# initial state
start            = (0,0)
targets          = []
static_obstacles = set()
missile_tracks   = set()
selected_cell    = None    #it is for terrain painting

# logging globals (it atumatically reset each mission)
last_path       = []
broken_by_fire  = []
path_lengths    = []
elapsed_time    = 0.0

# 8 way moves 
MOVES = [
    (1,0,1.0),   (-1,0,1.0),  (0,1,1.0),   (0,-1,1.0),
    (1,1,math.sqrt(2)), (1,-1,math.sqrt(2)),
    (-1,1,math.sqrt(2)),(-1,-1,math.sqrt(2)),
]

#pygame
pygame.init()
screen = pygame.display.set_mode((W,H), pygame.DOUBLEBUF)
pygame.display.set_caption("Missile Navigation System")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont(None, 24)

#drawing utilities
def draw_grid():
    for x in range(0, W, CELL):
        pygame.draw.line(screen, GRAY, (x,0), (x,H))
    for y in range(0, H, CELL):
        pygame.draw.line(screen, GRAY, (0,y), (W,y))

def draw_ui():
    mx, my = pygame.mouse.get_pos()
    gx, gy = mx//CELL, my//CELL
    if 0 <= gx < DIM_X and 0 <= gy < DIM_Y:
        tc = terrain_map[gx][gy]
        txt0 = f"Terrain({gx},{gy}) cost={tc}"
    else:
        txt0 = "Terrain: N/A"
    txt1 = f"Last path len: {len(last_path)}"
    txt2 = f"Missile breaks: {len(broken_by_fire)}"
    txt3 = f"Elapsed: {elapsed_time:.2f}s"
    for i, t in enumerate((txt0, txt1, txt2, txt3)):
        surf = font.render(t, True, BLACK)
        screen.blit(surf, (5, 5 + i*20))

def draw_elements(path=None, missile_pos=None):
    screen.fill(WHITE)
    # terrain cells
    for x in range(DIM_X):
        for y in range(DIM_Y):
            col = terrain_colors[terrain_map[x][y]]
            pygame.draw.rect(screen, col, (x*CELL, y*CELL, CELL, CELL))

    # selected cell highlight
    if selected_cell:
        sx, sy = selected_cell
        pygame.draw.rect(screen, ORANGE, (sx*CELL, sy*CELL, CELL, CELL), 3)

    draw_grid()

    # static obstacles
    for ox, oy in static_obstacles:
        pygame.draw.rect(screen, RED, (ox*CELL, oy*CELL, CELL, CELL))

    # missile tracks
    for tx, ty in missile_tracks:
        pygame.draw.rect(screen, PURPLE, (tx*CELL, ty*CELL, CELL, CELL))

    # start & targets
    pygame.draw.rect(screen, BLUE, (start[0]*CELL, start[1]*CELL, CELL, CELL))
    for tgt in targets:
        col = BLACK if tgt in missile_tracks else DARK_GREEN
        pygame.draw.rect(screen, col, (tgt[0]*CELL, tgt[1]*CELL, CELL, CELL))

    # current missile
    if missile_pos:
        pygame.draw.rect(screen, BLACK, (missile_pos[0]*CELL, missile_pos[1]*CELL, CELL, CELL))

    draw_ui()
    pygame.display.flip()

# -------------------
# EFFECTS
# -------------------
def explosion_effect(pos):
    cx = pos[0]*CELL + CELL//2
    cy = pos[1]*CELL + CELL//2
    for r in range(5, CELL*2, 5):
        draw_elements()
        pygame.draw.circle(screen, YELLOW, (cx,cy), r)
        pygame.display.flip()
        pygame.event.pump()
        clock.tick(60)

def crumbling_effect(cell):
    """Animate a 2×2 crumble of the obstacle at `cell`."""
    cx, cy = cell
    x0, y0 = cx*CELL, cy*CELL
    size   = CELL // 2
    pieces = []
    for ix in (0,1):
        for iy in (0,1):
            rect = pygame.Rect(x0 + ix*size, y0 + iy*size, size, size)
            vel  = [random.uniform(-2,2), random.uniform(-5,-1)]
            pieces.append({'rect': rect, 'vel': vel})

    for _ in range(20):
        draw_elements()
        for p in pieces:
            p['vel'][1] += 0.3
            p['rect'].x += int(p['vel'][0])
            p['rect'].y += int(p['vel'][1])
            pygame.draw.rect(screen, RED, p['rect'])
        pygame.display.flip()
        pygame.event.pump()
        clock.tick(30)

# -------------------
# HEURISTIC & PATHFINDING
# -------------------
def heuristic(a, b):
    dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
    return (dx+dy) + (math.sqrt(2)-2)*min(dx,dy)

def astar_with_breaks(start, goal, break_cost=8):
    """
    A* that may traverse obstacles at an extra fixed cost.
    Returns a path including obstacle cells when optimal.
    """
    global last_path
    open_heap = [(0, start)]
    g_score   = {start: 0}
    came_from = {}
    closed    = set()

    while open_heap:
        f, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            # reconstruct
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            last_path = path[::-1]
            return last_path

        closed.add(current)
        for dx, dy, cost in MOVES:
            nx, ny = current[0]+dx, current[1]+dy
            if not (0 <= nx < DIM_X and 0 <= ny < DIM_Y):
                continue

            step_cost = cost * terrain_map[nx][ny]
            if (nx, ny) in static_obstacles:
                step_cost += break_cost

            tentative = g_score[current] + step_cost
            if (nx,ny) not in g_score or tentative < g_score[(nx,ny)]:
                g_score[(nx,ny)] = tentative
                came_from[(nx,ny)] = current
                heapq.heappush(open_heap, (tentative + heuristic((nx,ny), goal), (nx,ny)))

    last_path = []
    return []

# -------------------
# MISSILE & STRIKE
# -------------------
def fire_missile(path):
    for pos in path:
        missile_tracks.add(pos)
        draw_elements(path=path, missile_pos=pos)
        pygame.event.pump()
        clock.tick(30)
        if pos in static_obstacles:
            crumbling_effect(pos)
            static_obstacles.discard(pos)
            broken_by_fire.append(pos)
            print(f"  -> Broke obstacle at {pos}")

def strike_targets():
    global broken_by_fire, path_lengths, elapsed_time
    broken_by_fire = []
    path_lengths   = []
    start_time     = time.perf_counter()
    original_targets = list(targets)

    for tgt in original_targets:
        print(f"\nTarget {tgt}: planning with breaks-allowed A*")
        path = astar_with_breaks(start, tgt, break_cost=8)
        if not path:
            print("  No path found even allowing breaks → skipping.")
            path_lengths.append(0)
            continue

        print(f"  Path length {len(path)}; firing missile…")
        path_lengths.append(len(path))
        fire_missile(path)
        explosion_effect(tgt)
        draw_elements()
        clock.tick(30)

    elapsed_time = time.perf_counter() - start_time
    return {
        "start": start,
        "targets": original_targets,
        "broken_by_fire": broken_by_fire,
        "path_lengths": path_lengths,
        "elapsed_time_sec": round(elapsed_time, 3)
    }

# -------------------
# LOGGING
# -------------------
def log_mission(data):
    with open("mission_logs.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")
    print("Mission logged to mission_logs.jsonl")

# -------------------
# UI/PLACEMENT & MAIN
# -------------------
def place_elements():
    global selected_cell
    placing = True
    while placing:
        draw_elements()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return False
            if e.type == pygame.MOUSEBUTTONDOWN:
                gx, gy = e.pos[0]//CELL, e.pos[1]//CELL
                if not (0 <= gx < DIM_X and 0 <= gy < DIM_Y):
                    continue
                if e.button == 1:  # left: toggle obstacle
                    if (gx,gy) in targets:
                        targets.remove((gx,gy))
                    else:
                        static_obstacles.symmetric_difference_update({(gx,gy)})
                elif e.button == 2:  # middle: select terrain‐paint cell
                    selected_cell = (gx,gy)
                elif e.button == 3:  # right: place target
                    if (gx,gy) not in static_obstacles and (gx,gy) not in targets:
                        targets.append((gx,gy))
            if e.type == pygame.KEYDOWN and selected_cell:
                x, y = selected_cell
                if e.key == pygame.K_i:
                    terrain_map[x][y] = 1
                elif e.key == pygame.K_j:
                    terrain_map[x][y] = 2
                elif e.key == pygame.K_k:
                    terrain_map[x][y] = 3
                elif e.key == pygame.K_l:
                    terrain_map[x][y] = 4
            if e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN and targets:
                placing = False
    return True

def wait_for_restart():
    ft = pygame.font.SysFont(None,32)
    t1 = ft.render("Mission Complete!", True, BLACK)
    t2 = ft.render("Press R to Restart or Q to Quit", True, BLACK)
    screen.blit(t1, (W//2-100, H//2-40))
    screen.blit(t2, (W//2-150, H//2))
    pygame.display.flip()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    return True
                if e.key == pygame.K_q:
                    return False

# MAIN LOOP
running = True
while running:
    if not place_elements():
        break
    if targets:
        mission_log = strike_targets()
        log_mission(mission_log)
    else:
        print("No targets selected.")
    running = wait_for_restart()
    targets.clear()
    missile_tracks.clear()
    selected_cell = None

pygame.quit()