
import pygame
import random
import heapq
import copy
import asyncio 

# Constants
WIDTH, HEIGHT = 750, 500
GRID_COLS, GRID_ROWS = 15, 10
CELL_SIZE = WIDTH // GRID_COLS
MAX_ELIXIR = 10
INFO_PANEL_WIDTH = 100
TROOP_MOVE_DELAY = 25
ELIXIR_RECHARGE_RATE = 0.25  # Increase this to speed up elixir recharge
AI_SPAWN_DELAY = 2000
MAX_DEPTH = 2
BOTTOM_PANEL_WIDTH=150

# Define positions for troop options in the bottom panel as global constants
TROOP_OPTIONS_POSITIONS = [
    WIDTH // 4,
    WIDTH // 2,
    3 * WIDTH // 4
]
TROOP_OPTION_Y_POSITION = HEIGHT + 30  # Vertical position for troop options

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# RED = (255, 43, 100)
# BLUE = (0, 128, 255)

BLUE = (90, 29, 255)
RED = (254,76,255)

#GREEN = (0, 255, 0)
GREEN = (0, 204, 204)

PURPLE = (149,43,255)
LIGHT_PURPLE=(227,206,255)
GRAY = (169, 169, 169)
YELLOW = (255, 255, 0)
LIGHTEST_PURPLE = (234,232,253)
# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH + 2 * INFO_PANEL_WIDTH, HEIGHT+BOTTOM_PANEL_WIDTH))
pygame.display.set_caption("Tactical Battle Game")
clock = pygame.time.Clock()

# Troop class
class Troop:
    def __init__(self, x, y, power, health, cost, shape, team):
        self.x, self.y = x, y
        self.power = power
        self.health = health
        self.cost = cost
        self.shape = shape
        self.team = team  # "player" or "ai"
        self.alive = True
        self.move_counter = 0
        self.paused = True  # New attribute to pause troop initially
        self.pause_duration = 20


    async def move_toward(self, targets, obstacles, team, difficulty):
        # Pausing if needed
        if self.paused:
            self.pause_duration -= 1
            if self.pause_duration <= 0:
                self.paused = False
            return

        # Check move delay
        if difficulty == 'h':
            TROOP_MOVE_DELAY = 10
        else:
            TROOP_MOVE_DELAY = 25

        if self.move_counter % TROOP_MOVE_DELAY == 0:
            path = None
            if team == 'ai':
                if difficulty == 'e':
                    path = await asyncio.to_thread(greedy_bfs, (self.x, self.y), targets, obstacles)
                elif difficulty == 'm':
                    path = await asyncio.to_thread(uniform_cost_search, (self.x, self.y), targets, obstacles)
                else:
                    path = await asyncio.to_thread(a_star, (self.x, self.y), targets, obstacles)
            else:
                path = await asyncio.to_thread(a_star, (self.x, self.y), targets, obstacles)
            
            if path and len(path) > 1:
                self.x, self.y = path[1]
        self.move_counter += 1

# Square troop subclass
class Square(Troop):
    def __init__(self, x, y, team):
        super().__init__(x, y, power=3, health=500, cost=4, shape='square', team=team)  # Assign specific values for Square troops

# Triangle troop subclass
class Triangle(Troop):
    def __init__(self, x, y, team):
        super().__init__(x, y, power=4, health=400, cost=3,shape='triangle', team=team)  # Assign specific values for Triangle troops

# Circle troop subclass
class Circle(Troop):
    def __init__(self, x, y, team):
        super().__init__(x, y, power=2, health=300, cost=2,shape='circle', team=team)  # Assign specific values for Circle troops

def evaluate_game_state(game_state):
    """Evaluate the game state from the AI's perspective."""
    score = 0
    # Factor in distance to player flag
    for troop in game_state.ai_troops:
        score -= manhattan_distance((troop.x, troop.y), game_state.player_flag)
    # Factor in distance of player troops to AI flag
    for troop in game_state.player_troops:
        score += manhattan_distance((troop.x, troop.y), game_state.ai_flag)
    # Factor in number of AI troops and health
    score += len(game_state.ai_troops) * 10
    score -= len(game_state.player_troops) * 10
    return score


async def minimax(game_state, depth, alpha, beta, is_maximizing_player):
    if depth == 0 or game_state.game_over:
        return evaluate_game_state(game_state)

    if is_maximizing_player:
        max_eval = float('-inf')
        for troop_type in ["circle", "square", "triangle"]:
            for x, y in get_possible_positions(game_state, "ai"):
                simulated_state = simulate_troop_placement(game_state, x, y, "ai", troop_type)
                eval = await minimax(simulated_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
        return max_eval
    else:
        min_eval = float('inf')
        for troop_type in ["circle", "square", "triangle"]:
            for x, y in get_possible_positions(game_state, "player"):
                simulated_state = simulate_troop_placement(game_state, x, y, "player", troop_type)
                eval = await minimax(simulated_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
        return min_eval

async def get_best_move(game_state):
    """Find the best AI troop placement move considering troop type and position using minimax."""
    best_score = float('-inf')
    best_move = None
    for troop_type in ["circle", "square", "triangle"]:
        for x, y in get_possible_positions(game_state, "ai"):
            simulated_state = simulate_troop_placement(game_state, x, y, "ai", troop_type)
            move_score = await minimax(simulated_state, MAX_DEPTH - 1, float('-inf'), float('inf'), False)
            if move_score > best_score:
                best_score = move_score
                best_move = (x, y, troop_type)  # Store position and troop type
    return best_move

def get_possible_positions(game_state, team):
    """Generate possible positions, prioritizing positions closer to the enemy flag."""
    if team == "ai":
        positions = [(x, y) for x in range(GRID_COLS - 3, GRID_COLS) for y in range(GRID_ROWS) if (x,y) not in game_state.obstacles]
    else:  # For player
        positions = [(x, y) for x in range(3) for y in range(GRID_ROWS) if (x,y) not in game_state.obstacles]

    # Filter top moves closest to the target flag
    enemy_troops = game_state.ai_troops if team == "player" else game_state.player_troops
    enemy_targets = [(troop.x, troop.y) for troop in enemy_troops if troop.alive]

            # List of targets: first the flag, then enemy troops
    target_flag = game_state.player_flag if team == "ai" else game_state.ai_flag
    targets = [target_flag] + enemy_targets
    pos5 = []
    for pos in positions:
        for targ in targets:
            pos5.append((manhattan_distance(pos, targ), pos))
    pos5 = sorted(pos5)[:5]
    pos = [x[1] for x in pos5]
    #positions = sorted(positions, key=lambda pos: manhattan_distance(pos, target_flag))[:5]  # Top 5 closest positions
    return pos

def simulate_troop_placement(game_state, x, y, team, troop_type):
    """Simulate placing a specific type of troop at (x, y) and return the new game state."""
    new_state = copy.deepcopy(game_state)  # Assuming `deepcopy` is imported and used to avoid altering the real game state
    
    # Add troop based on type
    if troop_type == "circle":
        troop = Circle(x, y, team)
    elif troop_type == "square":
        troop = Square(x, y, team)
    elif troop_type == "triangle":
        troop = Triangle(x, y, team)

    if team == "ai" and new_state.ai_elixir >= troop.cost:
        new_state.ai_troops.append(troop)
        new_state.ai_elixir -= troop.cost
    elif team == "player" and new_state.player_elixir >= troop.cost:
        new_state.player_troops.append(troop)
        new_state.player_elixir -= troop.cost

    return new_state

def manhattan_distance(a, b):
    """Calculate Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
# Game State
class GameState:
    def __init__(self):
        self.player_flag = None
        self.ai_flag =(GRID_COLS - 1, random.randint(0,GRID_ROWS - 1))
        self.player_troops = []
        self.ai_troops = []
        self.obstacles = [(random.randint(0, GRID_COLS - 1), random.randint(0, GRID_ROWS - 1)) for _ in range(10)]
        self.player_elixir = 0
        self.ai_elixir = 0
        self.game_over = False
        self.winner = None

    def add_troop(self, x, y, team,shape=None):
        if team == "player" and self.player_elixir > 0:
            # Create troop based on selected shape
            if shape == "circle":
                troop = Circle(x, y, team)
            elif shape == "square":
                troop = Square(x, y, team)
            elif shape == "triangle":
                troop = Triangle(x, y, team)
            
            # Deduct elixir if enough is available
            if self.player_elixir >= troop.cost:
                self.player_troops.append(troop)
                self.player_elixir -= troop.cost
                return True
            return False
        
        elif team == "ai" and self.ai_elixir > 0:
            shape = random.choice(["circle", "square", "triangle"])
            if shape == "circle":
                troop = Circle(x, y, team)
            elif shape == "square":
                troop = Square(x, y, team)
            elif shape == "triangle":
                troop = Triangle(x, y, team)
            
            if self.ai_elixir >= troop.cost:
                self.ai_troops.append(troop)
                self.ai_elixir -= troop.cost



background_image = pygame.image.load(r"D:\Users\karpa\Pappu\SSN files\SEM - 5\AI\Game\bg.png")
background_image = pygame.transform.scale(background_image, (WIDTH + 2 * INFO_PANEL_WIDTH, HEIGHT + BOTTOM_PANEL_WIDTH))
grid_area = pygame.Rect(INFO_PANEL_WIDTH, 0, WIDTH, HEIGHT)

def draw_grid():
    # First, draw the background image
    screen.blit(background_image, grid_area, grid_area)

                        # +CELL_SIZE is for last line
    for x in range(0, WIDTH+CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x + INFO_PANEL_WIDTH, 0), (x + INFO_PANEL_WIDTH, HEIGHT))
    for y in range(0, HEIGHT+CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (INFO_PANEL_WIDTH, y), (WIDTH + INFO_PANEL_WIDTH, y))

def draw_obstacles(obstacles):
    for obstacle in obstacles:
        pygame.draw.rect(screen, BLACK, (obstacle[0] * CELL_SIZE + INFO_PANEL_WIDTH, obstacle[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def draw_flag(flag, color):
    if flag:
        pygame.draw.rect(screen, color, (flag[0] * CELL_SIZE + INFO_PANEL_WIDTH, flag[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def draw_troops(troops, color):
    for troop in troops:
        if troop.alive:
            x, y = troop.x * CELL_SIZE + CELL_SIZE // 2 + INFO_PANEL_WIDTH, troop.y * CELL_SIZE + CELL_SIZE // 2
            if troop.shape == "circle":
                pygame.draw.circle(screen, color, (x, y), CELL_SIZE // 3)
            elif troop.shape == "square":
                pygame.draw.rect(screen, color, (troop.x * CELL_SIZE + CELL_SIZE // 4 + INFO_PANEL_WIDTH, troop.y * CELL_SIZE + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2))
            elif troop.shape == "triangle":
                pygame.draw.polygon(screen, color, [(x, y - CELL_SIZE // 3), (x - CELL_SIZE // 3, y + CELL_SIZE // 3), (x + CELL_SIZE // 3, y + CELL_SIZE // 3)])

def draw_elixir_bar(current_elixir, x_pos):
    elixir_height = int((HEIGHT - 20) * (current_elixir / MAX_ELIXIR))
    pygame.draw.rect(screen, LIGHT_PURPLE, (x_pos, 10, 20, HEIGHT - 20))  # Bar outline
    pygame.draw.rect(screen, PURPLE, (x_pos, HEIGHT - elixir_height - 10, 20, elixir_height))
    font = pygame.font.SysFont(None, 24)
    elixir_text = font.render(f"{int(current_elixir)}/{MAX_ELIXIR}", True, BLACK)
    screen.blit(elixir_text, (x_pos - 10, HEIGHT - elixir_height - 25))  # Show elixir value


def draw_troop_options():
    font = pygame.font.SysFont(None, 24)
    
    # Draw Circle troop option
    pygame.draw.circle(screen, BLUE, (TROOP_OPTIONS_POSITIONS[0], TROOP_OPTION_Y_POSITION), CELL_SIZE // 3)
    circle_texts = ["Power: 2", "Cost: 2", "Health: 300"]
    for i, text in enumerate(circle_texts):
        line = font.render(text, True, BLACK)
        screen.blit(line, (TROOP_OPTIONS_POSITIONS[0] - line.get_width() // 2, TROOP_OPTION_Y_POSITION + 20 + i * 20))

    # Draw Square troop option
    pygame.draw.rect(screen, BLUE, (TROOP_OPTIONS_POSITIONS[1] - CELL_SIZE // 4, TROOP_OPTION_Y_POSITION - CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2))
    square_texts = ["Power: 3", "Cost: 4", "Health: 500"]
    for i, text in enumerate(square_texts):
        line = font.render(text, True, BLACK)
        screen.blit(line, (TROOP_OPTIONS_POSITIONS[1] - line.get_width() // 2, TROOP_OPTION_Y_POSITION + 20 + i * 20))

    # Draw Triangle troop option
    pygame.draw.polygon(screen, BLUE, [
        (TROOP_OPTIONS_POSITIONS[2], TROOP_OPTION_Y_POSITION - CELL_SIZE // 3),
        (TROOP_OPTIONS_POSITIONS[2] - CELL_SIZE // 3, TROOP_OPTION_Y_POSITION + CELL_SIZE // 3),
        (TROOP_OPTIONS_POSITIONS[2] + CELL_SIZE // 3, TROOP_OPTION_Y_POSITION + CELL_SIZE // 3)
    ])
    triangle_texts = ["Power: 4", "Cost: 3", "Health: 400"]
    for i, text in enumerate(triangle_texts):
        line = font.render(text, True, BLACK)
        screen.blit(line, (TROOP_OPTIONS_POSITIONS[2] - line.get_width() // 2, TROOP_OPTION_Y_POSITION + 20 + i * 20))

# A* Search Algorithm (for troop movement)
def a_star(start, goals, obstacles):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    closest_goal = None

    # Set of goals for quick lookup
    goal_set = set(goals)
    while open_list:
        _, current = heapq.heappop(open_list)

        # If the current node is a goal, stop search
        if current in goal_set:
            closest_goal = current
            break

        neighbors = get_neighbors(current)
        for next in neighbors:
            if next in obstacles:
                continue
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + min(heuristic(next, goal) for goal in goals)
                heapq.heappush(open_list, (priority, next))
                came_from[next] = current

    # Reconstruct the path to the closest goal if found
    path = []
    if closest_goal:
        node = closest_goal
        while node:
            path.append(node)
            node = came_from[node]
        path.reverse()

    return path

# Uniform Cost Search (UCS) Algorithm for troop movement
def uniform_cost_search(start, goals, obstacles):
    open_list = []
    heapq.heappush(open_list, (0, start))  # (cost, position)
    came_from = {start: None}
    cost_so_far = {start: 0}
    closest_goal = None

    # Set of goals for quick lookup
    goal_set = set(goals)

    while open_list:
        current_cost, current = heapq.heappop(open_list)

        # If the current node is a goal, stop search
        if current in goal_set:
            closest_goal = current
            break

        neighbors = get_neighbors(current)
        for next in neighbors:
            if next in obstacles:
                continue
            new_cost = current_cost + 1  # Cost of moving to the next node
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                heapq.heappush(open_list, (new_cost, next))
                came_from[next] = current

    # Reconstruct the path to the closest goal if found
    path = []
    if closest_goal:
        node = closest_goal
        while node:
            path.append(node)
            node = came_from[node]
        path.reverse()
    return path



# Greedy Best-First Search (Greedy BFS) for troop movement
def greedy_bfs(start, goals, obstacles):
    open_list = []
    # Push the start node with heuristic only (ignoring path cost)
    heapq.heappush(open_list, (0, start))
    came_from = {start: None}
    closest_goal = None

    # Set of goals for quick lookup
    goal_set = set(goals)

    while open_list:
        _, current = heapq.heappop(open_list)

        # If the current node is a goal, stop search
        if current in goal_set:
            closest_goal = current
            break

        neighbors = get_neighbors(current)
        for next in neighbors:
            if next in obstacles or next in came_from:
                continue

            # Only consider the heuristic for the priority
            priority = min(heuristic(next, goal) for goal in goals)
            heapq.heappush(open_list, (priority, next))
            came_from[next] = current

    # Reconstruct the path to the closest goal if found
    path = []
    if closest_goal:
        node = closest_goal
        while node:
            path.append(node)
            node = came_from[node]
        path.reverse()
    return path




def get_neighbors(pos):
    x, y = pos
    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    return [(nx, ny) for nx, ny in neighbors if 0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS]

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

async def move_all_troops(game_state, difficulty, obstacles):
    # Gather all the troop movement tasks
    move_tasks = []
    
    for troop in game_state.player_troops + game_state.ai_troops:
        if troop.alive:
            target_flag = game_state.ai_flag if troop.team == "player" else game_state.player_flag

            # Get the enemy troops that the current troop needs to interact with
            enemy_troops = game_state.ai_troops if troop.team == "player" else game_state.player_troops
            enemy_targets = [(troop.x, troop.y) for troop in enemy_troops if troop.alive]

            # List of targets: first the flag, then enemy troops
            targets = [target_flag] + enemy_targets

            # Add the troop's move_toward task to the list
            move_tasks.append(
                troop.move_toward(targets, obstacles, troop.team, difficulty)
            )

    # Run all troop movements concurrently
    await asyncio.gather(*move_tasks)

    # Check if any troop has reached the opponent's flag
    for troop in game_state.player_troops + game_state.ai_troops:
        if troop.alive and (troop.x, troop.y) == (game_state.ai_flag if troop.team == "player" else game_state.player_flag):
            game_state.game_over = True
            game_state.winner = "You" if troop.team == "player" else "AI"

async def start_game(difficulty):
    # Main Game Loop
    if difficulty == 'e':
        AI_SPAWN_DELAY = 6000
    elif difficulty == 'm':
        AI_SPAWN_DELAY = 4000
    else:
        AI_SPAWN_DELAY = 2000
    game_state = GameState()
    placing_flag = True
    running = True


    last_ai_spawn_time = pygame.time.get_ticks()  # Track time since the last AI spawn


    selected_troop_shape=None

    while running:
        screen.fill(LIGHTEST_PURPLE)
        draw_grid()
        draw_obstacles(game_state.obstacles)
        draw_flag(game_state.player_flag, BLUE)
        draw_flag(game_state.ai_flag, RED)
        draw_troops(game_state.player_troops, BLUE)
        draw_troops(game_state.ai_troops, RED)
        draw_elixir_bar(game_state.player_elixir, 20)  # Player elixir bar on the left
        draw_elixir_bar(game_state.ai_elixir, WIDTH + INFO_PANEL_WIDTH + 60)  # AI elixir bar on the right
        draw_troop_options()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_state.game_over:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                
                grid_x, grid_y = (mouse_x - INFO_PANEL_WIDTH) // CELL_SIZE, mouse_y // CELL_SIZE
                #print(grid_x, grid_y)
                if placing_flag and grid_x == 0 and mouse_y <= HEIGHT:
                    if (grid_x, grid_y) not in game_state.obstacles:
                        game_state.player_flag = (grid_x, grid_y)
                        placing_flag = False

                # Selecting troops
                elif HEIGHT <= mouse_y <= HEIGHT + 60 and game_state.player_flag:
                    
                    if TROOP_OPTIONS_POSITIONS[0] - 20 <= mouse_x <= TROOP_OPTIONS_POSITIONS[0] + 20:
                        selected_troop_shape = "circle"
                    elif TROOP_OPTIONS_POSITIONS[1] - 20 <= mouse_x <= TROOP_OPTIONS_POSITIONS[1] + 20:
                        selected_troop_shape = "square"
                    elif TROOP_OPTIONS_POSITIONS[2] - 20 <= mouse_x <= TROOP_OPTIONS_POSITIONS[2] + 20:
                        selected_troop_shape = "triangle"

                # Place troop on grid if a troop shape is selected and elixir is sufficient
                elif selected_troop_shape and grid_x < 3 and mouse_y <= HEIGHT and mouse_x >= INFO_PANEL_WIDTH:
                    grid_x, grid_y = (mouse_x - INFO_PANEL_WIDTH) // CELL_SIZE, mouse_y // CELL_SIZE
                    if (grid_x, grid_y) not in game_state.obstacles:
                        game_state.add_troop(grid_x, grid_y, team="player", shape=selected_troop_shape)
                        


                """ elif game_state.player_flag and grid_x < 3:
                    if (grid_x, grid_y) not in game_state.obstacles:
                        game_state.add_troop(grid_x, grid_y, team="player") """
                
                

        """ if game_state.player_flag and not placing_flag:


            ai_x, ai_y = random.randint(GRID_COLS - 3, GRID_COLS - 1), random.randint(0,GRID_ROWS - 1)
            if (ai_x, ai_y) not in game_state.obstacles:
                game_state.add_troop(ai_x, ai_y, team="ai") """
        '''
        easy = 0.5
        med = 0.3
        hard
        '''    
            

        # Inside the main game loop
        if game_state.player_flag and not placing_flag:
            current_time = pygame.time.get_ticks()
            if current_time - last_ai_spawn_time >= AI_SPAWN_DELAY:
                if difficulty == 'e':
                    if random.random() > 0.8:
                        best_move = await get_best_move(game_state)
                        if best_move:
                            x, y, troop_type = best_move
                            game_state.add_troop(x, y, "ai", troop_type)
                        last_ai_spawn_time = current_time

                elif difficulty == 'm':
                    if random.random() > 0.5:
                        best_move = await get_best_move(game_state)
                        if best_move:
                            x, y, troop_type = best_move
                            game_state.add_troop(x, y, "ai", troop_type)
                        last_ai_spawn_time = current_time
                else:
                    best_move = await get_best_move(game_state)
                    if best_move:
                        x, y, troop_type = best_move
                        game_state.add_troop(x, y, "ai", troop_type)
                    last_ai_spawn_time = current_time

        obstacles = set(game_state.obstacles)
        await move_all_troops(game_state, difficulty, obstacles)


        # Handle combat when troops meet
        for player_troop in game_state.player_troops:
            for ai_troop in game_state.ai_troops:
                if player_troop.alive and ai_troop.alive and (player_troop.x, player_troop.y) == (ai_troop.x, ai_troop.y):
                    # Troops fight; reduce health based on each other's power
                    player_troop.health -= ai_troop.power
                    ai_troop.health -= player_troop.power

                    # Determine which troop, if any, is defeated
                    if player_troop.health <= 0:
                        player_troop.alive = False
                    if ai_troop.health <= 0:
                        ai_troop.alive = False

        # Check if the game is over and display the winner message
        if game_state.game_over:
            font = pygame.font.SysFont(None, 48)
            text = font.render(f"{game_state.winner} Win!", True, YELLOW)
            text_rect = text.get_rect(center=(WIDTH // 2 + INFO_PANEL_WIDTH, HEIGHT // 2))
            screen.blit(text, text_rect)
        else:
            # Recharge elixir for both players if the game is still running
            if pygame.time.get_ticks() % 1000 < 30:
                game_state.player_elixir = min(MAX_ELIXIR, game_state.player_elixir + ELIXIR_RECHARGE_RATE)
                game_state.ai_elixir = min(MAX_ELIXIR, game_state.ai_elixir + ELIXIR_RECHARGE_RATE)

        # Refresh the display and control frame rate
        pygame.display.flip()
        clock.tick(240)

        # Pause the game when game over
        if game_state.game_over:
            pygame.time.delay(3000)  # Display the message for 2 seconds
            running = False

        game_state.player_troops = [troop for troop in game_state.player_troops if troop.alive]
        game_state.ai_troops = [troop for troop in game_state.ai_troops if troop.alive]

    #pygame.quit()


#start_game()
