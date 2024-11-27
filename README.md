# Vexillium  - A Battle Strategy Game ⚔️🛡️🔥

<br>

## Overview  
*Vexillium* is a battle strategy game between a human player and AI. The goal is to outmaneuver your opponent and capture their flag. The game challenges players with strategic troop placement, resource management, and combat while providing adaptive AI with multiple difficulty levels.

<br>

## Features 
- **Grid-Based Gameplay:** A 15x10 grid where troops are placed and move towards targets.
- **AI Difficulty Levels:**
  - **Movement-Based:** AI uses different algorithms for troop movement:
    - Easy: Greedy BFS
    - Medium: UCS
    - Hard: A*
  - **Placement-Based:** The rate of AI troop placement varies with difficulty.
  - **Speed-Based:** Troop movement speed increases with difficulty.
- **Troop Types:**
  - **Circle Troop:** Power: 2, Cost: 2, Health: 300
  - **Square Troop:** Power: 3, Cost: 4, Health: 500
  - **Triangle Troop:** Power: 4, Cost: 3, Health: 400

## Technologies Used
- **Python 3.12**
- **Pygame:** For game development and rendering.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tactical-battle-game.git
   cd tactical-battle-game
2. Install the dependencies:
   ```bash
   pip install pygame
3. Run the game:
   ```bash
   python main.py

## How to Play 

### 1. Starting the Game:

- Choose a difficulty level: Easy , Medium , God Level.
- Place your flag on any row in your starting column.

### 2. Troop Deployment:

- Deploy troops within the first three columns of your side.
- Select troop types (Circle, Square, Triangle) based on available elixir and strategy
- The elixir is recharged for both teams at an equal rate.

### 3. Combat:

- Troops automatically move towards the opponent's flag or nearest enemy.
- Combat resolves when opposing troops meet:
   - Higher health troop wins, but its health is reduced.
   - If health is equal, both are removed.

### 4. Winning the Game:

- Capture the opponent’s flag by reaching it with your troop.

## AI Overview  

### Pathfinding Algorithms:

- Greedy BFS (Easy): Moves directly toward the target, prioritizing speed.
- Uniform Cost Search (Medium): Considers movement costs and obstacles.
- A* (God Level): Combines cost and heuristic for optimal paths.


### Minimax Algorithm:

- Determines troop placement by evaluating offensive and defensive strategies.
- Incorporates alpha-beta pruning for efficiency.

## License
This project is licensed under the [MIT License.](LICENSE)

## Ready, Set, CONQUER!
Happy gaming! 🎮
