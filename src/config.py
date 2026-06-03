from enum import Enum

GRID_SIZE = (12, 12)  # dimenzije sobe
NUM_AGENTS = 3  # broj roomba agenata
CELL_SIZE = 50  # veličina kvadrata u pygame
MAX_EPISODES: int = 300
REWARD_CLEAN = 1.0  # Stronger signal for cleaning new tiles
REWARD_ILLEGAL = -0.5  # Stronger penalty for hitting walls/agents
REWARD_CLEANED = -0.05  # Small penalty for revisiting
REWARD_WAIT = -0.10  # Small penalty for waiting
REWARD_COMPLETE = 200.0
FPS = 60  # brzina simulacije
STEP_INTERVAL = 0.00005  # U sekundama
WALL_FRACTION = 0.0
MAX_STEP = 400
OBS_DIM = 3 * GRID_SIZE[0] * GRID_SIZE[1]  # 75
GLOBAL_STATE_DIM = 2 * GRID_SIZE[0] * GRID_SIZE[1] + 2 * NUM_AGENTS  # 54
N_ACTIONS = 5
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995  # Per episode decay: reaches ~0.05 after 600 episodes
epsilon = EPSILON_START


class Akcija(Enum):
    GORE = 0
    DOLE = 1
    LIJEVO = 2
    DESNO = 3
    CEKAJ = 4
