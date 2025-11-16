"""Configuration file for AlphaZero Chess Engine"""

# Neural Network Architecture
NUM_RESIDUAL_BLOCKS = 19
NUM_FILTERS = 256
BOARD_SIZE = 8
NUM_PLANES = 18  # 6 piece types * 2 colors + 2 history planes

# MCTS Parameters
NUM_MCTS_SIMS = 800  # Number of MCTS simulations per move
CPUCT = 1.0  # PUCT exploration constant
DIRICHLET_ALPHA = 0.3  # Dirichlet noise alpha for root exploration
DIRICHLET_EPSILON = 0.25  # Dirichlet noise weight
TEMPERATURE = 1.0  # Temperature for move selection

# Training Parameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
TRAIN_EXAMPLES_HISTORY = 20  # Number of recent game iterations to keep

# Self-Play Parameters
NUM_SELF_PLAY_GAMES = 100  # Games per iteration
MAX_GAME_LENGTH = 512  # Maximum moves per game

# Evaluation Parameters
EVAL_GAMES = 40  # Number of games for evaluation
WIN_THRESHOLD = 0.55  # Win rate threshold to accept new model

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
DATA_DIR = "training_data"
LOG_DIR = "logs"

# Device
DEVICE = "cuda"  # or "cpu"