# AlphaZero Chess Engine

A complete implementation of the AlphaZero algorithm for chess using PyTorch, featuring:
- Monte Carlo Tree Search (MCTS) with PUCT
- Deep neural network with residual architecture
- Self-play training pipeline
- Model evaluation arena

## Project Structure

```
chesshacks/
├── config.py                      # Configuration and hyperparameters
├── chess_board.py                 # Chess game rules and move generation
├── encoder_decoder.py             # Board encoding/decoding for neural network
├── alpha_net.py                   # AlphaGoZero neural network (19 residual blocks)
├── MCTS_chess.py                  # Monte Carlo Tree Search with PUCT
├── train.py                       # Neural network training
├── train_multiprocessing.py       # Parallel self-play training
├── evaluator.py                   # Model evaluation arena
├── pipeline.py                    # Full training pipeline orchestrator
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Components

### 1. Chess Board (`chess_board.py`)
- Complete chess rules implementation using `python-chess` library
- Move generation and validation
- Game state management
- Board encoding utilities

### 2. Encoder/Decoder (`encoder_decoder.py`)
- Encodes board state into 18-plane tensor representation:
  - 12 planes for pieces (6 types × 2 colors)
  - 2 planes for castling rights
  - 2 planes for en passant
  - 1 plane for turn color
  - 1 plane for move count
- Converts between chess moves and neural network outputs
- Policy vector creation and decoding

### 3. Neural Network (`alpha_net.py`)
AlphaGoZero architecture with:
- **Input**: 18×8×8 board representation
- **Architecture**:
  - Initial convolutional block (batch norm + ReLU)
  - 19 residual blocks (each with 2 conv layers)
  - 256 convolutional filters throughout
- **Output Heads**:
  - Policy head: 4096-dimensional move probabilities (log softmax)
  - Value head: Position evaluation in range [-1, 1] (tanh)

### 4. MCTS (`MCTS_chess.py`)
Monte Carlo Tree Search implementation featuring:
- PUCT (Polynomial Upper Confidence Trees) for node selection
- Dirichlet noise for root exploration
- Neural network guided search
- Self-play game generation

### 5. Training (`train.py`)
- PyTorch training loop with mixed loss (policy + value)
- KL divergence loss for policy
- MSE loss for value
- Adam optimizer with weight decay
- Batch training with data loader

### 6. Multiprocessing (`train_multiprocessing.py`)
- Parallel self-play using multiple CPU workers
- Efficient data generation across cores
- Queue-based result collection

### 7. Evaluator (`evaluator.py`)
- Arena for comparing model versions
- Head-to-head game evaluation
- Win rate calculation
- Model replacement logic

### 8. Pipeline (`pipeline.py`)
Full training iteration pipeline:
1. Self-play using MCTS to generate training data
2. Train neural network on generated data
3. Evaluate new model against previous best
4. Keep best performing model

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- python-chess
- NumPy
- tqdm

## Usage

### Quick Start

Run the full training pipeline:

```bash
python pipeline.py --iterations 10 --games 50 --evaluate
```

### Command Line Arguments

```bash
python pipeline.py [OPTIONS]

Options:
  --iterations N        Number of training iterations (default: 10)
  --games N            Self-play games per iteration (default: 100)
  --workers N          Number of parallel workers (default: CPU count)
  --evaluate           Enable model evaluation
  --eval-frequency N   Evaluate every N iterations (default: 5)
  --resume PATH        Resume from checkpoint
  --device DEVICE      Device to use: cpu or cuda (default: cuda)
```

### Examples

**Basic training** (10 iterations, 50 games each):
```bash
python pipeline.py --iterations 10 --games 50
```

**Training with evaluation** (compare models every 5 iterations):
```bash
python pipeline.py --iterations 20 --games 100 --evaluate --eval-frequency 5
```

**Resume from checkpoint**:
```bash
python pipeline.py --iterations 10 --resume checkpoints/checkpoint_iter_0010.pth
```

**CPU-only training** with 4 workers:
```bash
python pipeline.py --iterations 5 --games 20 --workers 4 --device cpu
```

## Configuration

Edit `config.py` to adjust hyperparameters:

### Neural Network
```python
NUM_RESIDUAL_BLOCKS = 19    # Number of residual blocks
NUM_FILTERS = 256           # Convolutional filters
```

### MCTS
```python
NUM_MCTS_SIMS = 800         # Simulations per move
CPUCT = 1.0                 # Exploration constant
DIRICHLET_ALPHA = 0.3       # Dirichlet noise alpha
```

### Training
```python
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_SELF_PLAY_GAMES = 100   # Games per iteration
```

### Evaluation
```python
EVAL_GAMES = 40             # Games for evaluation
WIN_THRESHOLD = 0.55        # Win rate to replace model
```

## Training Pipeline Details

### Full Iteration Cycle

1. **Self-Play Generation**
   - Neural network plays against itself using MCTS
   - Generates (state, policy, value) training examples
   - Explores with Dirichlet noise at root node
   - Temperature-based move selection

2. **Neural Network Training**
   - Trains on generated examples
   - Mixed loss: policy (KL divergence) + value (MSE)
   - Adam optimizer with weight decay
   - Batch normalization throughout

3. **Model Evaluation** (Optional)
   - Pits new model against previous best
   - Head-to-head games with alternating colors
   - Keeps model with higher win rate
   - Configurable win threshold

### Directory Structure After Training

```
chesshacks/
├── checkpoints/           # Model checkpoints
│   ├── checkpoint_iter_0000.pth
│   ├── checkpoint_iter_0001.pth
│   └── ...
├── training_data/         # Self-play game data
│   ├── iteration_0000.npy
│   ├── iteration_0001.npy
│   └── ...
└── logs/                  # Training history
    └── training_history.json
```

## Testing Individual Components

### Test Neural Network
```bash
python alpha_net.py
```

### Test MCTS
```bash
python MCTS_chess.py
```

### Test Training
```bash
python train.py
```

### Test Evaluator
```bash
python evaluator.py
```

### Test Multiprocessing
```bash
python train_multiprocessing.py
```

## Performance Notes

- **GPU Training**: Significantly faster with CUDA-enabled PyTorch
- **Self-Play**: CPU-bound, benefits from multiprocessing
- **Memory**: Each game generates ~100-200 training examples
- **Time**: ~1-2 hours per iteration with 100 games (depends on hardware)

## Training Strategy

### Early Training (Iterations 1-10)
- Disable evaluation to accumulate diverse training data
- Higher temperature for more exploration
- Focus on generating large dataset

### Mid Training (Iterations 10-50)
- Enable evaluation every 5 iterations
- Gradually reduce temperature
- Start comparing model improvements

### Late Training (Iterations 50+)
- Frequent evaluation
- Lower temperature for stronger play
- Fine-tune on high-quality games

## Algorithm Overview

AlphaZero combines three key techniques:

1. **Self-Play**: The system learns by playing against itself
2. **MCTS**: Guided tree search using neural network predictions
3. **Deep Learning**: CNN with residual blocks learns patterns

The neural network provides:
- **Policy**: Move probabilities (which moves are promising)
- **Value**: Position evaluation (who is winning)

MCTS uses these predictions to efficiently explore the game tree and select strong moves during self-play.

## Citation

Based on the AlphaZero algorithm:
```
Silver, D., et al. (2017). Mastering Chess and Shogi by Self-Play with a 
General Reinforcement Learning Algorithm. arXiv:1712.01815
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Areas for improvement:
- Distributed training across multiple machines
- Advanced data augmentation (board symmetries)
- Opening book integration
- Endgame tablebase integration
- Tensorboard visualization
- Model compression for faster inference

## Troubleshooting

**Out of Memory**: Reduce `BATCH_SIZE` or `NUM_RESIDUAL_BLOCKS` in config.py

**Slow Training**: 
- Ensure PyTorch has CUDA support: `torch.cuda.is_available()`
- Increase `--workers` for parallel self-play
- Reduce `NUM_MCTS_SIMS` for faster (but weaker) play

**Checkpoint Not Found**: Check `checkpoints/` directory exists and path is correct

**Multiprocessing Errors**: Set proper start method in your script:
```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

## Support

For issues and questions, please open an issue on the project repository.