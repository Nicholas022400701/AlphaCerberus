# Alpha-Cerberus: Neuro-Symbolic Gomoku AI

![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)

This is the source code repository for Alpha-Cerberus. Unlike AlphaZero which relies purely on computational power, this project adopts a **Neuro-Symbolic** architecture.

Core logic: Intuition (Neural Network) + Logic (VCF/VCT Solver).

The JIT-compiled solver addresses the "computational poverty" problem, while ResNet fitting on human game records solves the "cold start" problem.

The code structure is as follows.



## 1. Core Architecture



This is the brain of the AI. Modifying this affects the fundamentals.

- **`config.py`**
  - **Purpose**: Global hyperparameter configuration.
  - **Logic**: Contains GPU settings, neural network structure (ResNet layers), MCTS simulation count, VCF/VCT search depth, etc.
  - **Note**: If computational power is limited, reduce `TRAINING_BATCH_SIZE`.
- **`config4greedy.py`**
  - **Purpose**: Simplified configuration customized for playing against greedy algorithms.
  - **Logic**: Disables exponentially time-consuming VCT detection, still easily defeats greedy algorithms.
- **`model.py`**
  - **Purpose**: Neural network definition.
  - **Logic**: Standard Alpha-Cerberus dual-head architecture (Policy head + Value head). Backbone is ResNet.
  - **Input**: 15x15x4 Tensor (self, opponent, last move, color).
- **`mcts.py`**
  - **Purpose**: Monte Carlo Tree Search (MCTS) implementation.
  - **Logic**: Implements a "short-circuit MCTS" mechanism.
    1. **Level 1**: Hard-coded rules (immediate win/must defend).
    2. **Level 2**: Calls VCF/VCT solver from `game.py` for winning/defensive move detection.
    3. **Level 3**: Neural network intervenes to guide MCTS search.
  - **Advantage**: Focuses computational power where needed; winning positions don't need neural network guessing.
- **`game.py`**
  - **Purpose**: Game environment and rule engine.
  - **Logic**: **Critically important**. All functions use `numba.jit` for static compilation acceleration. Contains board state updates, move legality checks, and core **VCF (Victory by Continuous Fours)** and **VCT (Victory by Continuous Threats)** recursive solvers.
  - **Performance**: Over 60x faster than pure Python.
- **`heuristic.py`**
  - **Purpose**: Standalone heuristic evaluation module.
  - **Logic**: Implements `GreedyGomokuAgent` class with pattern-based scoring. Evaluates moves based on attack and defense potential using pattern matching (five-in-a-row, open-four, open-three, etc.).



## 2. Training Pipeline



Data is the fuel. Without high-quality data, the model is garbage.

- **`parse.py`**
  - **Purpose**: ETL tool.
  - **Usage**: `python parse.py <psq_folder> <output.npz>`
  - **Logic**: Parses Gomocup `.psq` game record files, performs data augmentation (rotation, flipping), converts to `(state, policy, value)` tuples and compresses for storage.
- **`merge.py`**
  - **Purpose**: Data merging.
  - **Logic**: Merges multiple `.npz` files into a single large Replay Buffer for convenient one-time memory loading.
- **`SLTrain.py`**
  - **Purpose**: Supervised Learning (SL) main program.
  - **Usage**: `python SLTrain.py dataset.npz`
  - **Logic**: Load data -> Mixed precision training (AMP) -> Minimize KL divergence and MSE. Includes TensorBoard logging.
- **`utils.py`**
  - **Purpose**: Utility functions.
  - **Logic**: Mainly implements data augmentation (D4 symmetry group transformations).
- **`checkNPZ.py`**
  - **Purpose**: Data inspector.
  - **Logic**: Reads only `.npz` header information without loading data into memory, for quick data quantity and dimension checks.



## 3. Deployment & Interface



Once the model is trained, it needs to be put to use.

- **`play_web.py`**
  - **Purpose**: Web game backend (Flask).
  - **Usage**: Run and visit `localhost:5000`.
  - **Logic**: Loads trained model (`sl_best.ckpt`), instantiates `AIPilot` class. Processes coordinates from frontend, calls MCTS to return AI moves. Logs game records.
- **`ABArena.py`**
  - **Purpose**: Batch arena (GPU accelerated).
  - **Logic**: Uses multiprocessing + shared GPU inference to let AI play against itself or Minimax. For rapid model win rate evaluation without watching the screen.
- **`test_greedy.py`**
  - **Purpose**: Benchmark testing.
  - **Logic**: Lets SL model play against traditional greedy algorithm. If it can't beat this, don't bother with RL, go check the data.
- **`benchmark_vcf.py`**
  - **Purpose**: Performance benchmark testing.
  - **Logic**: Compares speed difference between Python recursion and Numba JIT recursion. Data source for paper writing.



## 4. Web Frontend & Traditional Algorithms (Legacy/Web)



Located in the `gomoku_web/` directory.

- **`server.py`** & **`templates/index.html`** & **`static/gomoku.js`**:
  - Used for visual interaction. Frontend contains hardcoded logic for drawing the board and move animations.
- **`simulate.py`**:
  - **Purpose**: AI vs AI simulation tool.
  - **Logic**: Runs automated games between Level 1 (Greedy) and Level 2 (Minimax) algorithms. Alternates first-move advantage and outputs win rate statistics.
- **`ai/core.py`**:
  - **Purpose**: Core definitions and utilities for traditional AI.
  - **Logic**: Defines board constants (EMPTY, BLACK, WHITE), board size (15x15), winning condition (5-in-a-row). Contains `check_winner_board()`, `generate_moves()`, and `find_immediate_win()` functions.
- **`ai/init.py`**:
  - **Purpose**: Package initialization file (empty).
- **`ai/level1_greedy.py`**:
  - **Purpose**: Level 1 Greedy AI (basic version).
  - **Logic**: Pure traditional algorithm (no neural network). Evaluates moves based on heuristic scoring considering both offensive and defensive patterns. Used as baseline comparison or for running without GPU.
- **`ai/level1_greedy_new.py`**:
  - **Purpose**: Level 1 Greedy AI (improved version).
  - **Logic**: Enhanced greedy algorithm with center position bonus (Tengen priority). Provides stronger opening play while maintaining the same pattern-based evaluation.
- **`ai/level2_minimax_alphabeta.py`**:
  - **Purpose**: Level 2 Minimax AI with Alpha-Beta pruning.
  - **Logic**: Traditional minimax search with alpha-beta optimization. Stronger than greedy but slower.
- **`ai/registry.py`**:
  - **Purpose**: AI level registry.
  - **Logic**: Maps AI level numbers to their corresponding move functions. Provides extensibility for adding future AI levels.

------

## Requirements

### Python Version
- **Python 3.12.7** (Required)

### Dependencies
Install the following libraries before running the project:

```bash
pip install torch numpy numba flask tqdm tensorboard
```

**Core Libraries:**
- `torch` - PyTorch deep learning framework for neural network training and inference
- `numpy` - Numerical computing library for array operations
- `numba` - JIT compiler for accelerating Python/NumPy code (VCF/VCT solver optimization)
- `flask` - Web framework for the game interface
- `tqdm` - Progress bar for training iterations
- `tensorboard` - TensorBoard logging for training visualization (via PyTorch)

**Note**: PyTorch installation may require CUDA-specific versions for GPU support. Visit [pytorch.org](https://pytorch.org/) for installation instructions based on your system.

------

**Usage Guide:**

1. **Environment**: Install Python 3.12.7 and all required dependencies listed above. Don't run training without a GPU.
2. **Data**: Process game records with `parse.py`, merge multiple `.npz` files with `merge.py`.
3. **Training**: Run `SLTrain.py` to get `sl_best.ckpt`.
4. **Run**: Execute `play_web.py` to start playing.

**File documentation complete. Read the code above for detailed logic.**
