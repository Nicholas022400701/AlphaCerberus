# parse.py
import sys
import re
import numpy as np
import logging
import os
import glob
import torch

from config import CONF
from game import GomokuEnv, P1, P2
from utils import prepare_samples_for_queue

# Global settings for parsing .psq files
# Typically PSQ files are 1-indexed (1..15)
INDEX_OFFSET = 1 
PARSE_FORMAT = "row_col" # or "col_row" depending on your data source

def process_psq_file(filepath):
    """
    Parse a single .psq file, simulate game, and extract (s, p, v) samples.
    """
    logging.info(f"Processing: {filepath}")
    
    env = GomokuEnv(CONF.BOARD_SIZE, CONF.N_IN_ROW, CONF.INPUT_CHANNELS)
    game_history = []
    
    move_regex = re.compile(r'^(\d+),(\d+),(\d+)$')
    moves_parsed = 0
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                match = move_regex.match(line)
                
                if not match: continue

                parts = line.split(',')
                
                if PARSE_FORMAT == "row_col":
                    row_idx = int(parts[0]) - INDEX_OFFSET
                    col_idx = int(parts[1]) - INDEX_OFFSET
                else: 
                    col_idx = int(parts[0]) - INDEX_OFFSET
                    row_idx = int(parts[1]) - INDEX_OFFSET
                
                action = row_idx * CONF.BOARD_SIZE + col_idx
                
                if not (0 <= row_idx < CONF.BOARD_SIZE and 0 <= col_idx < CONF.BOARD_SIZE):
                    continue
                
                if env.board[row_idx, col_idx] != 0:
                    continue

                # Store (State, Policy Target, Player)
                state_np = env.get_state()
                player = env.current_player
                
                policy_target_np = np.zeros(CONF.ACTION_SIZE, dtype=np.float32)
                policy_target_np[action] = 1.0
                
                game_history.append((state_np, policy_target_np, player))
                
                env.step(action)
                moves_parsed += 1
                
                if env.done: break

    except Exception as e:
        logging.error(f"Error parsing {filepath}: {e}")
        return []

    if moves_parsed == 0:
        return []

    # Winner logic
    winner = env.winner
    # If game ended abruptly in file without winner mark, infer from last move?
    # Usually PSQ files contain full games. If no winner marked by engine logic, assume draw or use file metadata.
    # Here we use engine logic.
    
    # Augment and format
    augmented_samples = prepare_samples_for_queue(game_history, winner)
    return augmented_samples

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse.py <input_path> [output_file.npz]")
        sys.exit(1)
        
    input_path = sys.argv[1].strip().replace('"', '')
    default_output = "dataset.npz"
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    if not output_path.endswith(".npz"): output_path += ".npz"

    # Find files
    file_list = []
    if os.path.isdir(input_path):
        search_pattern = os.path.join(input_path, "*.psq")
        file_list = glob.glob(search_pattern)
    elif os.path.isfile(input_path):
        file_list.append(input_path)
    else:
        print("Invalid path.")
        return

    if not file_list:
        print("No .psq files found.")
        return

    logging.info(f"Found {len(file_list)} files.")

    all_samples = []
    
    for psq_file in file_list:
        samples = process_psq_file(psq_file)
        if samples:
            all_samples.extend(samples)
    
    if not all_samples:
        print("No valid samples generated.")
        return

    logging.info(f"Total augmented samples: {len(all_samples)}")
    logging.info("Stacking tensors...")

    all_states = [s for s, p, v in all_samples]
    all_policies = [p for s, p, v in all_samples]
    all_values = [v for s, p, v in all_samples]
    
    del all_samples # Free memory

    stacked_states = torch.stack(all_states).numpy()
    stacked_policies = torch.stack(all_policies).numpy()
    stacked_values = torch.stack(all_values).numpy()
    
    logging.info(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        states=stacked_states,
        policies=stacked_policies,
        values=stacked_values
    )
    logging.info("Done.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    main()