# config.py
import torch
import logging
import multiprocessing as mp
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class Config:
    # --- Hardware ---
    FORCE_CUDA = True 
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        logging.info(f"Device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
        logging.warning("Using CPU")

    # --- Paths ---
    SL_MODEL_PATH = "sl_best.ckpt"           
    # Backwards compatibility: some scripts expect MODEL_PATH
    MODEL_PATH = SL_MODEL_PATH

    # --- Game Environment ---
    BOARD_SIZE = 15
    N_IN_ROW = 5
    INPUT_CHANNELS = 4 
    ACTION_SIZE = BOARD_SIZE * BOARD_SIZE

    # --- MCTS (Inference) ---
    MCTS_SIMULATION_SCHEDULE = [
        (0, 800),    
    ]
    C_PUCT = 2.0           
    MCTS_FPU_VALUE = 0.1   

    # [Fix] Add back compatibility parameter
    # Set to -1 to disable randomness completely (Argmax), only play the strongest move
    TEMP_THRESHOLD = -1   
    
    # [Fix] Add back noise parameters (to prevent errors, setting to 0 has no effect)
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25

    # --- Solver ---
    ENABLE_VCF = True
    VCF_DEPTH = 16       
    ENABLE_VCT = True
    VCT_DEPTH = 6        
    VCT_WIDTH = 4        

    # --- Dynamic Compute ---
    ENABLE_DYNAMIC_COMPUTE = True
    DYNAMIC_CONFIDENCE_THRESHOLD = 0.3
    DYNAMIC_BOOST_MULTIPLIER = 2.0

    # --- Network ---
    NUM_RES_BLOCKS = 6
    NUM_FILTERS = 128

    # --- SL Training ---
    TRAINING_BATCH_SIZE = 512    
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4           

    # --- System ---
    CAN_COMPILE = False
    if sys.platform != "win32":
        try:
            if (sys.version_info >= (3, 9) and torch.__version__ >= "2.0"): CAN_COMPILE = True
        except: pass

CONF = Config()