# config4greedy.py
import torch
import sys

class config4greedy:
    FORCE_CUDA = True 
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    SL_MODEL_PATH = "sl_best.ckpt"           


    BOARD_SIZE = 15
    N_IN_ROW = 5
    INPUT_CHANNELS = 4 
    ACTION_SIZE = BOARD_SIZE * BOARD_SIZE


    MCTS_SIMULATION_SCHEDULE = [
        (0, 800),    
    ]
    C_PUCT = 2.0           
    MCTS_FPU_VALUE = 0.1   

    
    TEMP_THRESHOLD = -1   
    
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25

    
    ENABLE_VCF = True
    VCF_DEPTH = 16       
    
    ENABLE_VCT = False   
    VCT_DEPTH = 6        
    VCT_WIDTH = 4        

    
    ENABLE_DYNAMIC_COMPUTE = True
    DYNAMIC_CONFIDENCE_THRESHOLD = 0.3
    DYNAMIC_BOOST_MULTIPLIER = 1.5 

    NUM_RES_BLOCKS = 6
    NUM_FILTERS = 128

    TRAINING_BATCH_SIZE = 512    
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4           

    CAN_COMPILE = False
    if sys.platform != "win32":
        try:
            if (sys.version_info >= (3, 9) and torch.__version__ >= "2.0"): CAN_COMPILE = True
        except: pass

CONF = config4greedy()