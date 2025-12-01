# utils.py
import numpy as np
import torch
from config import CONF

def augment_data(state_np, policy_np):
    augmented = []
    policy_2d = policy_np.reshape(CONF.BOARD_SIZE, CONF.BOARD_SIZE)
    if not state_np.flags['C_CONTIGUOUS']: state_np = np.ascontiguousarray(state_np)

    for i in range(4):
        s_rot = np.rot90(state_np, i, axes=(1, 2))
        p_rot = np.rot90(policy_2d, i)
        augmented.append((s_rot.copy(), p_rot.flatten().copy()))
        
        s_flip = np.flip(s_rot, axis=2) 
        p_flip = np.flip(p_rot, axis=1) 
        augmented.append((s_flip.copy(), p_flip.flatten().copy()))
    return augmented

def prepare_samples_for_queue(game_history, winner):
    batch = []
    for s, p, player in game_history:
        v = 0.0
        if winner != 0: v = 1.0 if player == winner else -1.0
        aug = augment_data(s, p)
        for as_, ap_ in aug:
            batch.append((torch.from_numpy(as_), torch.from_numpy(ap_).float(), torch.tensor([v], dtype=torch.float32)))
    return batch