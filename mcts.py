# mcts.py
import math
import numpy as np
import torch
import logging
import torch.multiprocessing as mp
from config import CONF
from game import GomokuEnv

class Node:
    __slots__ = ['P', 'N', 'W', 'Q', 'children', 'parent']

    def __init__(self, parent=None, P=0.0):
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = P

    def get_puct_score(self, c_puct, N_parent_sqrt):
        # Q value from opponent's perspective (canonical form)
        Q_value = self.Q if self.N > 0 else CONF.MCTS_FPU_VALUE
        return -Q_value + c_puct * self.P * N_parent_sqrt / (1 + self.N)

    def select(self, c_puct, legal_mask):
        N_parent_sqrt = math.sqrt(self.N + 1e-8)
        best_score = -np.inf
        best_action = -1
        
        for action, node in self.children.items():
            if not legal_mask[action]: continue
            
            score = node.get_puct_score(c_puct, N_parent_sqrt)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def expand(self, policy, legal_mask):
        for action, prob in enumerate(policy):
            if legal_mask[action] and action not in self.children:
                self.children[action] = Node(parent=self, P=prob)

    def backup(self, value):
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value
            node = node.parent

class MCTS:
    def __init__(self, inference_pipeline):
        self.pipeline = inference_pipeline 
        self.c_puct = CONF.C_PUCT
        self.sim_env = GomokuEnv(CONF.BOARD_SIZE, CONF.N_IN_ROW, CONF.INPUT_CHANNELS)
        self.root = Node() 
        
        # Pre-compile simulation schedule
        self.sim_schedule_moves = sorted([move for move, sims in CONF.MCTS_SIMULATION_SCHEDULE], reverse=True)
        self.sim_schedule_map = {move: sims for move, sims in CONF.MCTS_SIMULATION_SCHEDULE}

    def _get_dynamic_sims(self, move_count):
        for move_threshold in self.sim_schedule_moves:
            if move_count >= move_threshold:
                return self.sim_schedule_map[move_threshold]
        return self.sim_schedule_map[self.sim_schedule_moves[-1]]

    def reset_tree(self):
        self.root = Node()

    def reuse_tree(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = Node()

    def run_search(self, env: GomokuEnv, stop_signal=None, verbose=False, progress_callback=None): 
        if stop_signal is None: stop_signal = mp.Event() 
        
        decision_source = "MCTS"

        # =====================================================================
        # LAYER 1: ABSOLUTE URGENCY (Rules)
        # =====================================================================
        
        # 1.1 Check Instant Win / Forced Block
        urgent_move, is_win = env.get_urgent_move()
        
        if urgent_move != -1:
            if is_win:
                if verbose: logging.info(f"   >>> ⚡ Auto-Win found!")
                return self._one_hot(urgent_move), "Urgent Win"
            else:
                # Urgent Block found (Defense).
                # We need to check: Is this a "Block 5" (Instant Death) or "Block 4/Double" (Lag)?
                # If it's Instant Death, we MUST block.
                # If it's Lag, we might have time for a Counter-VCF.
                
                r, c = divmod(urgent_move, CONF.BOARD_SIZE)
                opponent = -env.current_player
                
                # [FIXED] Use the instance method _check_exact_win
                # If opponent plays here, do they win (make 5)?
                opp_wins_immediately = env._check_exact_win(r, c, opponent)
                
                if not opp_wins_immediately and CONF.ENABLE_VCF:
                     # Opponent is NOT winning immediately (it's just an Open 4 or Double 3).
                     # We have a chance to counter-attack!
                     my_vcf = env.solve_vcf(CONF.VCF_DEPTH)
                     if my_vcf != -1:
                         if verbose: logging.info("   >>> ⚔️ Counter-Attack VCF Found!")
                         return self._one_hot(my_vcf), "Counter VCF"

                if verbose: logging.info(f"   >>> 🛡️ Urgent Block!")
                return self._one_hot(urgent_move), "Urgent Block"

        # =====================================================================
        # LAYER 2: DEEP DEFENSE (Opponent VCF)
        # =====================================================================
        
        defensive_move = self._check_defensive_vcf(env)
        if defensive_move != -1:
             if verbose: logging.info(f"   >>> 🛡️ Deep Defense (VCF Block)!")
             return self._one_hot(defensive_move), "Defensive VCF"
        if CONF.ENABLE_VCT:
            defensive_vct = self._check_defensive_vct(env)
            if defensive_vct != -1:
                if verbose: logging.info(f"   >>> 🛡️ Deep Defense (VCT Block)!")
                return self._one_hot(defensive_vct), "Defensive VCT"
        # =====================================================================
        # LAYER 3: OFFENSIVE SOLVERS
        # =====================================================================

        if CONF.ENABLE_VCF:
            vcf = env.solve_vcf(CONF.VCF_DEPTH)
            if vcf != -1:
                if verbose: logging.info("   >>> ⚔️ VCF Found!")
                return self._one_hot(vcf), "VCF Attack"
        
        if CONF.ENABLE_VCT:
            vct = env.solve_vct(CONF.VCT_DEPTH, CONF.VCT_WIDTH)
            if vct != -1:
                if verbose: logging.info("   >>> 🗡️ VCT Found!")
                return self._one_hot(vct), "VCT Attack"

        # =====================================================================
        # LAYER 4: NEURAL MCTS (Intuition)
        # =====================================================================
        
        if self.root.N == 0:
            if not self._evaluate(self.root, env, stop_signal=stop_signal): return None, "Error"
        
        # Dynamic Compute Budget
        dynamic_sims = self._get_dynamic_sims(env.move_count)
        
        if CONF.ENABLE_DYNAMIC_COMPUTE and self.root.N > 0:
            # If Value is ambiguous (near 0), think deeper
            if abs(self.root.Q) < CONF.DYNAMIC_CONFIDENCE_THRESHOLD:
                dynamic_sims = int(dynamic_sims * CONF.DYNAMIC_BOOST_MULTIPLIER)
                decision_source = "MCTS (Boosted)"

        current = self.root.N
        needed = max(0, dynamic_sims - current)
        
        for i in range(needed):
            if stop_signal.is_set(): return None, "Interrupted"
            
            if progress_callback: progress_callback(i, needed)
            
            if verbose and (i+1) % 100 == 0:
                logging.info(f"   >>> MCTS: {i+1}/{needed}")
                
            if not self._run_simulation(self.root, env, stop_signal): 
                break 
        
        return self._calculate_policy_target(self.root, env.move_count), decision_source

    def _check_defensive_vcf(self, env):
        """ 
        Simulate opponent's turn to check if they have a VCF win.
        If yes, return the move to block it.
        """
        self.sim_env.copy_from(env)
        self.sim_env.current_player = -env.current_player
        return self.sim_env.solve_vcf(CONF.VCF_DEPTH)
    
    def _check_defensive_vct(self, env):
        """
        Simulates the opponent's turn and checks if the opponent has a VCT kill move.
        If so, the blocking point must be returned immediately.
        """
        self.sim_env.copy_from(env)
        self.sim_env.current_player = -env.current_player
        return self.sim_env.solve_vct(CONF.VCT_DEPTH, CONF.VCT_WIDTH)

    def _one_hot(self, action):
        p = np.zeros(CONF.ACTION_SIZE, dtype=np.float32)
        p[action] = 1.0
        return p

    def _run_simulation(self, root, root_env, stop_signal): 
        node = root
        self.sim_env.copy_from(root_env)

        while node.children:
            if stop_signal.is_set(): return False
            
            mask = self.sim_env.get_legal_moves_mask_flat()
            action = node.select(self.c_puct, mask)
            
            if action == -1: break
            
            node = node.children[action]
            self.sim_env.step(action)
            if self.sim_env.done: break
        
        value = self._evaluate(node, self.sim_env, stop_signal=stop_signal)
        if value is None: return False
        
        node.backup(value)
        return True

    def _evaluate(self, node, env, stop_signal=None): 
        if env.done:
            if env.winner == 0: v = 0.0
            else: v = 1.0 if env.winner == env.current_player else -1.0
            return v

        # [Solver in Rollout]
        urgent, is_win = env.get_urgent_move()
        mask = env.get_legal_moves_mask_flat()
        
        if urgent != -1:
            if is_win:
                policy = np.zeros(CONF.ACTION_SIZE, dtype=np.float32); policy[urgent]=1.0
                node.expand(policy, mask)
                return 1.0
            else:
                # Forced block, check value
                result = self.pipeline.infer(env.get_state(), stop_signal)
                if result is None: return None
                _, value = result
                
                policy = np.zeros(CONF.ACTION_SIZE, dtype=np.float32); policy[urgent]=1.0
                node.expand(policy, mask)
                return value

        # Standard NN Inference
        result = self.pipeline.infer(env.get_state(), stop_signal) 
        if result is None: return None
        policy, value = result

        policy *= mask
        p_sum = np.sum(policy)
        if p_sum > 0: policy /= p_sum
        else: policy = mask.astype(np.float32) / mask.sum()
        
        node.expand(policy, mask)
        return value

    def _calculate_policy_target(self, root, move_count):
        visits = np.zeros(CONF.ACTION_SIZE, dtype=np.float32) 
        for a, n in root.children.items(): visits[a] = n.N
        
        temp = 1.0 if move_count < CONF.TEMP_THRESHOLD else 0.01
        
        if temp < 0.05:
            target = np.zeros_like(visits)
            if visits.sum() > 0:
                target[np.argmax(visits)] = 1.0
        else:
            try:
                visits = visits ** (1/temp)
                v_sum = visits.sum()
                if v_sum > 0: target = visits / v_sum
                else: target = visits
            except:
                target = visits / visits.sum()

        return target
