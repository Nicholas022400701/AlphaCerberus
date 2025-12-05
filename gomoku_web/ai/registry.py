from ai.level1_greedy import greedy_move
from ai.level2_minimax_alphabeta import alphabeta_move

# For future levels (Alpha-Cerberus/Level4), simply register them here:
# from ai.alpha_cerberus import alpha_cerberus_move
# AI_LEVELS[3] = alpha_cerberus_move


AI_LEVELS = {
    1: greedy_move,
    2: alphabeta_move,
    # 3: alpha_cerberus_move,  # Reserved for future Alpha-Cerberus implementation
}
