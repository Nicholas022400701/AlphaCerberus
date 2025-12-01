from ai.level1_greedy import greedy_move
from ai.level2_minimax_alphabeta import alphabeta_move

# For future levels (Level3/Level4), simply register them here:
# from ai.level3_xxx import level3_move
# AI_LEVELS[3] = level3_move


AI_LEVELS = {
    1: greedy_move,
    2: alphabeta_move,
    # 3: level3_move,  # Resrved for future Level 3 implementation
}
