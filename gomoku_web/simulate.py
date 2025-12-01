import random
from copy import deepcopy

# import AI logic
from ai.core import EMPTY, BLACK, WHITE, N, check_winner_board
from ai.level1_greedy import greedy_move
from ai.level2_minimax_alphabeta import alphabeta_move


# play one game  first_player=1 means L1 plays Black first  first_player=2 means L2 plays Black first
def play_one_game(first_player=1):
    board = [[EMPTY] * N for _ in range(N)]

    # assign stone colors  first player uses Black second player uses White
    if first_player == 1:        # L1 moves first
        L1_stone = BLACK
        L2_stone = WHITE
        current = L1_stone
    else:                        # L2 moves first
        L2_stone = BLACK
        L1_stone = WHITE
        current = L2_stone

    # alternate turns until a winner appears
    while True:
        if current == L1_stone:
            x, y = greedy_move(board, L1_stone)
            board[x][y] = L1_stone
            if check_winner_board(board, x, y, L1_stone):
                return "L1"
            current = L2_stone

        else:
            x, y = alphabeta_move(board, L2_stone)
            board[x][y] = L2_stone
            if check_winner_board(board, x, y, L2_stone):
                return "L2"
            current = L1_stone


# simulate multiple games  first player alternates in each round
def simulate_games(rounds=20):
    L1_wins = 0
    L2_wins = 0

    for i in range(rounds):
        # alternating first move  L1 first in even index games  L2 first in odd index games
        first = 1 if i % 2 == 0 else 2

        result = play_one_game(first_player=first)

        if result == "L1":
            L1_wins += 1
        else:
            L2_wins += 1

        print(f"Game {i+1}/{rounds} → Winner: {result}   (first: {'L1' if first == 1 else 'L2'})")

    # final summary output
    print("\n===== FINAL RESULTS =====")
    print(f"Total games : {rounds}")
    print(f"Level1 wins : {L1_wins}")
    print(f"Level2 wins : {L2_wins}")
    print(f"Level1 win rate : {L1_wins / rounds * 100:.2f}%")
    print(f"Level2 win rate : {L2_wins / rounds * 100:.2f}%")


if __name__ == "__main__":
    simulate_games(rounds=50)
