from src.game import Connect4
from src.agents.humanAgent import HumanAgent
from src.agents.randomAgent import RandomAgent
from src.agents.basicHeuristicAgent import HeuristicAgent
from src.agents.QLearning import QLearningAgent

import sys
import time

SYMBOLS = {0: ' ', 1: 'X', 2: 'O'}

def print_board(board):
    for row in board:
        line = '|' + '|'.join(SYMBOLS[cell] for cell in row) + '|'
        print(line, flush=True)
    print('-' * (2 * len(board[0]) + 1), flush=True)
    print(' ' + ' '.join(str(i+1) for i in range(len(board[0]))), flush=True)

def play_game():
    game = Connect4()
    player1 = HumanAgent(player_id=1)
    #player1 = QLearningAgent(player_id=2)
    #player1.load("../assets/q_table.pkl")

    #player2 = HeuristicAgent(player_id=2)
    player2 = QLearningAgent(player_id=2)
    player2.load("../assets/q_agent_dual_5.pkl")

    print("Bienvenido a 4 en línea! Tú eres el Jugador 1 (X)", flush=True)
    print_board(game.board)

    while not game.winner and not game.is_draw():
        current_agent = player1 if game.current_player == 1 else player2
        action = current_agent.select_action(game)
        game.make_move(action)
        print_board(game.board)
        time.sleep(0.3)

    if game.winner:
        print(f"Ganó el Jugador {SYMBOLS[game.winner]}!", flush=True)
    else:
        print("Empate!", flush=True)

if __name__ == "__main__":
    play_game()
