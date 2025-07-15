from src.game import Connect4
from src.agents.q_learning import QLearningAgent
from src.agents.randomAgent import RandomAgent
from src.agents.heuristicAgent import HeuristicAgent

import tqdm

SAVE_PATH = "../assets/q_table.pkl"


def train_q_agent(episodes=10_000_000):
    agent = QLearningAgent(
        player_id=1,
        alpha=0.4,
        gamma=0.95,           # greedy con recompensas mas cercanas
        epsilon=1.0,          # empieza explorando
        epsilon_decay=0.9992,
        epsilon_min=0.05
    )
    opponent = HeuristicAgent (player_id=2)

    for _ in tqdm.tqdm(range(episodes)):
        game = Connect4()
        state = agent.serialize(game.board)

        while not game.winner and not game.is_draw():
            current_agent = agent if game.current_player == 1 else opponent
            action = current_agent.select_action(game)
            game.make_move(action)
            next_state = agent.serialize(game.board)

            if isinstance(current_agent, QLearningAgent):
                if game.winner == agent.player_id:
                    reward = 1
                elif game.winner == opponent.player_id:
                    reward = -1
                elif game.is_draw():
                    reward = 0.5
                else:
                    reward = -0.01

                done = game.winner is not None or game.is_draw()
                agent.update(state, action, reward, next_state, done)
                state = next_state

    agent.save(SAVE_PATH)
    print(f"Entrenamiento terminado. Tabla Q guardada en {SAVE_PATH}")


if __name__ == "__main__":
    train_q_agent()