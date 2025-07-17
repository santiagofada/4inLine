from src.game import Connect4
from src.agents.QLearning import QLearningAgent
from src.agents.randomAgent import RandomAgent
from src.agents.advancedHeuristicAgent import AdvancedHeuristicAgent
from src.agents.QLearning import QLearningAgent
import tqdm
ITER = 1
SAVE_PATH = f"../../assets/q_table_{ITER}.pkl"


def train_q_agent(episodes=2_000_000):
    agent = QLearningAgent(
        player_id=1,
        alpha=0.2,
        gamma=0.99,           # greedy con recompensas mas cercanas
        epsilon=1.0,          # empieza explorando
        epsilon_decay=0.9995,
        epsilon_min=0.05,
        strategy="softmax"
    )

    opponent = AdvancedHeuristicAgent(player_id=2)


    for _ in tqdm.tqdm(range(episodes)):
        game = Connect4()

        while not game.winner and not game.is_draw():

            if game.current_player == agent.player_id:  # ← player-1’s turn
                state = agent.serialize(game.board, agent.player_id)
                action = agent.select_action(game)
                game.make_move(action)
                next_state = agent.serialize(game.board, agent.player_id)

                done = game.winner is not None or game.is_draw()  # game over
                if game.winner == agent.player_id:
                    reward = 1
                elif game.winner == opponent.player_id:
                    reward = -1
                elif game.is_draw():
                    reward = 0.5
                else:
                    reward = -0.01

                agent.update(state, action, reward, next_state, done)

            else:  # opponent’s turn
                action = opponent.select_action(game)
                game.make_move(action)

        # ---- episode finished ----
        agent.alpha = max(0.02, agent.alpha * 0.9999)
        # epsilon already decayed inside select_action once per move; no extra decay here to avoid over-shrinking.

    agent.save(SAVE_PATH)
    print(f"Entrenamiento terminado. Tabla Q guardada en {SAVE_PATH}")

    print("\nEvaluando desempeño tras entrenamiento...")
    test_episodes = 1000
    wins, draws, losses = 0, 0, 0

    for _ in range(test_episodes):
        game = Connect4()
        while not game.winner and not game.is_draw():
            current_agent = agent if game.current_player == agent.player_id else opponent
            action = current_agent.select_action(game)
            game.make_move(action)

        if game.winner == agent.player_id:
            wins += 1
        elif game.winner == opponent.player_id:
            losses += 1
        else:
            draws += 1

    print(f"\nResultados después de {test_episodes} partidas:")
    print(f"Ganadas:   {wins} ({wins / test_episodes:.1%})")
    print(f"Empatadas: {draws} ({draws / test_episodes:.1%})")
    print(f"Perdidas:  {losses} ({losses / test_episodes:.1%})")


if __name__ == "__main__":
    train_q_agent()
