from src.game import Connect4
import random
from src.agents.QLearning import QLearningAgent
import tqdm
ITER = 5

def train_self_play(episodes=1_000_000):
    agent1 = QLearningAgent(player_id=1,
                            alpha=0.4,
                            gamma=0.9,  # greedy con recompensas mas cercanas
                            epsilon=1.0,  # empieza explorando
                            epsilon_decay=0.9995,
                            epsilon_min=0.05
                            )
    agent2 = QLearningAgent(player_id=2,
                            alpha=0.4,
                            gamma=0.9,
                            epsilon=1.0,
                            epsilon_decay=0.9995,
                            epsilon_min=0.05
                            )

    #agent1.load(f"../assets/q_table_{ITER}.pkl")
    #agent2.load(f"../assets/q_table_{ITER-1}.pkl")

    for _ in tqdm.tqdm(range(episodes)):
        game = Connect4()
        state1 = agent1.serialize(game.board,player_id=1)
        state2 = agent2.serialize(game.board,player_id=2)

        starting_player = 1 if random.random() < 0.5 else 2
        game.current_player = starting_player


        while not game.winner and not game.is_draw():
            if game.current_player == 1:
                action = agent1.select_action(game)
                game.make_move(action)
                next_state1 = agent1.serialize(game.board,player_id=1)

                reward = (
                    1 if game.winner == 1 else
                    -1 if game.winner == 2 else
                    0.5 if game.is_draw() else
                    -0.01
                )
                done = game.winner or game.is_draw()
                agent1.update(state1, action, reward, next_state1, done)
                state1 = next_state1

            else:
                action = agent2.select_action(game)
                game.make_move(action)
                next_state2 = agent2.serialize(game.board, player_id=2)

                reward = (
                    1 if game.winner == 2 else
                    -1 if game.winner == 1 else
                    0.5 if game.is_draw() else
                    -0.01
                )
                done = game.winner or game.is_draw()
                agent2.update(state2, action, reward, next_state2, done)
                state2 = next_state2

    agent1.save(f"../assets/q_agent_dual_{ITER}.pkl")
    agent2.save(f"../assets/q_agent_dual_{ITER-1}.pkl")
    print("Self-play terminado y agentes guardados.")


    print("\nEvaluando desempeño: Agent1 vs Agent2...\n")
    test_episodes = 1000
    wins_1, wins_2, draws = 0, 0, 0

    for _ in range(test_episodes):
        game = Connect4()
        starting_player = 1 if random.random() < 0.5 else 2
        game.current_player = starting_player

        while not game.winner and not game.is_draw():
            current_agent = agent1 if game.current_player == 1 else agent2
            action = current_agent.select_action(game)
            game.make_move(action)

        if game.winner == 1:
            wins_1 += 1
        elif game.winner == 2:
            wins_2 += 1
        else:
            draws += 1

    total = wins_1 + wins_2 + draws
    print(f"Agent1 (X): {wins_1} ganadas ({wins_1 / total:.1%})")
    print(f"Agent2 (O): {wins_2} ganadas ({wins_2 / total:.1%})")
    print(f"Empates:     {draws} ({draws / total:.1%})")

if __name__ == "__main__":
    train_self_play(episodes=5_000_000)