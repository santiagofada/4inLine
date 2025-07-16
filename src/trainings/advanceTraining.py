import os
import random
import csv
import tqdm
from src.game import Connect4
from src.agents.QLearning import QLearningAgent
from src.agents.advancedHeuristicAgent import AdvancedHeuristicAgent
from src.agents.randomAgent import RandomAgent

MARK = 0

SAVE_BASE = f"../../assets/q_agent_mix_{MARK}"
PREVIOUS_AGENT_PATH = "../../assets/old/q_agent_dual_5.pkl"
EPISODES = 2_000_000
EVAL_INTERVAL = 100_000

def train_mixed():
    agent = QLearningAgent(
        player_id=1,
        alpha=0.4,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.05,
        strategy="softmax"
    )

    if os.path.exists(PREVIOUS_AGENT_PATH):
        print("Cargando agente previo...")
        opponent_snapshot = QLearningAgent(player_id=2)
        opponent_snapshot.load(PREVIOUS_AGENT_PATH)
    else:
        opponent_snapshot = None

    opponents = [
        lambda: AdvancedHeuristicAgent(player_id=2),
        lambda: RandomAgent(player_id=2),
        lambda: QLearningAgent(player_id=2).load_and_return(PREVIOUS_AGENT_PATH) if opponent_snapshot else RandomAgent(player_id=2)
    ]

    for episode in tqdm.tqdm(range(1, EPISODES + 1)):
        game = Connect4()
        game.current_player = 1 if random.random() < 0.5 else 2

        # Alternamos oponente cada 10k episodios
        opponent_index = (episode // EVAL_INTERVAL) % len(opponents)
        opponent = opponents[opponent_index]()

        # Alternar roles entre jugador 1 y 2
        if random.random() < 0.5:
            agent.player_id = 1
            opponent.player_id = 2
        else:
            agent.player_id = 2
            opponent.player_id = 1

        state = agent.serialize(game.board, agent.player_id)

        while not game.winner and not game.is_draw():
            current = agent if game.current_player == agent.player_id else opponent
            action = current.select_action(game)
            game.make_move(action)

            if isinstance(current, QLearningAgent):
                next_state = agent.serialize(game.board, agent.player_id)
                reward = (
                    1 if game.winner == agent.player_id else
                    -1 if game.winner == 3 - agent.player_id else
                    0.5 if game.is_draw() else
                    -0.01
                )
                done = game.winner or game.is_draw()
                agent.update(state, action, reward, next_state, done)
                state = next_state

        # === Evaluación ===
        if episode % EVAL_INTERVAL == 0:
            print(f"\nEvaluando episodio {episode}...")
            evaluate_agent(agent, episode=episode)

        if episode > EPISODES * 0.8:
            agent.alpha = max(0.05, agent.alpha * 0.999)

    # Guardar versión final y snapshot
    agent.save(f"{SAVE_BASE}_final.pkl")
    agent.save(PREVIOUS_AGENT_PATH)
    print("\nEntrenamiento completo.")

def evaluate_agent(agent, test_episodes=1000,episode=None):
    heur = AdvancedHeuristicAgent(player_id=2)
    rand = RandomAgent(player_id=2)
    results = {"heuristic": [0, 0, 0], "random": [0, 0, 0]}  # [wins, losses, draws]

    for name, opponent in [("heuristic", heur), ("random", rand)]:
        for _ in range(test_episodes):
            game = Connect4()
            game.current_player = 1 if random.random() < 0.5 else 2
            while not game.winner and not game.is_draw():
                current = agent if game.current_player == agent.player_id else opponent
                action = current.select_action(game)
                game.make_move(action)

            if game.winner == agent.player_id:
                results[name][0] += 1
            elif game.winner == 3 - agent.player_id:
                results[name][1] += 1
            else:
                results[name][2] += 1

        w, l, d = results[name]
        total = w + l + d
        print(f"\nResultados contra {name.upper()}:")
        print(f"  Ganadas: {w} ({w/total:.1%})")
        print(f"  Perdidas: {l} ({l/total:.1%})")
        print(f"  Empates: {d} ({d/total:.1%})")

        if episode is not None:
            with open("train_eval_log.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([episode, name, w, l, d])

if __name__ == "__main__":
    train_mixed()