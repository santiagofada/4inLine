import random
import tqdm
from src.game import Connect4
from src.agents.QLearning import QLearningAgent
from src.agents.randomAgent import RandomAgent
from src.agents.advancedHeuristicAgent import AdvancedHeuristicAgent

ITER = 2
SAVE_PATH = f"../../assets/q_table_{ITER}.pkl"
EPISODES  = 2_000_000
CURRICULUM_SWITCH = 200_000          # después de estos episodios pasa de Random → Heurístico
EVAL_INTERVAL     = 50_000           # evalúa cada tanto (opcional)

# --- helper: detectar si hay exactamente 3 en línea propia o rival ---
def made_three(game, player_id):
    """Devuelve True si player_id tiene 3 en línea sin cuarta ficha."""
    rows, cols = 6, 7
    board = game.board
    # horizontal & vertical & diagonales
    directions = [(0,1), (1,0), (1,1), (1,-1)]
    for r in range(rows):
        for c in range(cols):
            if board[r][c] != player_id:
                continue
            for dr, dc in directions:
                cnt = 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == player_id:
                    cnt += 1
                    rr += dr; cc += dc
                if cnt == 3:       # justo tres
                    # Comprueba que no haya una cuarta ficha contigua
                    if 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == player_id:
                        continue
                    # Comprueba que detrás tampoco haya cuarta
                    rr, cc = r - dr, c - dc
                    if 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == player_id:
                        continue
                    return True
    return False


def train_q_agent(episodes=EPISODES):
    agent = QLearningAgent(
        player_id=1,
        alpha=0.2,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,          # ← temperatura final más baja
        strategy="softmax"
    )

    for episode in tqdm.tqdm(range(1, episodes + 1)):
        # -------- curriculum de oponentes ----------
        if episode < CURRICULUM_SWITCH:
            opponent = RandomAgent(player_id=2)
        else:
            opponent = AdvancedHeuristicAgent(player_id=2)

        # ---- alterna colores ----------
        if random.random() < 0.5:
            agent.player_id, opponent.player_id = 1, 2
        else:
            agent.player_id, opponent.player_id = 2, 1

        game = Connect4()
        game.current_player = 1 if random.random() < 0.5 else 2   # quién mueve primero

        while not game.winner and not game.is_draw():
            if game.current_player == agent.player_id:
                state = agent.serialize(game.board, agent.player_id)
                action = agent.select_action(game)
                game.make_move(action)
                next_state = agent.serialize(game.board, agent.player_id)

                # ---------- reward shaping ----------
                reward = -0.01                                        # penalización base
                if made_three(game, agent.player_id):
                    reward += 0.2
                if made_three(game, opponent.player_id):
                    reward -= 0.2

                if game.winner == agent.player_id:
                    reward = 1
                elif game.winner == opponent.player_id:
                    reward = -1
                elif game.is_draw():
                    reward = 0.5

                done = game.winner is not None or game.is_draw()
                agent.update(state, action, reward, next_state, done)

            else:
                # turno del oponente
                action = opponent.select_action(game)
                game.make_move(action)

        # --- fin del episodio: baja alpha al final ---
        if episode > episodes * 0.8:
            agent.alpha = max(0.02, agent.alpha * 0.9999)

        # --- evaluación opcional ---
        if episode % EVAL_INTERVAL == 0:
            print(f"\nEp {episode}: α={agent.alpha:.3f}, ε={agent.epsilon:.3f}")

    agent.save(SAVE_PATH)
    print(f"\nEntrenamiento terminado. Modelo guardado en {SAVE_PATH}")


if __name__ == "__main__":
    train_q_agent()
