import random
import numpy as np
import pickle
import os
from src.agents.base import Agent

def softmax(x, tau=1.0):
    x = np.array(x)
    x = x - np.max(x)  # para estabilidad numérica
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()

class QLearningAgent(Agent):
    def __init__(self, player_id, alpha=0.1, gamma=0.95, epsilon=0.2,
                 epsilon_decay=1.0, epsilon_min=0.01, strategy="softmax"):
        super().__init__(player_id)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon        # ε-greedy
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.strategy = strategy

    def serialize(self, board,player_id):
        # Convierte el tablero en una cadena única (para usar como clave)
        return ''.join(str(cell) for row in board for cell in row) + f'-{player_id}'

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)



    def select_action(self, game):
        self.last_board = [row[:] for row in game.board]

        state = self.serialize(game.board, self.player_id)
        actions = game.available_actions()

        # ε-greedy
        if self.strategy == "greedy":
            if random.random() < self.epsilon:
                action = random.choice(actions)
            else:
                q_values = [(self.get_q(state, a), a) for a in actions]
                max_q = max(q_values, key=lambda x: x[0])[0]
                best_actions = [a for q, a in q_values if q == max_q]
                action = random.choice(best_actions)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        elif self.strategy == "softmax":
            q_values = [self.get_q(state, a) for a in actions]
            probs = softmax(q_values, tau=self.epsilon + 0.1)  # usamos epsilon como temperatura
            action = np.random.choice(actions, p=probs)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        else:
            raise ValueError(f"Estrategia de exploración desconocida: {self.strategy}")

        return action

    def mirror_board(self, board):
        return [list(reversed(row)) for row in board]

    def update(self, state, action, reward, next_state, done):
        old_q = self.get_q(state, action)
        next_qs = [self.get_q(next_state, a) for a in range(7)] if not done else [0]
        target = reward + self.gamma * max(next_qs)
        new_q = old_q + self.alpha * (target - old_q)
        self.q_table[(state, action)] = new_q

        # === SIMETRÍA ===
        mirrored_board = self.mirror_board(self.last_board)
        mirrored_state = self.serialize(mirrored_board, self.player_id)
        mirrored_action = 6 - action
        old_q_m = self.get_q(mirrored_state, mirrored_action)
        new_q_m = old_q_m + self.alpha * (target - old_q_m)
        self.q_table[(mirrored_state, mirrored_action)] = new_q_m

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

    def load_and_return(self, path):
        self.load(path)
        return self