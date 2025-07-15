import random
import pickle
import os
from src.agents.base import Agent

class QLearningAgent(Agent):
    def __init__(self, player_id, alpha=0.1, gamma=0.95, epsilon=0.2):
        super().__init__(player_id)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def serialize(self, board):
        # Convierte el tablero en una cadena única (para usar como clave)
        return ''.join(str(cell) for row in board for cell in row)

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def select_action(self, game):
        state = self.serialize(game.board)
        actions = game.available_actions()

        if random.random() < self.epsilon:
            return random.choice(actions)

        q_values = [(self.get_q(state, a), a) for a in actions]
        max_q = max(q_values, key=lambda x: x[0])[0]
        best_actions = [a for q, a in q_values if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        old_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            next_qs = [self.get_q(next_state, a) for a in range(7)]
            target = reward + self.gamma * max(next_qs)

        new_q = old_q + self.alpha * (target - old_q)
        self.q_table[(state, action)] = new_q

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)
