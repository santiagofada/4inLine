import random
from src.agents.base import Agent

class RandomAgent(Agent):
    def select_action(self, game):
        return random.choice(game.available_actions())