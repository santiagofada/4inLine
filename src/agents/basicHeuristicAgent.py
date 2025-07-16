from src.agents.base import Agent
import random
import copy

class BasicHeuristicAgent(Agent):
    def select_action(self, game):
        opponent_id = 3 - self.player_id

        # 1. Ver si puedo ganar en la próxima jugada
        for action in game.available_actions():
            temp_game = copy.deepcopy(game)
            temp_game.current_player = self.player_id
            temp_game.make_move(action)
            if temp_game.winner == self.player_id:
                return action

        # 2. Bloquear al oponente si puede ganar
        for action in game.available_actions():
            temp_game = copy.deepcopy(game)
            temp_game.current_player = opponent_id
            temp_game.make_move(action)
            if temp_game.winner == opponent_id:
                return action

        # 3. Jugar al centro si está disponible
        center = len(game.board[0]) // 2
        if center in game.available_actions():
            return center

        # 4. Si no, elegir al azar entre las válidas
        return random.choice(game.available_actions())