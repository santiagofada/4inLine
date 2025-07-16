from src.agents.base import Agent
import random
import copy

class AdvancedHeuristicAgent(Agent):
    def select_action(self, game):
        opponent_id = 3 - self.player_id
        valid_moves = game.available_actions()

        # 1. Ganar si puede
        for action in valid_moves:
            temp = copy.deepcopy(game)
            temp.current_player = self.player_id
            temp.make_move(action)
            if temp.winner == self.player_id:
                return action

        # 2. Bloquear si el oponente puede ganar
        for action in valid_moves:
            temp = copy.deepcopy(game)
            temp.current_player = opponent_id
            temp.make_move(action)
            if temp.winner == opponent_id:
                return action

        # 3. Buscar jugadas que generen 2 amenazas simultáneas (fork)
        for action in valid_moves:
            temp = copy.deepcopy(game)
            temp.current_player = self.player_id
            temp.make_move(action)
            count = 0
            for follow_up in temp.available_actions():
                g2 = copy.deepcopy(temp)
                g2.current_player = self.player_id
                g2.make_move(follow_up)
                if g2.winner == self.player_id:
                    count += 1
            if count >= 2:
                return action

        # 4. Bloquear fork del oponente
        for action in valid_moves:
            temp = copy.deepcopy(game)
            temp.current_player = opponent_id
            temp.make_move(action)
            count = 0
            for follow_up in temp.available_actions():
                g2 = copy.deepcopy(temp)
                g2.current_player = opponent_id
                g2.make_move(follow_up)
                if g2.winner == opponent_id:
                    count += 1
            if count >= 2:
                return action

        # 5. Priorizar columnas centrales
        center = len(game.board[0]) // 2
        prioritized = sorted(valid_moves, key=lambda x: abs(center - x))
        return prioritized[0]
