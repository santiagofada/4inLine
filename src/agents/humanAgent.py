from src.agents.base import Agent

class HumanAgent(Agent):
    def select_action(self, game):
        valid_actions = game.available_actions()
        while True:
            try:
                action = int(input(f"Jugador {self.player_id}, elige una columna (1-7): ")) - 1
                if action in valid_actions:
                    return action
                else:
                    print("Columna inválida o llena. Intenta otra.")
            except ValueError:
                print("Entrada inválida. Debes ingresar un número del 0 al 6.")
