from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, player_id):
        self.player_id = player_id  # 1 o 2

    @abstractmethod
    def select_action(self, game):
        """
        Dado el estado actual del juego, devuelve la columna (0 a 6) donde el agente quiere jugar.
        """
        pass