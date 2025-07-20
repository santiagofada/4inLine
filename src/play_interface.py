import tkinter as tk
from tkinter import messagebox
import random
from src.game import Connect4, SYMBOLS, ROWS, COLS
from src.agents.humanAgent import HumanAgent

from src.agents.qLearningAgent import QLearningAgent
from src.agents.randomAgent import RandomAgent
from src.agents.advancedHeuristicAgent import AdvancedHeuristicAgent
from src.agents.basicHeuristicAgent import BasicHeuristicAgent
from src.agents.minimaxAgent import MinimaxAgent


CELL_SIZE = 80      # pixels per square
PADDING   = 20      # margin around board
DISC_R    = CELL_SIZE // 2 - 4
BOARD_W   = COLS * CELL_SIZE
BOARD_H   = ROWS * CELL_SIZE

# Define colors (easily tweakable)
COLOR_BG      = "#1877f2"   # Board blue
COLOR_EMPTY   = "white"      
COLOR_P1      = "#e74c3c"   # Red
COLOR_P2      = "#f1c40f"   # Yellow
COLOR_HL      = "#2ecc71"   # Highlight (green)



AGENT1 = "human"
AGENT2 = "minimax"

class Connect4GUI:
    def __init__(self, master):
        self.master = master
        master.title("4-en-Línea Algortimos Toma de Decisiónes")
        self.game = Connect4()

        # pick a rancom player to start
        self.game.current_player = 1 if random.random() < 0.5 else 2



        self.player1 = create_agent(AGENT1, player_id=1,file_path="../assets/q_table_1.pkl")
        self.player2 = create_agent(AGENT2, player_id=2)

        if AGENT1 != "human":
            self.master.after(100, self.auto_play)


        self.canvas = tk.Canvas(master, width=BOARD_W + 2 * PADDING,
                                height=BOARD_H + 2 * PADDING,
                                bg=COLOR_BG, highlightthickness=0)
        self.canvas.pack()

        self.draw_board()

        if AGENT1 == "human":
            self.canvas.bind("<Button-1>", self.on_click)

        # Text/status label
        self.status_var = tk.StringVar()
        self.update_status()
        tk.Label(master, textvariable=self.status_var, font=("Arial", 14)).pack(pady=4)

        tk.Button(master, text="Reiniciar", command=self.reset).pack()

        if self.game.current_player == 2:
            master.after(100, self.oponent_move)

    # ----- drawing helpers -----
    def draw_board(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                x0 = PADDING + c * CELL_SIZE + 4
                y0 = PADDING + r * CELL_SIZE + 4
                x1 = x0 + CELL_SIZE - 8
                y1 = y0 + CELL_SIZE - 8
                pid = self.game.board[r][c]
                color = COLOR_EMPTY if pid == 0 else (COLOR_P1 if pid == 1 else COLOR_P2)
                self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline="black")

    def on_click(self, event):
        if self.game.winner or self.game.is_draw():
            return

        current_player_id = self.game.current_player
        current_agent = self.player1 if current_player_id == 1 else self.player2

        # Solo permitir clicks si el jugador actual es humano
        if not isinstance(current_agent, HumanAgent):
            return

        col = (event.x - PADDING) // CELL_SIZE
        if 0 <= col < COLS:
            self.player_move(col)



    def player_move(self, col):
        # movimiento de un jugador, si no es humano usa select_action
        if not self.game.make_move(col):
            return  # columna ilegal

        self.draw_board()
        if self.check_end():
            return

        # Si el próximo jugador es un agente no humano, que juegue automáticamente
        next_agent = self.player1 if self.game.current_player == 1 else self.player2
        if not isinstance(next_agent, HumanAgent):
            self.master.after(100, self.oponent_move)

    def oponent_move(self):
        if self.game.current_player != 2 or self.game.winner:
            return
        col = self.player2.select_action(self.game)
        self.game.make_move(col)
        self.draw_board()
        self.check_end()

    def check_end(self):
        # mensajes de juego finalizado
        if self.game.winner:
            self.highlight_winning_discs()
            msg = f"¡Ganó el Jugador {self.game.winner}!"
            self.status_var.set(msg)
            #self.master.after(100, lambda: messagebox.showinfo("Fin de partida", msg))
            return True

        if self.game.is_draw():
            self.status_var.set("Empate!")
            #messagebox.showinfo("Fin de partida", "¡Empate!")

            return True
        self.update_status()
        return False

    def reset(self):
        # boton para reinciar una partida
        self.game = Connect4()
        self.game.current_player = 1 if random.random() < 0.5 else 2

        # Vuelve a crear los agentes con los mismos roles
        self.player1 = create_agent(AGENT1, player_id=1)
        self.player2 = create_agent(AGENT2, player_id=2)

        # Redibuja el tablero y actualiza el estado
        self.draw_board()
        self.update_status()

        if AGENT1 == "human":
            self.canvas.bind("<Button-1>", self.on_click)
        else:
            self.canvas.unbind("<Button-1>")

        # Si ambos son agentes o el jugador que comienza no es humano, arrancar automáticamente
        if AGENT1 != "human" and AGENT2 != "human":
            self.master.after(100, self.auto_play)
        elif self.game.current_player == 2 and AGENT2 != "human":
            self.master.after(100, self.oponent_move)

    def update_status(self):
        # mensaje sobre el turno
        return self.status_var.set(f"Turno del Jugador {self.game.current_player}")

    def highlight_winning_discs(self):
        # esta funcion solo enmarca las casillas que generaron una victoria
        pid = self.game.winner
        if not pid:
            return
        dirs = [(0,1),(1,0),(1,1),(1,-1)]
        board = self.game.board
        for row in range(ROWS):
            for col in range(COLS):
                if board[row][col] != pid:
                    continue
                for dr,dc in dirs:
                    cells = [(row + i*dr, col + i*dc) for i in range(4)]
                    # found 4-in-a-row with the same player id
                    if all(0<=rr<ROWS and 0<=cc<COLS and board[rr][cc]==pid for rr,cc in cells):
                        # mark cells by drawing highlight circles on top
                        for rr,cc in cells:
                            x0 = PADDING + cc * CELL_SIZE + 12
                            y0 = PADDING + rr * CELL_SIZE + 12

                            x1 = x0 + CELL_SIZE - 24
                            y1 = y0 + CELL_SIZE - 24

                            self.canvas.create_oval(x0, y0, x1, y1,
                                                    outline=COLOR_HL, width=4)
                        return

    def auto_play(self):
        # funcion que sirve para que 2 agentes juegueen de forma automatica
        if self.game.winner or self.game.is_draw():
            self.check_end()
            return

        current_agent = self.player1 if self.game.current_player == 1 else self.player2
        col = current_agent.select_action(self.game)
        self.game.make_move(col)
        self.draw_board()

        if self.check_end():
            return

        self.master.after(100, self.auto_play)

def create_agent(name, player_id, file_path=None):
    if name == "minimax":
        return MinimaxAgent(player_id=player_id, depth=8)

    elif name == "qlearning":
        if file_path:
            agent = QLearningAgent(player_id=player_id)
            agent.load(file_path)
            return agent
        else:
            raise ValueError(f"Agente desconocido: {name}")


    elif name == "random":
        return RandomAgent(player_id=player_id)

    elif name == "human":
        return HumanAgent(player_id=player_id)

    elif name == "Aheuristic":

        return AdvancedHeuristicAgent(player_id=player_id)

    elif name == "Bheuristic":

        return BasicHeuristicAgent(player_id=player_id)

    else:
        raise ValueError(f"Agente desconocido: {name}")


if __name__ == "__main__":


    root = tk.Tk()
    gui = Connect4GUI(root)
    root.mainloop()
