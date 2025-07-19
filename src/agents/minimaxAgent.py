import math, random, hashlib
from src.agents.base import Agent

# Constantes del juego
ROWS, COLS = 6, 7
CENTER     = COLS // 2
WIN        = 1_000_000
ORDER      = [3, 2, 4, 1, 5, 0, 6]  # Orden preferido de columnas (centro primero)

def board_hash(board, turn):
    #  hashing para tabla de transposición, suirve para no repetir estados

    h = hashlib.blake2b(digest_size=8)
    for row in board: h.update(bytes(row))
    # incluye el jugador actual porque un mismo tablero puede significar lo mismo segun el jugador
    h.update(bytes([turn]))
    return h.digest()

def score_window(w, pid):
    # heuristica que evalua 4 celdas, y asigna un puntaje

    opp = 3 - pid
    if w.count(pid) == 4:                        return 1000
    if w.count(pid) == 3 and w.count(0) == 1:    return 50
    if w.count(pid) == 2 and w.count(0) == 2:    return 10
    if w.count(opp)== 3 and w.count(0) == 1:     return -80
    return 0

class MinimaxAgent(Agent):
    def __init__(self, player_id, depth=8):
        super().__init__(player_id)
        self.base_depth = depth   # profundidad máxima
        self.tt = {}              # tabla de transposición, sirve para no repetir estados

    # funcion heuristica para un tablero completo, parte de la heurística en score_window
    def evaluate(self, board):
        pid, score = self.player_id, 0
        # Prioriza fichas en el centro
        score += [board[r][CENTER] for r in range(ROWS)].count(pid) * 6

        # bloques verticales
        for r in range(ROWS):
            for c in range(COLS-3):
                score += score_window([board[r][c+i] for i in range(4)], pid)

        # bloques horizontales
        for c in range(COLS):
            for r in range(ROWS-3):
                score += score_window([board[r+i][c] for i in range(4)], pid)

        # Diagonales
        for r in range(ROWS-3):
            for c in range(COLS-3):
                score += score_window([board[r+i][c+i]   for i in range(4)], pid)
                score += score_window([board[r+3-i][c+i] for i in range(4)], pid)
        return score

    # --- aplicar una jugada y simular el cambio de turno ---
    def _make(self, game, col):
        for row in range(ROWS-1, -1, -1):
            if game.board[row][col] == 0:
                game.board[row][col] = game.current_player
                prev_winner = game.winner
                # actualiza si hay ganador
                if self._is_win(game.board, row, col, game.current_player):
                    game.winner = game.current_player
                else:
                    game.winner = None
                # cambia turno
                game.current_player = 3 - game.current_player
                return row, prev_winner

    # --- deshacer jugada y restaurar estado anterior ---
    @staticmethod
    def _undo(game, col, row, prev_winner):
        game.current_player = 3 - game.current_player
        game.board[row][col] = 0
        game.winner = prev_winner

    # --- verifica si hay 4 en línea desde (r,c) ---
    @staticmethod
    def _is_win(b, r, c, pid):
        for dr,dc in [(0,1),(1,0),(1,1),(1,-1)]:
            cnt = 1
            for d in (1,-1):
                rr,cc = r+d*dr, c+d*dc
                while 0 <= rr < ROWS and 0 <= cc < COLS and b[rr][cc] == pid:
                    cnt += 1
                    rr += d*dr
                    cc += d*dc
            if cnt >= 4: return True
        return False

    def _minimax(self, game, depth, alpha, beta, max_turn):
        # Crea clave unica para el estado actual del tablero, jugador
        key = board_hash(game.board, game.current_player)

        # Si ya evaluamos este estado a igual o mayor profundidad, reutilizamos el resultado
        if key in self.tt and self.tt[key][2] >= depth:
            return self.tt[key][:2]


        # Caso base: si llegamos a profundidad 0, hay un ganador o empate
        if depth == 0 or game.winner or game.is_draw():
            if game.winner == self.player_id:
                val = WIN
            elif game.winner == 3 - self.player_id:
                val = -WIN
            elif game.is_draw():
                val = 0
            else:
                val = self.evaluate(game.board)  # evaluar estado intermedio con heuristica

            return None, val

        # Ordena las acciones disponibles priorizando el centro
        moves = game.available_actions()
        moves.sort(key=lambda c: abs(c - CENTER))

        best_action = moves[0]  # valor por defecto

        if max_turn:
            # Jugador que maximiza
            best_value = -math.inf
            for action in moves:
                row, prev_w = self._make(game, action)  # hace jugada
                _, val = self._minimax(game, depth - 1, alpha, beta, False)  # llamada recursiva
                self._undo(game, action, row, prev_w)  # deshace jugada

                # Actualiza el mejor valor
                if val > best_value:
                    best_value, best_action = val, action

                alpha = max(alpha, best_value)  # actualiza alpha
                if alpha >= beta:
                    break  # poda beta: no vale la pena seguir
        else:
            # Jugador que minimiza
            best_value = math.inf
            for action in moves:
                row, prev_w = self._make(game, action)
                _, val = self._minimax(game, depth - 1, alpha, beta, True)#  llamada recursiva
                self._undo(game, action, row, prev_w)

                if val < best_value:
                    best_value, best_action = val, action

                beta = min(beta, best_value)  # actualiza beta
                if alpha >= beta:
                    break  # poda alpha, no vale la pena seguir

        # Guarda el resultado en la tabla de transposición
        self.tt[key] = (best_action, best_value, depth)
        return best_action, best_value

    # --- búsqueda con iterative deepening ---
    def select_action(self, game):
        empty = sum(r.count(0) for r in game.board)       # cuenta casillas vacías
        max_depth = min(self.base_depth, empty)           # limita profundidad máxima
        best = random.choice(game.available_actions())     # fallback

        if empty == 42:  # tablero vacío entonces limpiar TT
            self.tt.clear()

        # búsqueda con profundidad creciente
        for d in range(1, max_depth+1):
            action, _ = self._minimax(
                game, d, -math.inf, math.inf,
                max_turn=(game.current_player == self.player_id)
            )
            if action is not None:
                best = action  # actualiza mejor jugada encontrada
        return best
