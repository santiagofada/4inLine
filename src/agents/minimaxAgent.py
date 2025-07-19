import math, random, copy, hashlib
from src.agents.base import Agent

ROWS, COLS = 6, 7
CENTER      = COLS // 2
WIN_VALUE   = 1_000_000            # valor de victoria
ORDER       = [3,2,4,1,5,0,6]      # columnas por prioridad (centro‑primero)

# ---------- utilidades ----------
def board_hash(board, player_turn):
    h = hashlib.blake2b(digest_size=8)
    for row in board: h.update(bytes(row))
    h.update(bytes([player_turn]))
    return h.digest()              # 64 bit hash

def score_window(w, pid):
    opp = 3 - pid
    if w.count(pid) == 4:                        return 1000
    if w.count(pid) == 3 and w.count(0) == 1:    return 50
    if w.count(pid) == 2 and w.count(0) == 2:    return 10
    if w.count(opp)== 3 and w.count(0) == 1:     return -80
    return 0

def count_immediate_wins(game, pid):
    wins = 0
    for col in game.available_actions():
        g = copy.deepcopy(game)
        g.current_player = pid
        g.make_move(col)
        if g.winner == pid:
            wins += 1
    return wins

# ---------- Minimax con mejoras ----------
class MinimaxAgent(Agent):
    def __init__(self, player_id, depth=8):
        super().__init__(player_id)
        self.max_depth = depth
        self.tt = {}                       # transposition table

    # -------- heurística mejorada --------
    def evaluate(self, board, game):
        pid, score = self.player_id, 0
        # centro
        score += [board[r][CENTER] for r in range(ROWS)].count(pid) * 6
        # alineaciones
        for r in range(ROWS):
            for c in range(COLS-3):
                score += score_window([board[r][c+i] for i in range(4)], pid)
        for c in range(COLS):
            for r in range(ROWS-3):
                score += score_window([board[r+i][c] for i in range(4)], pid)
        for r in range(ROWS-3):
            for c in range(COLS-3):
                score += score_window([board[r+i][c+i]   for i in range(4)], pid)
                score += score_window([board[r+3-i][c+i] for i in range(4)], pid)
        # amenazas dobles
        my_threats  = count_immediate_wins(game, pid)
        op_threats  = count_immediate_wins(game, 3-pid)
        score += 400 * (my_threats  >= 2)
        score -= 400 * (op_threats  >= 2)
        return score

    # -------- minimax α‑β + tabla --------
    def _minimax(self, game, depth, alpha, beta, maximizing):
        key = (board_hash(game.board, game.current_player), depth, maximizing)
        if key in self.tt:                        # ↳ transposition hit
            return self.tt[key]

        # terminal
        if depth == 0 or game.winner or game.is_draw():
            if game.winner == self.player_id:         return (None,  WIN_VALUE)
            if game.winner == 3 - self.player_id:     return (None, -WIN_VALUE)
            if game.is_draw():                        return (None, 0)
            return (None, self.evaluate(game.board, game))

        valid = game.available_actions()
        valid.sort(key=lambda c: ORDER.index(c))

        if maximizing:
            best_val, best_col = -math.inf, valid[0]
            for col in valid:
                g = copy.deepcopy(game)
                g.make_move(col)
                _, val = self._minimax(g, depth-1, alpha, beta, False)
                if val > best_val:
                    best_val, best_col = val, col
                alpha = max(alpha, best_val)
                if alpha >= beta: break
        else:
            best_val, best_col = math.inf, valid[0]
            for col in valid:
                g = copy.deepcopy(game)
                g.current_player = 3 - g.current_player
                g.make_move(col)
                _, val = self._minimax(g, depth-1, alpha, beta, True)
                if val < best_val:
                    best_val, best_col = val, col
                beta = min(beta, best_val)
                if alpha >= beta: break

        self.tt[key] = (best_col, best_val)      # guardar en TT
        return best_col, best_val

    # -------- iterative deepening ---------
    def select_action(self, game):
        best_move = random.choice(game.available_actions())
        for d in range(1, self.max_depth + 1):
            self.tt.clear()                      # opcional: limpiar entre iteraciones
            move, _ = self._minimax(game, d, -math.inf, math.inf,
                                    maximizing=(game.current_player == self.player_id))
            if move is not None:
                best_move = move
        return best_move
