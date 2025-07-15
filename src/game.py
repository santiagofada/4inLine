ROWS = 6
COLS = 7
SYMBOLS = {0: ' ', 1: 'X', 2: 'O'}

class Connect4:
    def __init__(self):
        self.board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.current_player = 1
        self.winner = None

    def available_actions(self):
        return [col for col in range(COLS) if self.board[0][col] == 0]

    def make_move(self, col):
        if col not in self.available_actions():
            return False
        for row in reversed(range(ROWS)):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                if self.check_winner(row, col):
                    self.winner = self.current_player
                self.current_player = 3 - self.current_player  # Alterna entre 1 y 2
                return True
        return False

    def check_winner(self, row, col):
        player = self.board[row][col]
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            for d in [-1, 1]:
                r, c = row + d*dr, col + d*dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == player:
                    count += 1
                    r += d*dr
                    c += d*dc
            if count >= 4:
                return True
        return False

    def is_draw(self):
        return all(self.board[0][col] != 0 for col in range(COLS)) and self.winner is None

    def print_board(self):
        for row in self.board:
            print('|' + '|'.join(SYMBOLS[cell] for cell in row) + '|', flush=True)
        print('-' * (2 * COLS + 1), flush=True)
        print(' ' + ' '.join(str(i+1) for i in range(COLS)), flush=True)
