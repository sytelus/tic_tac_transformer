from typing import List, Tuple
import torch
import numpy as np


def winning_moves(board, player):
    """Is there any immediate move that will win the game?"""
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                board[i][j] = player
                if check_winner(board) == player:
                    moves.append((i, j))
                board[i][j] = 0
    return moves


def board_full(board):
    return 0 not in board


def check_winner(board):
    """Return the player who has won for the given board, if any"""
    for player in [-1, 1]: # check each player
        for i in range(3):
            # check rows and columns
            if all(board[i, :] == player) or all(board[:, i] == player):
                return player
        # check diagonals
        if all(np.diag(board) == player) or all(np.diag(np.fliplr(board)) == player):
            return player
    return None


def optimal_moves(board, player):
    # Is there any immediate winning move
    moves = winning_moves(board, player)
    if moves:
        return moves

    # Does other player have any immediate move for win?
    moves = winning_moves(board, -player)
    if moves:
        return moves

    # Center move
    if board[1][1] == 0:
        return [(1, 1)]

    # Corner moves
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    return [corner for corner in corners if board[corner[0]][corner[1]] == 0]


def get_valid_moves(board):
    """Get all valid moves for a given board
    :param board: 3x3 numpy array
    :return: list of valid moves as tuples of row, column
    """
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]


def minimax_orig(board, player):
    """
    Minimax algorithm for finding the best move for a given board and player.
    The function evaluates the board for the given player and returns the best score and the best move.
    """
    winner = check_winner(board)
    if winner is not None:
        return winner*player, None
    if board_full(board):
        return 0, None

    best_score = float("-inf") * player
    best_move = None
    for i, j in get_valid_moves(board):
        board[i][j] = player
        # evaluate the board for the other player
        # if other player is winning, minimax score will be 1
        score, _ = minimax(board, -player)
        score = -score # their good score is our bad score and vice versa
        board[i][j] = 0
        if score > best_score:
            best_score = score
            best_move = (i, j)
    return best_score, best_move


def minimax_all(board, player, win_games:List[tuple], draw_games:List[tuple], lose_games:List[tuple], move_hist:tuple=()):
    """
    Minimax algorithm for finding the best move for a given board and player.
    The function evaluates the board for the given player and returns the best score and the best move.
    """
    winner = check_winner(board)
    if winner is not None:
        if winner == 1:
            win_games.append(move_hist)
        else:
            lose_games.append(move_hist)
        return winner*player, None
    if board_full(board):
        draw_games.append(move_hist)
        return 0, None

    best_score = float("-inf") * player
    best_move = None
    for i, j in get_valid_moves(board):
        board[i][j] = player
        # evaluate the board for the other player
        # if other player is winning, minimax score will be 1
        score, _ = minimax_all(board, -player, win_games, draw_games, lose_games, move_hist + ((i, j),))
        score = -score # their good score is our bad score and vice versa
        board[i][j] = 0
        if score > best_score:
            best_score = score
            best_move = (i, j)
    return best_score, best_move

def get_all_trajectories(player):
    """
    Given who plays first, generate all possible trajectories
    """
    board = np.zeros((3, 3), dtype=int)
    win_games, draw_games, lose_games = [], [], []
    minimax_all(board, player, win_games, draw_games, lose_games)

    return win_games, draw_games, lose_games

def encode_moves(moves:Tuple[Tuple[int, int]]):
    """
    Encode a sequence of moves as a string of the form "A1 B2 C3"
    """
    return " ".join([chr(ord('A') - 1 + i)+str(j) for i, j in moves])

if __name__ == "__main__":
    win_games_x, draw_games_x, lose_games_x = get_all_trajectories(1)

    lines = []
    for game in win_games_x:
        lines.append("X " + encode_moves(game))
    for game in draw_games_x:
        lines.append("X " + encode_moves(game))

    win_games_o, draw_games_o, lose_games_o = get_all_trajectories(-1)
    for game in lose_games_o:
        lines.append("O " + encode_moves(game))
    for game in draw_games_o:
        lines.append("O " + encode_moves(game))
