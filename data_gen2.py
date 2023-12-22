from collections import defaultdict
from typing import Callable, Set, List, Any, Optional, Iterator

import math
from dataclasses import dataclass
from typing import List, Tuple
import torch
import numpy as np

from tree import Tree


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


def minimax(board, player):
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

@dataclass
class GameNode:
    depth:int
    player: int # 1 or -1, whoes turn it is at this node
    to_here: Tuple[int, int] # the move that was made to get to this node, (-1, -1) for root node
    winner: Optional[int]=None # 1, 0, -1, or None if game is not over
    is_optimal: bool=False # whether parent made optimal move to get to this node
    from_here: Tuple[int, int]=(-1,-1) # the best move that should be made
    from_here_score: float=float('nan') # the score of the best move

def minimax_all(board, game_tree:Tree[GameNode], for_player)->Tree[GameNode]:
    """
    Minimax algorithm for finding the best move for a given board and player.
    The function evaluates the board for the given player and returns the best score and the best move.
    """
    assert len(game_tree) == 0
    assert game_tree.value.winner is None

    player = game_tree.value.player

    winner = check_winner(board)
    if winner is not None:
        game_tree.value.winner = winner
    elif board_full(board):
        game_tree.value.winner = 0
    if game_tree.value.winner is not None:
        game_tree.value.from_here = (-1, -1) # termial node
        # score is wrt to the for_player
        game_tree.value.from_here_score = game_tree.value.winner * player * for_player
        return game_tree

    best_score = float("-inf") * player * for_player
    best_node = None
    for i, j in get_valid_moves(board):
        assert board[i][j] == 0
        board[i][j] = player
        # evaluate the board for the other player
        # if other player is winning, minimax score will be 1
        node = game_tree.add(Tree(GameNode(
            depth = game_tree.value.depth + 1,
            player = -player,
            to_here = (i, j))))
        minimax_all(board, node, for_player)
        board[i][j] = 0

        # score is wrt to the for_player, so negate the score if current player is not same as for_player
        score = node.value.from_here_score
        if player == for_player:
            if score > best_score:
                best_score = score
                best_node = node
        else:
            if score < best_score:
                best_score = score
                best_node = node

    assert best_node is not None

    # multiple moves may have the same score, so we need to mark all of them as optimal
    # note that only parent can set this value as only parent knows the best moves
    for node in game_tree:
        node.value.is_optimal = node.value.from_here_score == best_node.value.from_here_score

    # pick score and next move for this node
    game_tree.value.from_here = best_node.value.to_here
    game_tree.value.from_here_score = best_score

    assert not math.isnan(game_tree.value.from_here_score)

    return best_node

def get_all_trajectories(player):
    """
    Given who plays first, generate all possible trajectories
    """
    root = Tree(GameNode(depth=0, player=player, to_here=(-1, -1), is_optimal=True))
    board = np.zeros((3, 3), dtype=int)
    minimax_all(board, root, for_player=player)
    return root

def encode_moves(moves:Tuple[Tuple[int, int]]):
    """
    Encode a sequence of moves as a string of the form "A1 B2 C3"
    """
    return " ".join([chr(ord('A') - 1 + i)+str(j) for i, j in moves])

def game_states(node:Tree[GameNode], states=defaultdict(int)):
    if node.value.winner is not None:
        assert node.is_leaf()
        states[node.value.winner] += 1
    return states

if __name__ == "__main__":
    root = get_all_trajectories(1)
    print(root.count_all(only_leaves=True))
    print(root.visit_leafs(game_states, defaultdict(int)))
    print(root.breadth_first_traversal(game_states, defaultdict(int)))


    # lines = []
    # for game in win_games_x:
    #     lines.append("X " + encode_moves(game))
    # for game in draw_games_x:
    #     lines.append("X " + encode_moves(game))

    # win_games_o, draw_games_o, lose_games_o = get_all_trajectories(-1)
    # for game in lose_games_o:
    #     lines.append("O " + encode_moves(game))
    # for game in draw_games_o:
    #     lines.append("O " + encode_moves(game))
