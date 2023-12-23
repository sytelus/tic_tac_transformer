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

TGameNodeKey = Tuple[int, int]
TGameTree = Tree[GameNode, TGameNodeKey]

def create_game_node(depth, start_player, to_here, is_optimal)->TGameTree:
    return TGameTree(GameNode(depth=depth, player=start_player, to_here=to_here, is_optimal=is_optimal),
                     node_key_fn=lambda node: node.to_here)

def minimax_all(board, game_tree:TGameTree, for_player)->TGameTree:
    """
    Minimax algorithm for finding the best move for a given board and player.
    The function evaluates the board for the given player and returns the best score and the best move.
    """
    assert for_player in [-1, 1]
    assert game_tree.value.player in [-1, 1]
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
        node = create_game_node(game_tree.value.depth + 1, -player, (i, j), is_optimal=False)
        game_tree.add(node)
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

def get_all_trajectories(for_player:int, start_player:int):
    """
    Given who plays first, generate all possible trajectories
    """
    root = create_game_node(0, start_player, (-1, -1), is_optimal=True)
    board = np.zeros((3, 3), dtype=int)
    minimax_all(board, root, for_player=for_player)
    return root

def encode_moves(ancestors:List[TGameTree])->str:
    """
    Encode a sequence of moves as a string of the form "X A1 B2 C3"
    First character is the player who played first, rest are moves
    """
    moves = "X" if ancestors[0].value.player == 1 else "O"
    moves += " " + " ".join([chr(ord('A') + node.value.to_here[0]) +
                             str(node.value.to_here[1])
                    for node in ancestors[1:]])
    return moves

def game_states(node:TGameTree, states=defaultdict(int)):
    if node.value.winner is not None:
        assert node.is_leaf()
        states[node.value.winner] += 1
    return states

def get_optimal_games(for_player:int, node:TGameTree, optimal_games, non_optimal_games)->None:
    if node.value.winner is not None:
        assert node.is_leaf()
        if node.value.winner != -for_player: # win or draw
            path = list(reversed(list(node.ancestors())))
            # if for_player played optimally
            if all([node.value.is_optimal or node.value.player != for_player for node in path]):
                optimal_games.append(encode_moves(path))
            else:
                non_optimal_games.append(encode_moves(path))
    else:
        for child in node:
            get_optimal_games(for_player, child, optimal_games, non_optimal_games)

def get_game_result(game:str, board=None)->Optional[int]:
    board = np.zeros((3, 3), dtype=int) if board is None else board
    moves = game.split()
    player = 1 if moves[0] == 'X' else -1
    for mi, move in enumerate(moves[1:]):
        i, j = ord(move[0]) - ord('A'), int(move[1])
        assert board[i][j] == 0
        board[i][j] = player
        player = -player
    return check_winner(board)

def get_game_node(game:str, root:TGameTree)->Optional[TGameTree]:
    moves = game.split()
    player = 1 if moves[0] == 'X' else -1

    node = root
    for move in moves[1:]:
        assert node.value.player == player
        i, j = ord(move[0]) - ord('A'), int(move[1])
        node = node.children[(i, j)]
        player = -player

    assert node.value.player == player
    return node

if __name__ == "__main__":
    for_player, start_player = 1, 1
    optimal_games, non_optimal_games = [], []

    root = get_all_trajectories(for_player=for_player, start_player=start_player)
    get_optimal_games(for_player, root, optimal_games=optimal_games, non_optimal_games=non_optimal_games)

    print(f"Optimal games: {len(optimal_games)}")
    print(f"Non-optimal games: {len(non_optimal_games)}")
    print(f"Total games: {len(optimal_games) + len(non_optimal_games)}")

    stats = root.depth_first_traversal(game_states, defaultdict(int))
    print(f"Winning games: {stats[1]}")
    print(f"Losing games: {stats[-1]}")
    print(f"Draw games: {stats[0]}")

    # validate optimal game move correctness
    for game in optimal_games:
        winner = get_game_result(game)
        assert winner == for_player or winner == 0
    print('All optimal games are indeed win or draw')

    # validate optimality
    checked_nodes = set()
    for game in optimal_games:
        game_leaf = get_game_node(game, root)
        assert game_leaf is not None
        assert game_leaf.value.winner is not None and (game_leaf.value.winner == for_player or game_leaf.value.winner == 0)
        path = list(reversed(list(game_leaf.ancestors())))
        for node in path:
            # each node in the path must be optimal for for_player
            assert node.value.player == -for_player or node.value.is_optimal

            # each move from other player results in win or draw if for_player plays optimally
            if node.value.player == -for_player:
                if node in checked_nodes:
                    continue
                checked_nodes.add(node)
                for leaf in node.all_leaves():
                    if all(n.value.is_optimal for n in leaf.ancestors() if n.value.player==for_player):
                        assert leaf.value.winner is not None and (leaf.value.winner == for_player or leaf.value.winner == 0)
    print('All optimal games are indeed optimal')
