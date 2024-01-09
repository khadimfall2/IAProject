import copy
import math
import pickle
import random
import time
import random
from math import sqrt, log
from math import log, sqrt, inf
#from typing import Self
import numpy as np
from rich.table import Table
from rich.progress import track
from rich.console import Console
from rich.progress import Progress
import random
import classes.logic as logic
import classes.logic as logic

# Base class for different player strategies in the game.
class PlayerStrat:
    def __init__(self, _board_state, player):
        """
        Initialize the player strategy with the current state of the board and the player number.

        :param _board_state: The current state of the board as a 2D list.
        :param player: The player number (1 or 2).
        """
        self.root_state = _board_state
        self.player = player

    def start(self):
        """
        Abstract method to select a tile from the board. To be implemented by subclasses.

        :returns: (x, y) tuple of integers corresponding to a valid and free tile on the board.
        """
        raise NotImplementedError

# Random strategy for a player. Chooses a move randomly from available tiles.
class RandomPlayer(PlayerStrat):
    def __init__(self, _board_state, player):
        super().__init__(_board_state, player)
        self.board_size = len(_board_state)

    def select_tile(self, board):
        """
        Randomly selects a free tile on the board.

        :param board: The current game board.
        :returns: (x, y) tuple of integers corresponding to a valid and free tile on the board.
        """
        free_tiles = [(x, y) for x in range(self.board_size) for y in range(self.board_size) if board[x][y] == 0]
        return random.choice(free_tiles) if free_tiles else None

    def start(self):
        return self.select_tile(self.root_state)
    




class MiniMax(PlayerStrat):
    def __init__(self, _board_state, player, depth=3):
        super().__init__(_board_state, player)
        self.board_size = len(_board_state)  # Taille du plateau de jeu
        self.depth = depth  # Profondeur de recherche pour l'algorithme Minimax

    def select_tile(self, board, player):
        # Sélectionne la meilleure case en fonction du score minimax
        best_score = float('-inf')
        best_move = None
        # Parcourt toutes les cases du plateau
        for x in range(self.board_size):
            for y in range(self.board_size):
                # Vérifie si la case est vide
                if board[x][y] == 0:
                    board[x][y] = player  # Effectue un mouvement hypothétique
                    score = self.minimax(board, self.depth - 1, False, player)
                    board[x][y] = 0  # Annule le mouvement hypothétique
                    # Met à jour le meilleur score et le meilleur mouvement
                    if score > best_score:
                        best_score = score
                        best_move = (x, y)
        return best_move

    def minimax(self, board, depth, is_maximizing, player):
        # Algorithme Minimax récursif pour trouver le meilleur score
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_board(board, player)
        if is_maximizing:
            best_score = float('-inf')
            # Parcourt toutes les cases du plateau pour le joueur maximisant
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if board[x][y] == 0:
                        board[x][y] = player
                        score = self.minimax(board, depth - 1, False, player)
                        board[x][y] = 0
                        best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            # Parcourt toutes les cases du plateau pour le joueur minimisant
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if board[x][y] == 0:
                        board[x][y] = 3 - player
                        score = self.minimax(board, depth - 1, True, player)
                        board[x][y] = 0
                        best_score = min(best_score, score)
            return best_score

    def is_game_over(self, board):
        # Vérifie si le jeu est terminé
        return logic.is_game_over(self.player, board) is not None

    def custom_heuristic(self, board, player):
        # Heuristique personnalisée qui évalue le plateau de jeu
        score = 0
        # Compte les pièces alignées pour le joueur
        for row in board:
            score += self.count_pieces_in_line(row, player)
        for col in np.transpose(board):
            score += self.count_pieces_in_line(col, player)
        diagonals = [np.diagonal(board), np.diagonal(np.flipud(board))]
        for diag in diagonals:
            score += self.count_pieces_in_line(diag, player)
        return score

    def count_pieces_in_line(self, line, player):
        # Compte les pièces d'un joueur dans une ligne, colonne ou diagonale
        count = 0
        for piece in line:
            if piece == player:
                count += 1
            elif piece == 3 - player:
                return 0
        return count

    def evaluate_board(self, board, player):
        # Évalue le plateau de jeu et retourne un score
        if logic.is_game_over(player, board):
            return 10
        elif logic.is_game_over(3 - player, board):
            return -10
        else:
            return self.custom_heuristic(board, player)

    def start(self):
        # Commence l'algorithme et retourne le meilleur mouvement
        return self.select_tile(self.root_state, self.player)
       






import numpy as np  # Assurez-vous d'importer numpy pour pouvoir utiliser ses fonctionnalités

class MiniMaxPlus(PlayerStrat):
    def __init__(self, _board_state, player, depth=3):
        super().__init__(_board_state, player)  # Appel au constructeur de la classe parente
        self.board_size = len(_board_state)  # La taille du plateau de jeu
        self.depth = depth  # La profondeur de l'arbre de recherche Minimax

    def select_tile(self, board, player):
        # Sélectionne le meilleur coup en utilisant l'algorithme Minimax
        best_score = float('-inf')  # Initialise le meilleur score à -infini
        best_move = None  # Initialise le meilleur coup à None

        # Parcourt toutes les cases du plateau
        for x in range(self.board_size):
            for y in range(self.board_size):
                # Vérifie si la case est vide
                if board[x][y] == 0:
                    board[x][y] = player  # Fait un coup hypothétique
                    score = self.minimax(board, self.depth - 1, False, player)  # Appelle la fonction minimax
                    board[x][y] = 0  # Annule le coup hypothétique
                    # Si le score obtenu est meilleur que le meilleur score, met à jour le meilleur score et le meilleur coup
                    if score > best_score:
                        best_score = score
                        best_move = (x, y)

        return best_move  # Retourne le meilleur coup

    def minimax(self, board, depth, is_maximizing, player, alpha=float('-inf'), beta=float('inf')):
        # Algorithme Minimax avec élagage alpha-bêta
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_board(board, player)  # Évalue le plateau de jeu si la profondeur est 0 ou si le jeu est terminé

        # Si c'est le tour du joueur maximisant
        if is_maximizing:
            best_score = float('-inf')  # Initialise le meilleur score à -infini
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if board[x][y] == 0:
                        board[x][y] = player
                        score = self.minimax(board, depth - 1, False, player, alpha, beta)  # Appelle récursivement minimax
                        board[x][y] = 0
                        best_score = max(best_score, score)  # Met à jour le meilleur score
                        alpha = max(alpha, best_score)  # Met à jour alpha
                        if beta <= alpha:
                            break  # Coupe les branches inutiles
            return best_score
        else:
            best_score = float('inf')  # Initialise le meilleur score à infini pour le joueur minimisant
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if board[x][y] == 0:
                        board[x][y] = 3 - player
                        score = self.minimax(board, depth - 1, True, player, alpha, beta)  # Appelle récursivement minimax
                        board[x][y] = 0
                        best_score = min(best_score, score)  # Met à jour le meilleur score
                        beta = min(beta, best_score)  # Met à jour beta
                        if beta <= alpha:
                            break  # Coupe les branches inutiles
            return best_score

    def proximity_to_victory_heuristic(self, board, player):
        # Heuristique pour évaluer la proximité d'un joueur à la victoire dans le jeu Hex
        score = 0
        opponent = logic.BLACK_PLAYER if player == logic.WHITE_PLAYER else logic.WHITE_PLAYER
        board_size = len(board)
        
        # Crée une matrice pour suivre les chemins presque complets
        path_matrix = np.zeros((board_size, board_size), dtype=int)
        
        # Parcourt toutes les cases pour marquer les positions du joueur actuel
        for x in range(board_size):
            for y in range(board_size):
                if board[x][y] == player:
                    path_matrix[x][y] = 1  # Marque la position du joueur

                    # Vérifie si le joueur est proche de gagner
                    if (player == logic.BLACK_PLAYER and y == board_size - 1) or \
                       (player == logic.WHITE_PLAYER and x == board_size - 1):
                        score += 10  # Attribue un score élevé pour être près de la victoire

        # Évalue les chemins presque complets pour augmenter le score
        for x in range(board_size):
            for y in range(board_size):
                if path_matrix[x][y] == 1:
                    neighbours = logic.get_neighbours((x, y), board_size)
                    path_length = 1
                    for neighbour in neighbours:
                        if path_matrix[neighbour] == 1:
                            path_length += 1
                        elif board[neighbour] == opponent:
                            path_length = 0
                            break
                    score += path_length  # Ajoute la longueur du chemin au score final
        
        return score  # Retourne le score basé sur la proximité de la victoire

    def evaluate_board(self, board, player):
        # Évalue le plateau de jeu et retourne un score basé sur l'état du jeu
        if logic.is_game_over(player, board):
            return float('inf')  # Retourne infini si le joueur a gagné
        elif logic.is_game_over(3 - player, board):
            return float('-inf')  # Retourne -infini si l'adversaire a gagné
        else:
            # Combine l'heuristique personnalisée et la proximité de la victoire pour obtenir le score
            return self.custom_heuristic(board, player) + self.proximity_to_victory_heuristic(board, player)

    def is_game_over(self, board):
        # Vérifie si le jeu est terminé
        return logic.is_game_over(self.player, board) is not None

    def custom_heuristic(self, board, player):
        # Heuristique personnalisée pour évaluer le plateau de jeu
        score = 0
        # Évalue chaque ligne, colonne et diagonale pour calculer le score
        for row in board:
            score += self.count_pieces_in_line(row, player)
        for col in np.transpose(board):
            score += self.count_pieces_in_line(col, player)
        diagonals = [np.diagonal(board), np.diagonal(np.flipud(board))]
        for diag in diagonals:
            score += self.count_pieces_in_line(diag, player)
        return score  # Retourne le score calculé

    def count_pieces_in_line(self, line, player):
        # Compte le nombre de pièces du joueur dans une ligne, colonne ou diagonale
        count = 0
        for piece in line:
            if piece == player:
                count += 1
            elif piece == 3 - player:
                return 0  # Retourne 0 si une pièce de l'adversaire est trouvée
        return count  # Retourne le nombre total de pièces en ligne

    def evaluate_board(self, board, player):
        # Évalue le plateau de jeu et retourne un score
        if logic.is_game_over(player, board):
            return 10  # Retourne 10 si le joueur a gagné
        elif logic.is_game_over(3 - player, board):
            return -10  # Retourne -10 si l'adversaire a gagné
        else:
            return self.custom_heuristic(board, player)  # Utilise l'heuristique personnalisée pour le score

    def start(self):
        # Commence l'algorithme Minimax et retourne le meilleur coup
        return self.select_tile(self.root_state, self.player)







class MonteCarloPlayer(PlayerStrat):
    def __init__(self, _board_state, player, time_limit=30):  # Temps limite augmenté
        super().__init__(_board_state, player)
        self.time_limit = time_limit
        self.board_size = len(_board_state)
        self.MAX_TURNS = self.determine_max_turns(self.board_size)

    def determine_max_turns(self, board_size):
        return board_size ** 2

    def create_node(self, state, parent, move):
        untried_moves = logic.get_possible_moves(state)
        return {"state": state, "parent": parent, "move": move, 
                "untried_moves": untried_moves, "children": [], 
                "wins": 0, "visits": 0}

    def add_child(self, node, move, state):
        new_node = self.create_node(state, node, move)
        node["children"].append(new_node)
        node["untried_moves"].remove(move)
        return new_node

    def select_child(self, node):
        best_score = -float('inf')
        best_child = None
        exploration_factor = 1.41  # Facteur d'exploration ajusté

        for child in node["children"]:
            exploit = child["wins"] / child["visits"]
            explore = exploration_factor * math.sqrt(math.log(node["visits"]) / child["visits"])
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def backpropagate(self, node, result):
        while node:
            node["visits"] += 1
            if node["parent"] and node["parent"]["state"][node["move"]] == self.player:
                node["wins"] += result
            node = node["parent"]

    def simulate(self, state):
        current_player = self.player
        turn_count = 0

        while not logic.is_game_over(current_player, state) and turn_count < self.MAX_TURNS:
            possible_moves = logic.get_possible_moves(state)
            if not possible_moves:
                break

            move = random.choice(possible_moves)
            state[move] = current_player
            current_player = logic.BLACK_PLAYER if current_player == logic.WHITE_PLAYER else logic.WHITE_PLAYER
            turn_count += 1

            if turn_count >= 2:  # Seuil pour l'heuristique
                score = self.heuristic_evaluation(state, self.player)
                if score > 0:
                    return score

        return 1 if logic.is_game_over(self.player, state) else -1 if logic.is_game_over(3 - self.player, state) else 0

    def get_best_move(self, root):
        best_visits = -float('inf')
        best_move = None

        for child in root["children"]:
            if child["visits"] > best_visits:
                best_visits = child["visits"]
                best_move = child["move"]

        return best_move

    def heuristic_evaluation(self, state, player):
        player_tiles = logic.get_player_tiles(state, player)
        score = 0

        for tile in player_tiles:
            score += self.evaluate_tile(tile, state, player)

        return score

    def evaluate_tile(self, tile, state, player):
        score = 0
        if self.is_close_to_border(tile, state.shape[0]):
            score += 1
        return score

    def is_close_to_border(self, tile, board_size):
        x, y = tile
        return x in [0, board_size - 1] or y in [0, board_size - 1]

    def start(self):
      root = self.create_node(self.root_state, None, None)
      end_time = time.time() + self.time_limit

      while time.time() < end_time:
        node = root
        state = copy.deepcopy(self.root_state)

        while node["untried_moves"] == [] and node["children"]:
            node = self.select_child(node)
            state[node["move"]] = self.player

        if node["untried_moves"]:
            move = random.choice(node["untried_moves"])
            state[move] = self.player
            node = self.add_child(node, move, state)

        result = self.simulate(copy.deepcopy(state))
        self.backpropagate(node, result)

      return self.get_best_move(root)  # Corrigé ici





class QLearningPlayer(PlayerStrat):
    def __init__(self, _board_state, player, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(_board_state, player)  # Initialisation de la classe parente
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur de discount pour les récompenses futures
        self.epsilon = epsilon  # Taux d'exploration
        self.q_table = {}  # Tableau Q pour stocker les valeurs Q
        self.board_size = len(_board_state)  # Taille du plateau de jeu

    def start(self):
        # Démarre le choix du coup
        return self.select_tile(self.root_state)

    def select_tile(self, board):
        # Sélectionne un coup sur le plateau
        state = self.state_to_key(board)  # Convertit l'état du plateau en clé pour la Q-table
        # Choix entre exploration et exploitation
        if random.uniform(0, 1) < self.epsilon:
            return self.choose_random_move(board)  # Exploration : choix aléatoire
        else:
            return self.choose_best_move(state)  # Exploitation : choix du meilleur coup selon la Q-table

    def choose_random_move(self, board):
        # Choix d'un coup aléatoire sur les cases libres
        free_tiles = [(x, y) for x in range(self.board_size) for y in range(self.board_size) if board[x][y] == 0]
        return random.choice(free_tiles) if free_tiles else None

    def choose_best_move(self, state):
        # Choix du meilleur coup basé sur la Q-table
        if state not in self.q_table:
            return self.choose_random_move(self.root_state)  # Si l'état n'est pas dans la Q-table, choix aléatoire

        # Sélectionne le coup avec la plus haute valeur Q
        max_q = max(self.q_table[state].values(), default=0)
        best_actions = [action for action in self.q_table[state] if self.q_table[state][action] == max_q]
        return random.choice(best_actions) if best_actions else None

    def update_q_table(self, prev_state, action, reward, new_state):
        # Met à jour la Q-table avec la nouvelle information
        if prev_state not in self.q_table:
            self.q_table[prev_state] = {}
        if action not in self.q_table[prev_state]:
            self.q_table[prev_state][action] = 0
        # Calcul de la nouvelle valeur Q
        prev_q_value = self.q_table[prev_state][action]
        max_future_q = max(self.q_table.get(new_state, {}).values(), default=0)
        new_q_value = prev_q_value + self.alpha * (reward + self.gamma * max_future_q - prev_q_value)
        self.q_table[prev_state][action] = new_q_value

    def update_epsilon(self, decay_rate, min_epsilon):
        # Met à jour le taux d'exploration (epsilon)
        self.epsilon = max(self.epsilon * decay_rate, min_epsilon)

    def state_to_key(self, board):
        # Convertit l'état du plateau en une clé utilisable pour la Q-table
        return str(board)

    def save_q_table(self, file_path):
        # Sauvegarde la Q-table dans un fichier
        with open(file_path, 'wb') as file:
            pickle.dump(self.q_table, file)

    def load_q_table(self, file_path):
        # Charge la Q-table à partir d'un fichier
        with open(file_path, 'rb') as file:
            self.q_table = pickle.load(file)

str2strat: dict[str, PlayerStrat] = {
    "human": None,
    "random": RandomPlayer,
    "minimax": MiniMax,
    "minimaxplus": MiniMaxPlus,
    "montecarlo": MonteCarloPlayer,
    "qlearning": QLearningPlayer,  
}

