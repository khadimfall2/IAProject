import logging
logging.getLogger('PIL').setLevel(logging.INFO)  # Réduit les messages de débogage de PIL

import argparse
import logging
import pandas as pd
import matplotlib  # Importez matplotlib en premier
matplotlib.use('TkAgg')  # Configurez le backend avant d'importer pyplot ou seaborn
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print
from rich.logging import RichHandler
from classes.strategy import QLearningPlayer, str2strat
from classes.tournament import Tournament

# Configuration du logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)  # Configurez le logging après les imports
FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

# La fonction et le reste de votre code...


def create_heatmap(csv_file):
    df = pd.read_csv(csv_file)
    pivot_table = df.pivot_table(index='player1_strategy', columns='player2_strategy', values='winner', aggfunc='count')
    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
    plt.title('Résultats du Championnat de Hex')
    plt.xlabel('Stratégie du Joueur 2')
    plt.ylabel('Stratégie du Joueur 1')
    plt.show()

def main(args):
    # Initialisation du tournoi avec les arguments
    arena = Tournament([args.size, [args.player, args.other], 0, args.games, not args.no_ui])

    q_learning_player = None
    # Logique pour QLearningPlayer
    if 'qlearning' in [args.player, args.other]:
        player_number = [args.player, args.other].index('qlearning') + 1
        initial_board_state = arena.get_initial_board_state()
        q_learning_player = QLearningPlayer(_board_state=initial_board_state, player=player_number, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
        if args.qtable_load:
            q_learning_player.load_q_table(args.qtable_load)

    # Exécuter le tournoi ou un simple jeu
    if args.games > 1:
        arena.championship()
    else:
        arena.single_game(black_starts=True)

    # Sauvegarder la table Q après le tournoi si nécessaire
    if q_learning_player and args.qtable_save:
        q_learning_player.save_q_table(args.qtable_save)

    # Créer et afficher le heatmap à la fin du championnat
    create_heatmap('tournament_results.csv')  # Remplacez par le chemin réel vers votre fichier CSV

def arguments():
    parser = argparse.ArgumentParser(description='Runs a game of Hex.')
    parser.add_argument('--alpha', default=0.1, type=float, help='Taux d\'apprentissage pour Q-learning (default: 0.1)')
    parser.add_argument('--gamma', default=0.9, type=float, help='Facteur de réduction pour Q-learning (default: 0.9)')
    parser.add_argument('--epsilon', default=0.1, type=float, help='Facteur d\'exploration pour Q-learning (default: 0.1)')
    parser.add_argument('--qtable-save', type=str, help='Chemin pour sauvegarder la table Q')
    parser.add_argument('--qtable-load', type=str, help='Chemin pour charger la table Q')
    parser.add_argument('--size', type=int, default=7, help='Size of the board (default: 7)')
    parser.add_argument('--games', type=int, default=5, help='Number of games to play (default: 5)')
    parser.add_argument('--player', type=str, choices=str2strat.keys(), help='Strategy for player1')
    parser.add_argument('--other', type=str, choices=str2strat.keys(), help='Strategy for player2')
    parser.add_argument('--no-ui', action='store_true', help='Disable the UI')
    return parser.parse_args()

if __name__ == "__main__":
    args = arguments()
    main(args)
