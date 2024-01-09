import os
import pickle
import logging
import pygame
import pandas as pd
from rich import print
from rich.logging import RichHandler
from classes.logic import player2str
from classes.game import Game

class Tournament:
    def __init__(self, args: list):
        self.args = args
        (self.BOARD_SIZE, self.STRAT, self.GAME_COUNT, self.N_GAMES, self.USE_UI) = args
        self.game_results = []  # Attribut pour stocker les résultats des jeux actuels

        if self.USE_UI:
            pygame.init()
            pygame.display.set_caption("Polyline")

    def single_game(self, black_starts: bool = True) -> int:
        game = Game(board_size=self.BOARD_SIZE, black_starts=black_starts, strat=self.STRAT, use_ui=self.USE_UI)
        game.print_game_info([self.BOARD_SIZE, self.STRAT, self.GAME_COUNT])

        while game.winner is None:
            game.play()

        game_winner = game.winner
        game_data = {
            'board_size': self.BOARD_SIZE,
            'player1_strategy': self.STRAT[0],
            'player2_strategy': self.STRAT[1],
            'winner': player2str[game_winner]
        }
        self.game_results.append(game_data)

        print(f"{player2str[game_winner]} player ({self.STRAT[game_winner-1]}) wins!")
        return game_winner

    def save_results_to_csv(self):
        df_results = pd.DataFrame(self.game_results)  # Modification ici pour inclure tous les résultats
        if os.path.exists('tournament_results.csv') and os.path.getsize('tournament_results.csv') > 0:
            df_existing_results = pd.read_csv('tournament_results.csv')
            df_final_results = pd.concat([df_existing_results, df_results], ignore_index=True)
        else:
            df_final_results = df_results

        df_final_results.to_csv('tournament_results.csv', index=False)

    def championship(self):
        scores = [0, 0]

        for _ in range(self.N_GAMES):
            self.GAME_COUNT = _
            winner = self.single_game(black_starts=self.GAME_COUNT < self.N_GAMES / 2)
            scores[winner-1] += 1

        log = logging.getLogger("rich")
        log.info("Design your own evaluation measure!")
        print(scores)

        self.save_results_to_csv()

    def get_initial_board_state(self):
        return [[0 for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
