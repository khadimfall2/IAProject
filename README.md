Pour lancer le projet de jeu de Hex avec des stratégies d'intelligence artificielle, suivez ces étapes simples :
Activer l'environnement virtuel Python Ouvrez un terminal et naviguez jusqu'au répertoire du projet sur votre bureau en utilisant la commande cd. Une fois dans le répertoire du projet, activez l'environnement virtuel en exécutant :
shell
source mon_env_python3.10/bin/activate
Vous remarquerez que le nom de l'environnement virtuel (mon_env_python3.10) apparaît avant le prompt de votre terminal, indiquant que l'environnement est actif.
Lancer le jeu Toujours dans le terminal, lancez le jeu en exécutant la commande suivante :
shell
python3.10 main.py --size 10 --games 10 --player qlearning --other random
Cette commande démarre le jeu avec les paramètres suivants :
--size 10 définit la taille du plateau à 10x10.
--games 10 spécifie que 10 parties seront jouées.
--player qlearning sélectionne la stratégie Q-Learning pour le joueur principal.
--other random définit le joueur adverse pour utiliser la stratégie Random.
Observer les résultats Après le lancement, le jeu va automatiquement jouer 10 parties en utilisant les stratégies spécifiées. Les résultats de chaque partie seront affichés dans le terminal, indiquant lequel des joueurs (White ou Black) a gagné.
Une fois que vous avez terminé de jouer ou d'effectuer des tests, vous pouvez désactiver l'environnement virtuel en tapant deactivate dans le terminal.
Assurez-vous que tous les paquets nécessaires sont installés dans votre environnement virtuel et que votre fichier main.py est correctement configuré pour utiliser les options de ligne de commande.
