import Arena
from organism.OrganismGame import OrganismGame
from organism.pytorch.NNet import NNetWrapper as NNet

from dotdict import dotdict

"""
use this script to play any two agents against each other, or play manually
with any agent.
"""
n_rings = 3
initial_food = [3, 2, 1]
n_tokens_to_win = 3
n_organisms_to_win = 2

g = OrganismGame(n_rings=n_rings, initial_food=initial_food,
                 n_tokens_to_win=n_tokens_to_win, n_organisms_to_win=n_organisms_to_win)

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./small_size', 'best.pth.tar')
n2 = NNet(g)
n2.load_checkpoint('./small_size', 'best.pth.tar')

args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0, 'softmaxTempPitting': 0.35,
                'longGameThreshold': 50, 'tempThresholdPitting': 20})

arena = Arena.Arena(n1, n2, g, args)
arena.playGames(10, verbose=True)
