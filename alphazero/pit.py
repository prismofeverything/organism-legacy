import Arena
from organism.OrganismGame import OrganismGame
from organism.OrganismPlayers import *
from organism.pytorch.NNet import NNetWrapper as NNet

from utils import *

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
n1.load_checkpoint('./models', 'small_checkpoint_1.pth.tar')
n2 = NNet(g)
n2.load_checkpoint('./models', 'small_checkpoint_1.pth.tar')

args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})

arena = Arena.Arena(n1, n2, g, args)
arena.playGames(10, verbose=True)
