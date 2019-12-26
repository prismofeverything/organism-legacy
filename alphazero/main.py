from Coach import Coach
from organism.OrganismGame import OrganismGame as Game
from organism.pytorch.NNet import NNetWrapper as nn
from utils import *
import numpy as np
import torch

args = dotdict({
    'numIters': 1000,
    'numEps': 250,
    'tempThreshold': 45,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 50,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('temp','checkpoint_1.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

SEED = 229

if __name__=="__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    g = Game(n_rings=4, initial_food=[5, 3, 1, 0], n_tokens_to_win=5, n_organisms_to_win=3)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
