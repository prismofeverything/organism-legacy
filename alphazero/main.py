from Coach import Coach
from organism.OrganismGame import OrganismGame as Game
from organism.pytorch.NNet import NNetWrapper as nn
from dotdict import dotdict
import numpy as np
import torch

args = dotdict({
    'numIters': 1000,  # Total number of self-play iterations
    'numEps': 100,  # Number of self-play games per iteration
    'tempThresholdTraining': 30,  # MCTS temperature set to zero after this many moves during training
    'tempThresholdPitting': 20,  # MCTS temperature set to zero after this many moves during pitting
    'softmaxTempTraining': 0.35,  # Softmax temperature used in MCTS during training
    'softmaxTempPitting': 0.35,  # Softmax temperature used in MCTS during pitting
    'updateThreshold': 0.6,  # Minimum win rate to update best agent
    'maxlenOfQueue': 200000,  # Maximum number of training examples to keep
    'numMCTSSims': 50,  # Number of MCTS sims to run per turn
    'longGameThreshold': 50,  # Self-play games are reset and pitting games are considered a tie after this many moves
    'arenaCompare': 50,  # Number of games played to determine best agent
    'cpuct': 1,

    'checkpoint': './small_size',
    'load_model': True,
    'load_folder_file': ('small_size', 'checkpoint_6.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

SEED = 229

if __name__=="__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    g = Game(n_rings=3, initial_food=[3, 1, 0], n_tokens_to_win=3, n_organisms_to_win=2)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
