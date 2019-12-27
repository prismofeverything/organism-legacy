from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThresholdTraining, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            if episodeStep >= self.args.longGameThreshold:
                # Restart long games to reduce self-play time and remove
                # confounding training examples
                self.mcts = MCTS(self.game, self.nnet, self.args)
                trainExamples = []
                board = self.game.getInitBoard()
                self.curPlayer = 1
                episodeStep = 0

            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThresholdTraining)

            pi = self.mcts.getActionProb(canonicalBoard, episodeStep, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard)
            for b in sym:
                trainExamples.append([b, self.curPlayer, None])

            action = np.random.choice(len(pi), p=pi)
            valid_moves = self.game.getValidMoves(board, self.curPlayer)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, valid_moves[action])

            r = self.game.getGameEnded(board, self.curPlayer)

            episodeStep += 1

            if r!=0:
                canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
                sym = self.game.getSymmetries(canonicalBoard)
                for b in sym:
                    trainExamples.append([b, self.curPlayer, None])
                return [(x[0], r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples], episodeStep

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')

            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
    
                eps_time = AverageMeter()
                n_turns = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()
    
                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                    episodeTrainExamples, episodeStep = self.executeEpisode()
                    iterationTrainExamples += episodeTrainExamples
    
                    # bookkeeping + plot progress
                    cur_eps_time = time.time() - end
                    eps_time.update(cur_eps_time)
                    n_turns.update(episodeStep)
                    end = time.time()
                    bar.suffix = '({eps}/{maxeps}) Time: {cet}s | Avg Turns: {avg_turn:.1f} | Avg Time: {et}s | Total: {total:} | ETA: {eta:}'.format(
                        eps=eps+1, maxeps=self.args.numEps, n_turn=episodeStep,
                        avg_turn=n_turns.avg, et=int(eps_time.avg),
                        cet=int(cur_eps_time), total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)
            
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            
            self.nnet.train(trainExamples)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(self.pnet, self.nnet, self.game, self.args)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
