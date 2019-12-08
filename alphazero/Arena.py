import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
from MCTS import MCTS
import time

TIE_THRESHOLD = 60
TEMP_THRESHOLD = 20

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, args, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.args = args
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        verbose = True

        player1_mcts = MCTS(self.game, self.player1, self.args)
        player2_mcts = MCTS(self.game, self.player2, self.args)
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer)==0 and it < TIE_THRESHOLD:
            it+=1

            if verbose:
                self.game._get_board_from_state(board).draw('%d.png' % (it))

            canonicalForm = self.game.getCanonicalForm(board, curPlayer)

            temp = int(it <= TEMP_THRESHOLD)*0.7

            if curPlayer == 1:
                pi = player1_mcts.getActionProb(canonicalForm, it - 1, temp=temp)
            else:
                pi = player2_mcts.getActionProb(canonicalForm, it - 1, temp=temp)

            action = np.random.choice(len(pi), p=pi)

            valid_moves, next_states = self.game.getValidMoves(board, curPlayer)
            board, curPlayer = next_states[action]

        if verbose:
            self.game._get_board_from_state(board).draw('%d.png' % (it))

        if it < TIE_THRESHOLD:
            print("Result ", str(self.game.getGameEnded(board, 1)))
            return self.game.getGameEnded(board, 1)
        else:  # Draws
            print("Result Draw")
            return 0

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player1
        
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult==-1:
                oneWon+=1                
            elif gameResult==1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
            
        bar.finish()

        return oneWon, twoWon, draws
