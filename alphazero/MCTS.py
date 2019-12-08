import math
import numpy as np
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, softmax_temp=0.7):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.softmax_temp = softmax_temp
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, turn_index, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, turn_index)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, turn_index, a)] if (s, turn_index, a) in self.Nsa else 0 for a in range(self.game.getActionSize(canonicalBoard, 1))]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA] = 1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs


    def search(self, canonicalBoard, turn_index):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        is_first_move = self.game.is_first_move(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            if is_first_move:
                return -self.Es[s]
            else:
                return self.Es[s]

        if (s, turn_index) not in self.Ps:
            # leaf node
            v = self.nnet.predict(canonicalBoard)

            # Calculate Ps's by the softmax of next v's
            valid_moves, next_states = self.game.getValidMoves(canonicalBoard, 1)
            next_vs = np.zeros(len(valid_moves))

            for i in range(len(valid_moves)):
                next_state, next_player = next_states[i]
                canonical_state = self.game.getCanonicalForm(next_state, next_player)
                next_vs[i] = (-1)**(next_player == -1)*self.nnet.predict(canonical_state)

            next_vs_exp = np.exp(next_vs/self.softmax_temp)
            self.Ps[(s, turn_index)] = next_vs_exp/next_vs_exp.sum()

            self.Vs[(s, turn_index)] = valid_moves
            self.Ns[(s, turn_index)] = 0

            if is_first_move:
                return -v
            else:
                return v

        valid_moves = self.Vs[(s, turn_index)]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(len(valid_moves)):
            if (s, turn_index, a) in self.Qsa:
                u = self.Qsa[(s, turn_index, a)] + self.args.cpuct*self.Ps[(s, turn_index)][a]*math.sqrt(self.Ns[(s, turn_index)])/(1+self.Nsa[(s, turn_index, a)])
            else:
                u = self.args.cpuct*self.Ps[(s, turn_index)][a]*math.sqrt(self.Ns[(s, turn_index)] + EPS)     # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, valid_moves[a])
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, turn_index + 1)

        if (s, turn_index, a) in self.Qsa:
            self.Qsa[(s, turn_index, a)] = (self.Nsa[(s, turn_index, a)]*self.Qsa[(s, turn_index, a)] + v)/(self.Nsa[(s, turn_index, a)]+1)
            self.Nsa[(s, turn_index, a)] += 1

        else:
            self.Qsa[(s, turn_index, a)] = v
            self.Nsa[(s, turn_index, a)] = 1

        self.Ns[(s, turn_index)] += 1

        if is_first_move:
            return -v
        else:
            return v
