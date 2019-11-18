"""

"""
from __future__ import print_function
import sys

sys.path.append('..')
from Game import Game
from .OrganismLogic import OrganismState
import numpy as np


class OrganismGame(Game):

	def __init__(self, rings, players):
		"""
		Input:
			rings: List of the names of rings on the board
		"""
		self.rings = rings
		self.n_rings = len(rings)
		self.players = players

	def getInitBoard(self):
		"""
		Returns:
			Initial state of the board, represented as an array
		"""
		state = OrganismState(self.rings, self.players)
		return state.get_tensor_form()

	def getBoardSize(self):
		"""
		Returns:
			(x, y): a tuple of board dimensions
		"""
		return (2 * self.n_rings - 1, 2 * self.n_rings - 1)

	def getActionSize(self):
		"""
		Returns:
			actionSize: number of all possible actions
		"""
		return 24 * (2 * self.n_rings - 1) ** 2

	def getNextState(self, state, player, action):
		"""
		Input:
			state: current state
			player: current player (1 or -1)
			action: action taken by current player

		Returns:
			nextState: state after applying action
			nextPlayer: player who plays in the next turn
		"""
		next_state = state.copy()
		next_state.execute_action(action, player)

		return (next_state.get_tensor_form(), next_state.get_next_player())

	def getValidMoves(self, state, player):
		"""
		Input:
			state: current state
			player: current player

		Returns:
			validMoves: a binary vector of length self.getActionSize(), 1 for
						moves that are valid from the current state and player,
						0 for invalid moves
		"""
		choices_list = state.choices_for(player)

		# TODO: convert the list of choices to vector form of validMoves

		return state.choices_for(player)

	def getGameEnded(self, state, player):
		"""
		Input:
			state: current state
			player: current player (1 or -1)

		Returns:
			r: 0 if game has not ended. 1 if player won, -1 if player lost
		"""
		return state.get_game_ended(player)

	def getCanonicalForm(self, state, player):
		"""
		Input:
			state: current board state
			player: current player (1 or -1)

		Returns:
			canonicalBoard: returns canonical form of state. The canonical form
							should be independent of player. For e.g. in chess,
							the canonical form can be chosen to be from the pov
							of white. When the player is white, we can return
							state as is. When the player is black, we can invert
							the colors and return the state.
		"""
		state_tensor = state.get_tensor_form()

		if player == -1:
			state_tensor = self._flip_board_state(state_tensor)

		return state_tensor

	def getSymmetries(self, state, pi):
		"""
		Input:
			state: current board state
			pi: policy vector of size self.getActionSize()

		Returns:
			symmForms: a list of [(board,pi)] where each tuple is a symmetrical
			form of the board and the corresponding pi vector. This
			is used when training the neural network from examples.
		"""
		symmetric_forms = []

		# TODO: get rotational and reflectional forms of current board and
		#   policy

		return symmetric_forms

	def stringRepresentation(self, state):
		"""
		Input:
			state: current board state

		Returns:
			boardString: a quick conversion of board to a string format.
						Required by MCTS for hashing.
		"""
		boardString = state.get_tensor_form().tostring()
		return boardString

	def _flip_board_state(self, state_tensor):
		pass
