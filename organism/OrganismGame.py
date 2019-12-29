from __future__ import print_function
import sys

sys.path.append('..')
from alphazero.Game import Game
from organism.OrganismLogic import OrganismBoard, OrganismTree, OrganismTurn
import numpy as np
import copy

RINGS = ['red', 'orange', 'green', 'blue', 'purple']

PLAYER_NAMES = {
	1: 'CS',
	-1: 'YH',
}
PLAYER_INDEXES = {
	'CS': 1,
	'YH': -1,
}

EAT = 'EAT'
MOVE = 'MOVE'
GROW = 'GROW'
CIRCULATE = 'CIRCULATE'


class OrganismGame(Game):

	def __init__(self, n_rings=5, initial_food=[5, 3, 1, 1, 0], n_tokens_to_win=7, n_organisms_to_win=3):
		"""
		Input:
			rings: List of the names of rings on the board
		"""
		assert n_rings == len(initial_food)
		assert n_rings >= 3

		self.n_rings = n_rings
		self.rings = RINGS[:self.n_rings]
		self.initial_food = initial_food
		self.n_tokens_to_win = n_tokens_to_win
		self.n_organisms_to_win = n_organisms_to_win

	def getInitBoard(self):
		"""
		Returns:
			Initial state of the board in array form
		"""
		board = OrganismBoard(
			self.rings, n_tokens_to_win=self.n_tokens_to_win,
			n_organisms_to_win=self.n_organisms_to_win
		)

		food_dict = {self.rings[i]: self.initial_food[i] for i in range(self.n_rings)}
		board.initialize_food(food_dict)

		board.place_element((self.rings[-1], 6*(self.n_rings - 1) - 1), 'CS', EAT)
		board.place_element((self.rings[-1], 0), 'CS', MOVE)
		board.place_element((self.rings[-1], 1), 'CS', GROW)

		board.add_food((self.rings[-1], 6*(self.n_rings - 1) - 1), 1)
		board.add_food((self.rings[-1], 0), 1)
		board.add_food((self.rings[-1], 1), 1)

		board.place_element((self.rings[-1], 3*(self.n_rings - 1) - 1), 'YH', EAT)
		board.place_element((self.rings[-1], 3*(self.n_rings - 1)), 'YH', MOVE)
		board.place_element((self.rings[-1], 3*(self.n_rings - 1) + 1), 'YH', GROW)

		board.add_food((self.rings[-1], 3*(self.n_rings - 1) - 1), 1)
		board.add_food((self.rings[-1], 3*(self.n_rings - 1)), 1)
		board.add_food((self.rings[-1], 3*(self.n_rings - 1) + 1), 1)

		board.set_current_player('CS')

		return board.get_array_form('CS')

	def getBoardSize(self):
		"""
		Returns:
			(x, y): a tuple of board dimensions
		"""
		return 2 * self.n_rings - 1, 2 * self.n_rings - 1

	def getActionSize(self, state, player):
		"""
		Returns:
			actionSize: number of all possible actions
		"""
		valid_moves = self.getValidMoves(state, player)
		return len(valid_moves)

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
		board = self.get_board_from_state(state)

		organism_index, move = action
		player_name = PLAYER_NAMES[player]
		next_board = copy.deepcopy(board)

		organisms = next_board.find_organisms()
		turn = OrganismTurn(next_board, organisms, player_name, organism_index)
		turn.take_turn(move)
		turn.apply_actions(next_board)

		return next_board.get_array_form('CS'), PLAYER_INDEXES[next_board.current_player]

	def getValidMoves(self, state, player):
		"""
		Input:
			state: current state
			player: current player

		Returns:
			validMoves: List of tuples (organism_index, move) that can be used
			by getNextState().
		"""
		board = self.get_board_from_state(state)
		player_name = PLAYER_NAMES[player]

		organisms = board.find_organisms()
		player_organisms = organisms[player_name]

		valid_moves = []

		for index, organism in player_organisms.items():
			organism_is_movable = False

			for movable_organism in board.organisms_to_move:
				if set(organism) == set(movable_organism):
					organism_is_movable = True
					break

			if organism_is_movable:
				hashed_moves = set()
				unique_moves = []
				tree = OrganismTree(board, organisms, player_name, index)
				walk = tree.walk()

				for m in walk:
					# Remove duplicate moves that lead to the same state
					if m[1] not in hashed_moves:
						unique_moves.append((index, m))
						hashed_moves.add(m[1])

				valid_moves.extend(unique_moves)

		return valid_moves

	def getGameEnded(self, state, player):
		"""
		Input:
			state: current state
			player: current player (1 or -1)

		Returns:
			r: 0 if game has not ended. 1 if player won, -1 if player lost
		"""
		board = self.get_board_from_state(state)

		winners = board.find_winners()

		if len(winners) == 0:
			# Player loses if the action size is zero
			if self.getActionSize(state, player) == 0:
				return -1

			return 0
		elif len(winners) == 1:
			if winners[0] == PLAYER_NAMES[player]:
				return 1
			else:
				return -1
		else:  # Tie? Current player wins game.
			if board.current_player == PLAYER_NAMES[player]:
				return 1
			else:
				return -1

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
		if player == -1:
			state = self._flip_board_state(state)

		return state

	def getSymmetries(self, state):
		"""
		Input:
			state: current board state

		Returns:
			symmForms: a list of arrays where each element is a symmetrical
			form of the state. This is used when training the neural network
			from examples.
		"""
		symmetric_forms = []

		temp_board = OrganismBoard(self.rings)
		pos_to_axial_coords = temp_board.space_to_axial_coords
		axial_coords_to_pos = temp_board.axial_coords_to_space

		# Add all rotations
		symmetric_forms.extend(self._get_rotated_states(state, axial_coords_to_pos, pos_to_axial_coords))
		reflected_state = np.transpose(state, (0, 2, 1))
		symmetric_forms.extend(
			self._get_rotated_states(reflected_state, axial_coords_to_pos, pos_to_axial_coords))

		return symmetric_forms

	def _get_rotated_states(self, state, axial_coords_to_pos, pos_to_axial_coords):
		"""
		Returns the six rotations of the given state.
		"""
		rotated_forms = []

		for rot in range(6):
			rotated_state = np.zeros_like(state)
			rotated_state[7:9, :, :] = state[7:9, :, :]
			rotated_state[10, :, :] = state[10, :, :]

			# Fix center point
			rotated_state[:, self.n_rings - 1, self.n_rings - 1] = state[:, self.n_rings - 1, self.n_rings - 1]

			for i in range(rotated_state.shape[0]):
				for j in range(rotated_state.shape[0]):
					if (i, j) in axial_coords_to_pos:
						pos = axial_coords_to_pos[(i, j)]
						ring_index = self.rings.index(pos[0])

						if ring_index == 0:  # Center point
							continue

						new_pos = (pos[0], (pos[1] - ring_index*rot) % (6*ring_index))
						new_axial_coords = pos_to_axial_coords[new_pos]

						rotated_state[:, new_axial_coords[0], new_axial_coords[1]] = state[:, i, j]

			rotated_forms.append(rotated_state)

		return rotated_forms

	def stringRepresentation(self, state):
		"""
		Input:
			state: current board state

		Returns:
			boardString: a quick conversion of board to a string format.
						Required by MCTS for hashing.
		"""
		boardString = state.tostring()
		return boardString

	def is_first_move(self, state):
		"""
		Returns True if the next move by the current player is the first move
		of the player's turn.
		"""
		board = self.get_board_from_state(state)
		organisms = board.find_organisms()

		if board.current_player in organisms:
			n_player_organisms = len(organisms[board.current_player])
		else:
			n_player_organisms = 0
		n_movable_organisms = len(board.organisms_to_move)

		return n_player_organisms == n_movable_organisms

	def get_board_from_state(self, state):
		"""
		Returns the board representation given a state array.
		"""
		board = OrganismBoard(self.rings, n_tokens_to_win=self.n_tokens_to_win,
			n_organisms_to_win=self.n_organisms_to_win)

		board.fill_from_array(state, ['CS', 'YH'])

		return board

	def _flip_board_state(self, state):
		"""
		Flips the board state to reverse players one and two.
		"""
		flipped_state = np.zeros_like(state)

		# Flip element positions
		flipped_state[0:3, :, :] = state[3:6, :, :]
		flipped_state[3:6, :, :] = state[0:3, :, :]

		# Food stays same
		flipped_state[6, :, :] = state[6, :, :]

		# Flip scores
		flipped_state[7, :, :] = state[8, :, :]
		flipped_state[8, :, :] = state[7, :, :]

		# Playable organisms stay same
		flipped_state[9, :, :] = state[9, :, :]

		# Current player flipped
		flipped_state[10, :, :] = -state[10, :, :]

		return flipped_state


def test_game():
	n_rings = 3
	initial_food = [3, 2, 1]
	n_tokens_to_win = 3
	n_organisms_to_win = 2

	game = OrganismGame(n_rings=n_rings, initial_food=initial_food,
	    n_tokens_to_win=n_tokens_to_win, n_organisms_to_win=n_organisms_to_win)
	init_state = game.getInitBoard()

	board = game.get_board_from_state(init_state)
	board.draw(filename='test.png')
	print(init_state)
	print(game.getGameEnded(init_state, 1))

	valid_moves = game.getValidMoves(init_state, 1)
	print(len(valid_moves))

	state, player = game.getNextState(init_state, 1, valid_moves[0])

	print(state)
	print(game.getGameEnded(state, 1))

	board = game.get_board_from_state(state)

	valid_moves = game.getValidMoves(state, -1)
	print(valid_moves)

	symmetric_forms = game.getSymmetries(state)
	print(symmetric_forms)


if __name__ == '__main__':
	test_game()