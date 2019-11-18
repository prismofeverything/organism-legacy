"""

"""

import numpy as np

def build_rows(rings):
	rows = []
	for ring in np.arange(rings):
		flank = np.arange(ring + 1, rings)
		pad = np.full(ring // 2, PAD)
		row = np.concatenate([
			np.full(ring // 2, PAD),
			np.flip(flank, 0),
			np.full(ring + 1, ring),
			flank,
			np.full(int(round(ring / 2.0)), PAD)])
		rows.append(row)

	return np.concatenate([
		np.flip(rows[1:], 0),
		rows])


def ring_length(ring):
	if ring == 0:
		return 1
	else:
		return ring * 6


def add_adjacency(adjacencies, source, destination):
	if not source in adjacencies:
		adjacencies[source] = []
	adjacencies[source].append(destination)


def symmetric_adjacency(adjacencies, source, destination):
	add_adjacency(adjacencies, source, destination)
	add_adjacency(adjacencies, destination, source)


def build_rings(rings):
	adjacencies = {}

	for ring_index, ring in enumerate(rings):
		length = ring_length(ring_index)
		for step in np.arange(length):
			space = (ring, step)

			if ring_index > 0:
				ratio = (step / float(ring_index)) * (ring_index - 1)
				inner = (rings[ring_index - 1], int(np.ceil(ratio)))
				symmetric_adjacency(adjacencies, space, inner)

				corner = step % ring_index == 0
				if not corner:
					other = (rings[ring_index - 1], int(np.floor(ratio)))
					symmetric_adjacency(adjacencies, space, other)

				symmetric_adjacency(adjacencies, space,
				                    (ring, (step + 1) % length))

	return adjacencies


class OrganismBoard():
	def __init__(self, rings):
		self.rings = rings
		self.rows = build_rows(len(rings))
		self.adjacencies = build_rings(rings)

	def spaces(self):
		return self.adjacencies.keys()


class OrganismState(object):
	def __init__(self, rings, players):
		self.board = OrganismBoard(rings)
		self.spaces = {
			space: {
				element: None,
				food: 0}
			for space in self.board.spaces()}

		self.players = players

	def choices_for(self, player):
		"""
		Input:
			player: current player (1 or -1)

		Returns:
			List of legal moves that can be made by the current player
		"""
		pass

	def execute_move(self, move):
		"""
		Change the state given a certain move.
		"""
		pass

	def get_game_ended(self, player):
		"""
		Input:
			player: current player (1 or -1)

		Returns:
			r: 0 if game has not ended. 1 if player won, -1 if player lost
		"""
		pass


def test_organism():
	board = OrganismBoard([
		'red',
		'orange',
		'green',
		'blue',
		'purple'])

	print(board.rows)
	print(board.adjacencies)

if __name__ == '__main__':
	test_organism()
